"""Fetch an arXiv LaTeX source package and turn it into model-ready text plus figures.

The module never runs TeX or any command found in the source. It statically
expands the document body, resolves every ``\\includegraphics`` to a file in the
package, and refuses (``SourceUnavailable``) whenever it cannot show that the
figures it found are the figures the paper draws. Callers are expected to fall
back to the PDF in that case.

TeX cannot be proven complete by static analysis, so ``LatexSource.manifest``
only lists what was actually read and resolved, plus the constructs that are
known not to be evaluated. It does not claim semantic completeness.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
import threading
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from .models import ArxivPaper

logger = logging.getLogger(__name__)

_SOURCE_URL = "https://arxiv.org/src/{arxiv_id}v{version}"
_ID_PATTERN = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})$")

# arXiv asks automated clients to stay below one request every 3 seconds.
_MIN_DOWNLOAD_INTERVAL_SECONDS = 3.0
_REQUEST_TIMEOUT_SECONDS = 120
_MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
_MAX_EXPANDED_BYTES = 200 * 1024 * 1024
_MAX_MEMBERS = 2000
_MAX_TEX_FILE_BYTES = 5 * 1024 * 1024
_MAX_TEXT_CHARS = 3_000_000
_MAX_INCLUDE_DEPTH = 20

# pdfTeX search order, with .eps last for sources written for dvips.
_IMAGE_EXTENSIONS = (".png", ".pdf", ".jpg", ".jpeg", ".mps", ".jbig2", ".jb2", ".eps")

_SOURCE_PACKAGE_LOCK = threading.Lock()
_last_download_finished: float | None = None


class SourceUnavailable(Exception):
    """The LaTeX source cannot be used; the caller should fall back to the PDF."""


@dataclass
class SourceFigure:
    path: Path
    # Root-relative POSIX path of the image. The same string follows the matching
    # \includegraphics in LatexSource.text as "[[IMAGE: <reference>]]".
    reference: str
    caption: str


@dataclass
class LatexSource:
    text: str
    figures: list[SourceFigure]
    manifest: dict


# ---------------------------------------------------------------------------
# Download and cache
# ---------------------------------------------------------------------------

def _new_client() -> httpx.Client:
    return httpx.Client(
        timeout=_REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": "arxiv-daily-papers-workflow source fetcher"},
    )


def _fetch_archive(url: str) -> bytes:
    try:
        with _new_client() as client, client.stream("GET", url) as resp:
            if resp.status_code == 429:
                # No retry and no parallel workaround: the caller falls back to the PDF.
                raise SourceUnavailable(f"arXiv rate limited the source request: {url}")
            if resp.status_code != 200:
                raise SourceUnavailable(f"arXiv source request returned HTTP {resp.status_code}: {url}")
            host = resp.url.host or ""
            if resp.url.scheme != "https" or not (host == "arxiv.org" or host.endswith(".arxiv.org")):
                raise SourceUnavailable(f"source request was redirected away from arXiv: {resp.url}")
            if "html" in resp.headers.get("content-type", "").lower():
                raise SourceUnavailable("arXiv returned an HTML page instead of a source package")
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > _MAX_ARCHIVE_BYTES:
                raise SourceUnavailable("source package exceeds the download size limit")
            buffer = bytearray()
            for chunk in resp.iter_bytes():
                buffer.extend(chunk)
                if len(buffer) > _MAX_ARCHIVE_BYTES:
                    raise SourceUnavailable("source package exceeds the download size limit")
            return bytes(buffer)
    except httpx.HTTPError as exc:
        raise SourceUnavailable(f"source download failed: {exc!r}") from exc


def _download_archive(url: str) -> bytes:
    """Download serially for the whole process, at least the minimum interval apart."""
    global _last_download_finished
    with _SOURCE_PACKAGE_LOCK:
        if _last_download_finished is not None:
            wait = _MIN_DOWNLOAD_INTERVAL_SECONDS - (time.monotonic() - _last_download_finished)
            if wait > 0:
                time.sleep(wait)
        try:
            return _fetch_archive(url)
        finally:
            _last_download_finished = time.monotonic()


def _gunzip_limited(data: bytes) -> bytes:
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        # One byte over the limit is enough to detect a decompression bomb.
        out = decompressor.decompress(data, _MAX_EXPANDED_BYTES + 1)
    except zlib.error as exc:
        raise SourceUnavailable(f"corrupt gzip stream: {exc}") from exc
    if len(out) > _MAX_EXPANDED_BYTES:
        raise SourceUnavailable("source package expands beyond the size limit")
    if not decompressor.eof:
        raise SourceUnavailable("truncated gzip stream")
    if decompressor.unused_data:
        # A concatenated gzip stream would silently lose every member after the first.
        raise SourceUnavailable("gzip stream has trailing or concatenated data")
    return out


_TEX_SIGNATURE = re.compile(rb"\\(?:documentclass|begin\s*\{document\}|bye)(?![a-zA-Z@])")


def _unpack(data: bytes, dest: Path) -> None:
    raw = _gunzip_limited(data) if data[:2] == b"\x1f\x8b" else data
    head = raw[:2048].lstrip().lower()
    if raw.startswith(b"%PDF-"):
        raise SourceUnavailable("source package is only a PDF")
    if head.startswith((b"<!doctype", b"<html", b"<?xml", b"<head", b"<body")):
        raise SourceUnavailable("source package is an HTML page")
    try:
        archive = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
    except tarfile.ReadError:
        archive = None
    if archive is None:
        if b"\x00" in raw[:8192] or raw.startswith(b"%!PS") or not _TEX_SIGNATURE.search(raw):
            raise SourceUnavailable("source package is neither a tar archive nor a single TeX file")
        (dest / "main.tex").write_bytes(raw)
        return
    with archive:
        _extract_tar(archive, dest)


def _extract_tar(archive: tarfile.TarFile, dest: Path) -> None:
    total = 0
    count = 0
    dest_root = dest.resolve()
    for member in archive:
        count += 1
        if count > _MAX_MEMBERS:
            raise SourceUnavailable("source package has too many members")
        parts = [p for p in PurePosixPath(member.name).parts if p != "."]
        if (
            not parts
            or member.name.startswith("/")
            or ".." in parts
            or "\\" in member.name
            or "\x00" in member.name
        ):
            raise SourceUnavailable(f"unsafe member path in source package: {member.name!r}")
        target = dest.joinpath(*parts)
        if not target.resolve().is_relative_to(dest_root):
            raise SourceUnavailable(f"member escapes the extraction directory: {member.name!r}")
        try:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isreg():
                if target.exists():
                    raise SourceUnavailable(f"duplicate member in source package: {member.name!r}")
                total += member.size
                if total > _MAX_EXPANDED_BYTES:
                    raise SourceUnavailable("source package expands beyond the size limit")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                with target.open("wb") as out:
                    shutil.copyfileobj(source, out)
            else:
                # Symlinks, hard links, devices and FIFOs are never created.
                raise SourceUnavailable(f"unsupported member type in source package: {member.name!r}")
        except OSError as exc:
            raise SourceUnavailable(f"cannot extract {member.name!r}: {exc}") from exc
    if count == 0:
        raise SourceUnavailable("source package is empty")


def _cache_is_valid(cache_dir: Path) -> bool:
    return (cache_dir / "source").is_dir() and (cache_dir / "download.json").is_file()


def _fetch_into_cache(paper: ArxivPaper, cache_dir: Path, url: str) -> None:
    parent = cache_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    data = _download_archive(url)
    # Extract beside the final directory and rename, so an interrupted run can
    # never leave a half-extracted directory that looks like a valid cache entry.
    staging = parent / f".tmp-{uuid.uuid4().hex}"
    try:
        (staging / "source").mkdir(parents=True)
        _unpack(data, staging / "source")
        (staging / "download.json").write_text(
            json.dumps(
                {
                    "arxiv_id": paper.arxiv_id,
                    "version": paper.version,
                    "source_url": url,
                    "archive_bytes": len(data),
                    "archive_sha256": hashlib.sha256(data).hexdigest(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if cache_dir.exists() and not _cache_is_valid(cache_dir):
            shutil.rmtree(cache_dir)  # Leftover of an older failed run, rebuilt from arXiv.
        try:
            os.replace(staging, cache_dir)
        except OSError:
            if not _cache_is_valid(cache_dir):
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


async def prepare_latex_source(paper: ArxivPaper, storage_dir: Path) -> LatexSource:
    if not _ID_PATTERN.match(paper.arxiv_id) or paper.version < 1:
        raise SourceUnavailable(f"unsupported arXiv identifier: {paper.arxiv_id!r} v{paper.version}")
    url = _SOURCE_URL.format(arxiv_id=paper.arxiv_id, version=paper.version)
    slug = paper.arxiv_id.replace("/", "_")
    cache_dir = storage_dir / "latex_source" / f"{slug}v{paper.version}"
    if not _cache_is_valid(cache_dir):
        await asyncio.to_thread(_fetch_into_cache, paper, cache_dir, url)
    source = await asyncio.to_thread(load_latex_source, cache_dir / "source")
    source.manifest["download"] = json.loads((cache_dir / "download.json").read_text(encoding="utf-8"))
    return source


# ---------------------------------------------------------------------------
# Static TeX reading
# ---------------------------------------------------------------------------

_VERBATIM = re.compile(r"\\begin\{(verbatim\*?|Verbatim|lstlisting|minted)\}.*?\\end\{\1\}", re.S)
_COMMENT = re.compile(r"(^|[^\\])((?:\\\\)*)%[^\n]*", re.M)
_COMMENT_ENV = re.compile(r"\\begin\{comment\}.*?\\end\{comment\}", re.S)
_IFFALSE = re.compile(r"\\iffalse(?![a-zA-Z@])")
_IF_TOKEN = re.compile(r"\\(if[a-zA-Z@]*|else|fi)(?![a-zA-Z@])")
# \iff is a math symbol and \ifthenelse is a macro; neither opens a TeX conditional.
_NOT_CONDITIONALS = {"iff", "ifthenelse"}

_INPUT_ONE_ARG = re.compile(r"\\(input|include|subfile)(?![a-zA-Z@])\s*(?:\{([^{}]*)\}|([^\s{}\\%]+))")
_INPUT_TWO_ARGS = re.compile(r"\\(import|subimport)\*?\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_INPUT_TOKEN = re.compile(r"\\(?:input|include|subfile|import|subimport)(?![a-zA-Z@])")

_INCLUDEGRAPHICS_TOKEN = re.compile(r"\\includegraphics(?![a-zA-Z@])")
_INCLUDEGRAPHICS = re.compile(r"\\includegraphics\*?\s*(?:\[([^\]]*)\])?\s*\{([^{}]*)\}")
_GRAPHICSPATH = re.compile(r"\\graphicspath\s*\{((?:\s*\{[^{}]*\}\s*)+)\}")
_FLOAT_ENV = re.compile(
    r"\\(begin|end)\s*\{(figure\*?|wrapfigure\*?|subfigure\*?|sidewaysfigure\*?|teaserfigure|SCfigure)\}"
)
_SUBFLOAT = re.compile(r"\\subfloat\*?\s*(?:\[([^\]]*)\])?\s*(?=\{)")
_CAPTION = re.compile(r"\\caption(?:of\s*\{[^{}]*\})?(?![a-zA-Z@])\s*(?:\[[^\]]*\])?\s*(?=\{)")
_LABEL = re.compile(r"\\label\s*\{([^{}]*)\}")
_REFERENCE = re.compile(r"\\(?:auto|c|C|eq|fig|page|name|v|V)?ref\*?\s*\{([^{}]*)\}")
# A float without an image is acceptable only when its content is plainly textual.
_TEXTUAL_FLOAT_CONTENT = re.compile(
    r"\\begin\{(?:tabular\*?|array|algorithmic|algorithm|verbatim\*?|Verbatim|lstlisting|minted|equation\*?|align\*?)\}"
)
_DOCUMENT_BEGIN = re.compile(r"\\begin\s*\{document\}")
_DOCUMENT_END = re.compile(r"\\end\s*\{document\}")
_PLAIN_TEX_END = re.compile(r"\\bye(?![a-zA-Z@])")
_DOCUMENTCLASS = re.compile(r"\\documentclass(?:\[[^\]]*\])?\s*\{([^{}]*)\}")
_STYLE_LOADERS = {
    "usepackage": ".sty", "RequirePackage": ".sty", "RequirePackageWithOptions": ".sty",
    "documentclass": ".cls", "LoadClass": ".cls", "LoadClassWithOptions": ".cls",
}
_STYLE_LOAD_TOKEN = re.compile(r"\\(?:%s)(?![a-zA-Z@])" % "|".join(_STYLE_LOADERS))
_STYLE_LOAD = re.compile(r"\\(%s)(?![a-zA-Z@])\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}" % "|".join(_STYLE_LOADERS))
_BIB_TOKEN = re.compile(r"\\(?:bibliography|addbibresource|addglobalbib|addsectionbib)(?![a-zA-Z@])")
_BIB_COMMAND = re.compile(
    r"\\(bibliography|addbibresource|addglobalbib|addsectionbib)(?![a-zA-Z@])\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}"
)

# Constructs that draw or load content in a way this module does not follow.
_UNSUPPORTED = {
    r"\\begin\{(?:tikzpicture|pgfpicture|pspicture|picture|axis|circuitikz|tikzcd|forest)\}": "generated picture environment",
    r"\\(?:tikz|pgfpicture|pgfimage|pgfdeclareimage)(?![a-zA-Z@])": "generated or pgf image command",
    r"\\(?:epsfig|psfig|epsfbox|includepdf|includesvg|includestandalone|includeinkscape)(?![a-zA-Z@])": "image command other than includegraphics",
    r"\\(?:includeonly|InputIfFileExists|IfFileExists)(?![a-zA-Z@])": "conditional or partial file inclusion",
}
_UNSUPPORTED_COMPILED = [(re.compile(pattern), why) for pattern, why in _UNSUPPORTED.items()]


def _blank(text: str) -> str:
    """Replace text by its newlines only, so line numbers stay aligned with the file."""
    return re.sub(r"[^\n]", "", text)


def _mask_verbatim(text: str) -> str:
    """Return text of equal length in which verbatim bodies are blanked for pattern scans."""
    return _VERBATIM.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _drop_iffalse(text: str) -> str:
    out: list[str] = []
    pos = 0
    while match := _IFFALSE.search(text, pos):
        out.append(text[pos:match.start()])
        depth, in_else, kept_from, closed = 1, False, match.end(), False
        for token in _IF_TOKEN.finditer(text, match.end()):
            name = token.group(1)
            if name == "fi":
                depth -= 1
                if depth == 0:
                    if in_else:
                        # The closing \fi belongs to the removed \iffalse and is dropped.
                        out.append(text[kept_from:token.start()])
                    else:
                        out.append(_blank(text[match.start():token.end()]))
                    pos, closed = token.end(), True
                    break
            elif name == "else":
                if depth == 1 and not in_else:
                    out.append(_blank(text[match.start():token.end()]))
                    kept_from, in_else = token.end(), True
            elif name not in _NOT_CONDITIONALS:
                depth += 1
        if not closed:
            raise SourceUnavailable("unbalanced \\iffalse block")
    out.append(text[pos:])
    return "".join(out)


def _strip_comments(text: str) -> str:
    """Remove % comments, comment environments and \\iffalse blocks, keeping line numbers."""
    pieces: list[str] = []
    cursor = 0
    for verbatim in _VERBATIM.finditer(text):
        pieces.append(_COMMENT.sub(r"\1\2", text[cursor:verbatim.start()]))
        pieces.append(verbatim.group(0))
        cursor = verbatim.end()
    pieces.append(_COMMENT.sub(r"\1\2", text[cursor:]))
    text = "".join(pieces)
    text = _COMMENT_ENV.sub(lambda m: _blank(m.group(0)), text)
    return _drop_iffalse(text)


def _read_group(text: str, open_pos: int) -> tuple[str, int]:
    """Read a balanced {...} group starting at open_pos; return its content and end offset."""
    depth = 0
    i = open_pos
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1:i], i + 1
        i += 1
    raise SourceUnavailable("unbalanced braces in source")


def _is_dynamic_name(name: str) -> bool:
    return not name or any(ch in name for ch in "\\#$%|~^")


@dataclass
class _Region:
    start: int
    end: int
    kind: str
    captions: list[str]
    labels: list[str]


class _Reader:
    """Reads one source tree into text. Instances are single-use."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.parts: list[str] = []
        # Origin (file, line) of every output line, used to report where figures occur.
        self.line_origins: list[tuple[str, int] | None] = [None]
        self.files_read: list[dict] = []
        self.stack: list[Path] = []
        self.hashes: dict[Path, str] = {}

    # -- tree inspection --------------------------------------------------

    def _walk(self) -> list[Path]:
        files: list[Path] = []
        for current, dirs, names in os.walk(self.root, followlinks=False):
            current_path = Path(current)
            dirs[:] = [d for d in sorted(dirs) if d != "__MACOSX"]
            for entry in [*dirs, *names]:
                if (current_path / entry).is_symlink():
                    raise SourceUnavailable(f"symbolic link in source tree: {current_path / entry}")
            files.extend(current_path / n for n in sorted(names) if not n.startswith("._"))
        return files

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _read_text(self, path: Path) -> str:
        if path.stat().st_size > _MAX_TEX_FILE_BYTES:
            raise SourceUnavailable(f"TeX file is too large: {self._rel(path)}")
        data = path.read_bytes()
        self.hashes[path] = hashlib.sha256(data).hexdigest()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _resolve(self, name: str, search_dirs: list[Path], what: str) -> Path:
        name = name.strip().strip('"')
        if _is_dynamic_name(name):
            raise SourceUnavailable(f"{what} uses a dynamic or unsupported file name: {name!r}")
        for base in search_dirs:
            for candidate in (name + ".tex", name):
                path = Path(os.path.normpath(base / candidate))
                if path.is_file() and path.resolve().is_relative_to(self.root):
                    return path
        raise SourceUnavailable(f"{what} refers to a missing file: {name!r}")

    # -- main file selection ----------------------------------------------

    def _input_targets(self, text: str, base: Path) -> list[Path]:
        scan = _mask_verbatim(text)
        targets = []
        for m in _INPUT_ONE_ARG.finditer(scan):
            targets.append((m.group(2) or m.group(3), [self.root, base]))
        for m in _INPUT_TWO_ARGS.finditer(scan):
            directory = Path(m.group(2).strip())
            anchors = [base] if m.group(1) == "subimport" else [self.root]
            targets.append((m.group(3), [anchor / directory for anchor in anchors]))
        resolved = []
        for name, dirs in targets:
            try:
                resolved.append(self._resolve(name, dirs, "include"))
            except SourceUnavailable:
                continue  # Reported precisely when the real expansion reaches it.
        return resolved

    def _find_main(self, files: list[Path]) -> tuple[Path, str]:
        tex_files = [p for p in files if p.suffix.lower() == ".tex"]
        if not tex_files:
            raise SourceUnavailable("source package contains no .tex file")
        readme = self.root / "00README.XXX"
        if readme.is_file():
            for line in self._read_text(readme).splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "toplevelfile":
                    named = self.root / parts[0]
                    if named in tex_files:
                        return named, "latex"
                    raise SourceUnavailable(f"00README.XXX names a missing top-level file: {parts[0]!r}")
        stripped = {p: _strip_comments(self._read_text(p)) for p in tex_files}
        documents = [
            p for p, t in stripped.items()
            if _DOCUMENT_BEGIN.search(t)
            and (m := _DOCUMENTCLASS.search(t)) is not None
            and m.group(1).strip() != "subfiles"
        ]
        if len(documents) == 1:
            return documents[0], "latex"
        if len(documents) > 1:
            referenced = {t for d in documents for t in self._input_targets(stripped[d], d.parent)}
            roots = [d for d in documents if d not in referenced]
            if len(roots) == 1:
                return roots[0], "latex"
            raise SourceUnavailable(
                "cannot identify a unique main TeX file among: " + ", ".join(self._rel(d) for d in documents)
            )
        if len(tex_files) == 1 and _PLAIN_TEX_END.search(stripped[tex_files[0]]):
            return tex_files[0], "plain-tex"
        raise SourceUnavailable("no TeX file with \\documentclass and \\begin{document}")

    # -- body expansion ---------------------------------------------------

    def _emit(self, text: str, rel: str, line: int) -> None:
        self.parts.append(text)
        for i, _ in enumerate(text.split("\n")):
            if i == 0:
                if self.line_origins[-1] is None:
                    self.line_origins[-1] = (rel, line)
            else:
                self.line_origins.append((rel, line + i))

    def _origin_of(self, text: str, pos: int) -> tuple[str, int]:
        origin = self.line_origins[text.count("\n", 0, pos)]
        return origin if origin else ("", 0)

    def _expand(self, path: Path, included_at: str | None, role: str) -> None:
        """Append path to the output. role is main, body (an included file) or style."""
        if path in self.stack:
            raise SourceUnavailable(f"circular include: {self._rel(path)}")
        if len(self.stack) >= _MAX_INCLUDE_DEPTH:
            raise SourceUnavailable("include nesting is too deep")
        rel = self._rel(path)
        text = _strip_comments(self._read_text(path))
        start_line = 1
        begin = _DOCUMENT_BEGIN.search(text)
        end = _DOCUMENT_END.search(text, begin.end()) if begin else None
        if role == "main":
            # Everything after \end{document} (or \bye) is never typeset.
            terminator = end or _PLAIN_TEX_END.search(text)
            if terminator:
                text = text[:terminator.start()]
        elif begin and role == "body":
            # A subfile has its own preamble; only its document body belongs to the paper.
            start_line = 1 + text.count("\n", 0, begin.end())
            text = text[begin.end():end.start() if end else len(text)]
        self.files_read.append(
            {"path": rel, "role": role, "chars": len(text), "included_at": included_at, "sha256": self.hashes[path]}
        )

        scan = _mask_verbatim(text)
        found = [(m.start(), m.end(), m.group(1), m.group(2) or m.group(3), None) for m in _INPUT_ONE_ARG.finditer(scan)]
        found += [(m.start(), m.end(), m.group(1), m.group(3), m.group(2)) for m in _INPUT_TWO_ARGS.finditer(scan)]
        found.sort()
        if len(found) != len(_INPUT_TOKEN.findall(scan)):
            raise SourceUnavailable(f"include command with unsupported syntax in {rel}")

        self.stack.append(path)
        cursor = 0
        for start, stop, command, name, directory in found:
            self._emit(text[cursor:start], rel, start_line + text.count("\n", 0, cursor))
            line = start_line + text.count("\n", 0, start)
            if directory is not None:
                anchor = path.parent if command == "subimport" else self.root
                dirs = [anchor / directory.strip()]
            else:
                dirs = [self.root, path.parent]
            target = self._resolve(name, dirs, f"\\{command} in {rel}:{line}")
            target_rel = self._rel(target)
            self._emit(f"\n%%% BEGIN FILE: {target_rel} (included at {rel}:{line}) %%%\n", target_rel, 0)
            self.line_origins[-1] = None  # The first line of the included body names its own origin.
            self._expand(target, f"{rel}:{line}", "style" if role == "style" else "body")
            self._emit(f"\n%%% END FILE: {target_rel} %%%\n", rel, line - 1)
            self.line_origins[-1] = (rel, line)  # The rest of the including line continues here.
            cursor = stop
        self._emit(text[cursor:], rel, start_line + text.count("\n", 0, cursor))
        self.stack.pop()

    # -- local styles and bibliography --------------------------------------

    def _local_style_file(self, name: str, suffix: str, main: Path, where: str) -> Path | None:
        name = name.strip()
        if _is_dynamic_name(name):
            # A package name built at run time could load a local file that carries figures.
            raise SourceUnavailable(f"{where} loads a package or class by a dynamic name: {name!r}")
        filename = name if name.endswith(suffix) else name + suffix
        for base in (self.root, main.parent):
            path = Path(os.path.normpath(base / filename))
            if path.is_file() and path.resolve().is_relative_to(self.root):
                return path
        return None  # Not shipped with the paper: a standard TeX distribution package.

    def _inline_local_styles(self, main: Path) -> list[dict]:
        """Append the local .sty/.cls files the document actually loads, as macro context.

        Only files reached through \\documentclass, \\usepackage, \\RequirePackage and
        \\LoadClass from the main text, and recursively from those style files, are read.
        Their content then goes through the same figure and unsupported-construct checks as
        the body, so an image macro hidden in a style file cannot slip past them.
        """
        loaded: list[dict] = []
        seen: set[Path] = set()
        text = "".join(self.parts)
        offset = 0
        while True:
            scan = _mask_verbatim(text[offset:])
            if len(_STYLE_LOAD.findall(scan)) != len(_STYLE_LOAD_TOKEN.findall(scan)):
                raise SourceUnavailable("package or class loading command with unsupported syntax")
            discovered = []
            for m in _STYLE_LOAD.finditer(scan):
                file, line = self._origin_of(text, offset + m.start())
                for name in m.group(2).split(","):
                    path = self._local_style_file(name, _STYLE_LOADERS[m.group(1)], main, f"{file}:{line}")
                    if path is not None and path not in seen:
                        seen.add(path)
                        discovered.append((path, f"{file}:{line}"))
            if not discovered:
                return loaded
            offset = len(text)
            for path, loaded_by in discovered:
                rel = self._rel(path)
                self._emit(
                    f"\n%%% BEGIN LOCAL STYLE FILE (macro context only): {rel} (loaded at {loaded_by}) %%%\n", rel, 0
                )
                self.line_origins[-1] = None
                self._expand(path, loaded_by, "style")
                self._emit(f"\n%%% END LOCAL STYLE FILE: {rel} %%%\n", rel, 0)
                self.line_origins[-1] = (rel, 0)
                loaded.append({"path": rel, "loaded_at": loaded_by, "sha256": self.hashes[path]})
            text = "".join(self.parts)

    def _bibliography(self, body_text: str, main: Path) -> tuple[str, dict | None]:
        """Return the reference list as raw text, preferring the compiled .bbl over .bib files."""
        scan = _mask_verbatim(body_text)
        commands = list(_BIB_COMMAND.finditer(scan))
        if len(commands) != len(_BIB_TOKEN.findall(scan)):
            raise SourceUnavailable("bibliography command with unsupported syntax")
        if not commands:
            return "", None
        bbl = next(
            (p for p in (main.with_suffix(".bbl"), self.root / f"{main.stem}.bbl") if p.is_file()), None
        )
        if bbl is not None:
            kind, paths = "bbl", [bbl]
        else:
            kind, paths = "bib", []
            for m in commands:
                for name in m.group(2).split(","):
                    name = name.strip()
                    if m.group(1) != "addbibresource" and not name.endswith(".bib"):
                        name += ".bib"
                    if _is_dynamic_name(name):
                        raise SourceUnavailable(f"bibliography resource with a dynamic name: {name!r}")
                    path = next(
                        (c for b in (self.root, main.parent) if (c := Path(os.path.normpath(b / name))).is_file()
                         and c.resolve().is_relative_to(self.root)),
                        None,
                    )
                    if path is None:
                        raise SourceUnavailable(f"bibliography resource is missing: {name!r}")
                    if path not in paths:
                        paths.append(path)
        sections, files = [], []
        for path in paths:
            rel = self._rel(path)
            raw = self._read_text(path)  # Raw on purpose: % is data in .bbl and .bib files.
            sections.append(f"\n%%% BEGIN BIBLIOGRAPHY ({kind}): {rel} %%%\n{raw}\n%%% END BIBLIOGRAPHY: {rel} %%%\n")
            files.append({"path": rel, "chars": len(raw), "sha256": self.hashes[path]})
        return "".join(sections), {"kind": kind, "files": files}

    # -- figures ----------------------------------------------------------

    @staticmethod
    def _own(region: _Region, regions: list[_Region], pattern: re.Pattern, scan: str, text: str, group: int | None):
        """Matches of pattern that lie in region but not in a region nested inside it."""
        nested = [r for r in regions if r is not region and region.start <= r.start and r.end <= region.end]
        for m in pattern.finditer(scan, region.start, region.end):
            if any(r.start <= m.start() < r.end for r in nested):
                continue
            if group is None:
                content, _ = _read_group(text, m.end())
                yield content
            else:
                yield m.group(group)

    def _float_regions(self, text: str, scan: str) -> list[_Region]:
        regions: list[_Region] = []
        open_envs: list[tuple[str, int]] = []
        for m in _FLOAT_ENV.finditer(scan):
            if m.group(1) == "begin":
                open_envs.append((m.group(2), m.start()))
                continue
            if not open_envs or open_envs[-1][0] != m.group(2):
                raise SourceUnavailable(f"mismatched \\end{{{m.group(2)}}}")
            name, start = open_envs.pop()
            regions.append(_Region(start, m.end(), name, [], []))
        if open_envs:
            raise SourceUnavailable(f"unclosed \\begin{{{open_envs[-1][0]}}}")
        for env in list(regions):
            if not _INCLUDEGRAPHICS_TOKEN.search(scan, env.start, env.end) and not _TEXTUAL_FLOAT_CONTENT.search(scan, env.start, env.end):
                raise SourceUnavailable(f"{env.kind} environment has no static image or textual content")
        for m in _SUBFLOAT.finditer(scan):
            _, body_end = _read_group(text, m.end())
            regions.append(_Region(m.start(), body_end, "subfloat", [m.group(1)] if m.group(1) else [], []))
        for region in regions:
            if region.kind != "subfloat":
                region.captions = list(self._own(region, regions, _CAPTION, scan, text, None))
            region.labels = list(self._own(region, regions, _LABEL, scan, text, 1))
        return regions

    def _graphics_dirs(self, scan: str) -> list[Path]:
        # LaTeX searches the working directory first, then each \graphicspath entry.
        declared = _GRAPHICSPATH.findall(scan)
        if len(declared) != len(re.findall(r"\\graphicspath(?![a-zA-Z@])", scan)):
            raise SourceUnavailable("\\graphicspath with unsupported syntax")
        definitions = []
        for group in declared:
            entries = [e.strip() for e in re.findall(r"\{([^{}]*)\}", group)]
            if any(_is_dynamic_name(e or ".") for e in entries):
                raise SourceUnavailable(f"dynamic \\graphicspath entry in {entries!r}")
            definitions.append(entries)
        if any(d != definitions[0] for d in definitions):
            # Each definition applies only after its position in the document, so merging
            # them would guess a priority. Refuse instead.
            raise SourceUnavailable("multiple different \\graphicspath definitions")
        return [self.root] + [self.root / e for e in (definitions[0] if definitions else [])]

    def _resolve_image(self, name: str, dirs: list[Path], where: str) -> Path:
        name = name.strip().strip('"')
        if _is_dynamic_name(name):
            raise SourceUnavailable(f"\\includegraphics at {where} uses a dynamic file name: {name!r}")
        variants = [name]
        if Path(name).suffix.lower() not in _IMAGE_EXTENSIONS:
            variants += [name + ext for ext in _IMAGE_EXTENSIONS]
            variants += [name + ext.upper() for ext in _IMAGE_EXTENSIONS]
        for base in dirs:
            for variant in variants:
                path = Path(os.path.normpath(base / variant))
                if path.is_file() and path.resolve().is_relative_to(self.root):
                    return path
        raise SourceUnavailable(f"\\includegraphics at {where} refers to a missing image: {name!r}")

    def _attach_figures(self, text: str, all_files: list[Path]) -> tuple[str, list[SourceFigure], list[dict], list[str]]:
        scan = _mask_verbatim(text)
        for pattern, why in _UNSUPPORTED_COMPILED:
            hit = pattern.search(scan)
            if hit:
                file, line = self._origin_of(text, hit.start())
                raise SourceUnavailable(f"unsupported construct ({why}) at {file}:{line}: {hit.group(0)!r}")
        tokens = [m.start() for m in _INCLUDEGRAPHICS_TOKEN.finditer(scan)]
        matches = []
        for pos in tokens:
            m = _INCLUDEGRAPHICS.match(scan, pos)
            if m is None:
                file, line = self._origin_of(text, pos)
                raise SourceUnavailable(f"\\includegraphics with unsupported syntax at {file}:{line}")
            matches.append(m)

        regions = self._float_regions(text, scan)
        dirs = self._graphics_dirs(scan)
        ref_sites: dict[str, list[dict]] = {}
        for m in _REFERENCE.finditer(scan):
            file, line = self._origin_of(text, m.start())
            for label in (part.strip() for part in m.group(1).split(",")):
                ref_sites.setdefault(label, []).append({"file": file, "line": line})

        figures: dict[str, SourceFigure] = {}
        entries: dict[str, dict] = {}
        out: list[str] = []
        cursor = 0
        for m in matches:
            file, line = self._origin_of(text, m.start())
            image = self._resolve_image(m.group(2), dirs, f"{file}:{line}")
            reference = self._rel(image)
            enclosing = sorted(
                (r for r in regions if r.start <= m.start() < r.end), key=lambda r: r.start - r.end
            )  # Widest first: figure caption, then the subfigure caption.
            caption = "\n".join(_squash(c) for r in enclosing for c in r.captions if _squash(c))
            labels = [label for r in enclosing for label in r.labels]
            figure = figures.setdefault(reference, SourceFigure(image, reference, caption))
            if caption and caption not in figure.caption:
                figure.caption = f"{figure.caption}\n{caption}".strip()
            entry = entries.setdefault(
                reference,
                {"reference": reference, "labels": [], "occurrences": [], "ref_sites": []},
            )
            entry["labels"] += [lb for lb in labels if lb not in entry["labels"]]
            entry["occurrences"].append({"file": file, "line": line, "options": (m.group(1) or "").strip()})
            out.append(text[cursor:m.end()])
            out.append(f"[[IMAGE: {reference}]]")
            cursor = m.end()
        out.append(text[cursor:])
        for entry in entries.values():
            entry["ref_sites"] = [s for lb in entry["labels"] for s in ref_sites.get(lb, [])]

        used = {self.root / ref for ref in figures}
        unreferenced = sorted(
            self._rel(p) for p in all_files if p.suffix.lower() in _IMAGE_EXTENSIONS and p not in used
        )
        return "".join(out), list(figures.values()), list(entries.values()), unreferenced

    # -- entry point ------------------------------------------------------

    def read(self) -> LatexSource:
        files = self._walk()
        main, source_format = self._find_main(files)
        self._expand(main, None, "main")
        body_text = "".join(self.parts)
        style_files = self._inline_local_styles(main)
        text, figures, figure_entries, unreferenced = self._attach_figures("".join(self.parts), files)
        # Appended after the figure scan: reference lists are data, not TeX to be checked.
        bibliography_text, bibliography = self._bibliography(body_text, main)
        text += bibliography_text
        if len(text) > _MAX_TEXT_CHARS:
            raise SourceUnavailable("expanded LaTeX text exceeds the size limit")

        read_paths = {f["path"] for f in self.files_read}
        manifest = {
            "source_format": source_format,
            "main_file": self._rel(main),
            "files_read": self.files_read,
            "unread_tex_files": sorted(
                self._rel(p) for p in files if p.suffix.lower() == ".tex" and self._rel(p) not in read_paths
            ),
            "local_style_files": style_files,
            "unreferenced_style_files": sorted(
                self._rel(p) for p in files
                if p.suffix.lower() in (".sty", ".cls") and self._rel(p) not in read_paths
            ),
            "bibliography": bibliography,
            "figures": figure_entries,
            "unreferenced_images": unreferenced,
            "text_chars": len(text),
            "verified": [
                "every \\input, \\include, \\subfile, \\import and \\subimport resolved to a file in the package",
                "every \\includegraphics resolved to an image file in the package",
                "no circular include, generated picture environment or dynamic file name was found",
                "every local .sty/.cls loaded by the document or by another local style was read and checked the same way",
                "a bibliography command was matched by a .bbl for the main file, or else by every named .bib file",
            ],
            "not_evaluated": [
                "TeX conditionals other than \\iffalse are not evaluated, so both branches are kept",
                "local .sty/.cls files are inlined as macro context, but macros are not expanded or evaluated",
                "images drawn by user-defined macros or environments are not detected",
            ],
        }
        return LatexSource(text=text, figures=figures, manifest=manifest)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_latex_source(root: Path) -> LatexSource:
    """Read an already extracted source directory without touching the network."""
    return _Reader(root).read()
