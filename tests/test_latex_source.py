from __future__ import annotations

import asyncio
import gzip
import io
import tarfile
from datetime import datetime

import httpx
import pytest

from src import latex_source
from src.latex_source import SourceUnavailable, load_latex_source, prepare_latex_source
from src.models import ArxivPaper


def _write(root, files: dict[str, str | bytes]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())


def _tar_gz(members: list[tuple[str, bytes | None, bytes]]) -> bytes:
    """Members are (name, content, tar type); content None means no payload."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.REGTYPE:
                info.size = len(content or b"")
            else:
                info.linkname = "/etc/passwd"
            archive.addfile(info, io.BytesIO(content or b"") if kind == tarfile.REGTYPE else None)
    return buffer.getvalue()


def _paper(version: int = 1) -> ArxivPaper:
    now = datetime(2026, 1, 1)
    return ArxivPaper(
        arxiv_id="2506.12345", version=version, title="t", abstract="a", authors=[], categories=[],
        primary_category="cs.RO", published=now, updated=now, pdf_url="", abs_url="",
    )


def test_multifile_body_appendix_and_figure_association(tmp_path):
    _write(tmp_path, {
        "main.tex": (
            "\\documentclass{article}\n\\usepackage{mymacros}\n\\graphicspath{{figs/}}\n\\begin{document}\n"
            "\\input{sections/intro}\n% \\input{ghost}\n"
            "\\begin{figure}\\includegraphics[width=\\linewidth]{arch}\\caption{Overall {\\bf design}}"
            "\\label{fig:arch}\\end{figure}\n"
            "\\iffalse \\includegraphics{ghost} \\fi\n"
            "\\appendix\n\\include{appendix}\n\\bibliography{refs}\n\\end{document}\nTRAILING"
        ),
        "sections/intro.tex": (
            "Intro with $a \\iff b$ and Figure~\\ref{fig:arch}. 100\\% sure.\n"
            "\\begin{figure}\\subfloat[Left view]{\\includegraphics{left.png}}\\caption{Views}\\end{figure}\n"
        ),
        "appendix.tex": "\\section{Extra}\nAppendix body \\begin{tabular}{c}x\\end{tabular}\n",
        "figs/arch.pdf": b"%PDF-1.5",
        "left.png": b"\x89PNG",
        "unused.png": b"\x89PNG",
        "mymacros.sty": "\\RequirePackage{mydeps}\n\\newcommand{\\method}{Foo} % trailing comment\n",
        "mydeps.sty": "\\newcommand{\\dep}{Bar}\n",
        "unrelated.sty": "\\newcommand{\\junk}{Never sent}\n",
        "main.bbl": "\\begin{thebibliography}{1}\\bibitem{a} Smith, 50\\% title % kept raw\n\\end{thebibliography}\n",
        "refs.bib": "@article{a, title={From bib}}",
    })

    source = load_latex_source(tmp_path)

    assert "Appendix body" in source.text and "Intro with" in source.text
    assert "ghost" not in source.text and "TRAILING" not in source.text
    assert "100\\% sure" in source.text  # Escaped percent sign is not a comment.
    assert "%%% BEGIN FILE: appendix.tex (included at main.tex:" in source.text
    assert "\\includegraphics[width=\\linewidth]{arch}[[IMAGE: figs/arch.pdf]]" in source.text
    by_ref = {f.reference: f for f in source.figures}
    assert set(by_ref) == {"figs/arch.pdf", "left.png"}
    assert by_ref["figs/arch.pdf"].caption == "Overall {\\bf design}"
    assert by_ref["left.png"].caption == "Views\nLeft view"
    assert by_ref["figs/arch.pdf"].path == (tmp_path / "figs/arch.pdf").resolve()
    arch = next(e for e in source.manifest["figures"] if e["reference"] == "figs/arch.pdf")
    assert arch["labels"] == ["fig:arch"]
    assert arch["ref_sites"] == [{"file": "sections/intro.tex", "line": 1}]
    assert source.manifest["unreferenced_images"] == ["unused.png"]
    # Local styles are macro context, loaded transitively, with provenance; unrelated ones stay out.
    assert "\\newcommand{\\method}{Foo}" in source.text and "\\newcommand{\\dep}{Bar}" in source.text
    assert "trailing comment" not in source.text and "Never sent" not in source.text
    assert [(f["path"], f["loaded_at"]) for f in source.manifest["local_style_files"]] == [
        ("mymacros.sty", "main.tex:2"), ("mydeps.sty", "mymacros.sty:1")
    ]
    assert source.manifest["unreferenced_style_files"] == ["unrelated.sty"]
    # The .bbl is preferred over the .bib and appended verbatim with its hash.
    assert "% kept raw" in source.text and "From bib" not in source.text
    bibliography = source.manifest["bibliography"]
    assert bibliography["kind"] == "bbl" and len(bibliography["files"][0]["sha256"]) == 64
    (tmp_path / "main.bbl").unlink()
    assert "From bib" in load_latex_source(tmp_path).text
    assert [f["path"] for f in source.manifest["files_read"]] == [
        "main.tex", "sections/intro.tex", "appendix.tex", "mymacros.sty", "mydeps.sty"
    ]


def test_unverifiable_structures_raise_instead_of_dropping_figures(tmp_path):
    base = "\\documentclass{article}\\begin{document}%s\\end{document}"
    cases = {
        "missing image": {"main.tex": base % "\\includegraphics{nothing}"},
        "missing input": {"main.tex": base % "\\input{nothing}"},
        "external listing": {"main.tex": base % "\\lstinputlisting{algorithm.py}"},
        "unsupported include": {"main.tex": base % "\\subincludefrom{appendix/}{results}"},
        "cycle": {"main.tex": base % "\\input{a}", "a.tex": "\\input{b}", "b.tex": "\\input{a}"},
        "generated tikz": {"main.tex": base % "\\begin{figure}\\begin{tikzpicture}\\end{tikzpicture}\\end{figure}"},
        "dynamic image name": {"main.tex": base % "\\includegraphics{\\figdir/a}", "a.png": b"x"},
        "figure without image": {"main.tex": base % "\\begin{figure}\\fbox{x}\\caption{c}\\end{figure}"},
        "two possible roots": {"main.tex": base % "a", "other.tex": base % "b"},
        "no tex": {"readme.txt": "hello"},
        "image macro in style": {
            "main.tex": base.replace("\\begin{document}", "\\usepackage{figs}\\begin{document}") % "\\fig{a}",
            "figs.sty": "\\newcommand{\\fig}[1]{\\includegraphics{pics/#1}}",
        },
        "tikz in local class": {
            "main.tex": base.replace("{article}", "{mycls}") % "x",
            "mycls.cls": "\\LoadClass{article}\\newcommand{\\logo}{\\begin{tikzpicture}\\end{tikzpicture}}",
        },
        "dynamic package name": {"main.tex": base.replace("\\begin{document}", "\\usepackage{\\pkg}\\begin{document}") % "x"},
        "missing bibliography": {"main.tex": base % "\\bibliography{refs}"},
        "conflicting graphicspath": {
            "main.tex": base % "\\graphicspath{{a/}}\\graphicspath{{b/}}\\includegraphics{x}",
            "a/x.png": b"1", "b/x.png": b"2",
        },
    }
    for name, files in cases.items():
        root = tmp_path / name.replace(" ", "_")
        _write(root, files)
        with pytest.raises(SourceUnavailable):
            load_latex_source(root)


def test_archive_safety_rate_limit_and_version_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(latex_source, "_MIN_DOWNLOAD_INTERVAL_SECONDS", 0)
    tex = b"\\documentclass{article}\\begin{document}Hello\\end{document}"
    good = _tar_gz([("paper/main.tex", tex, tarfile.REGTYPE)])
    responses = {"v1": good, "v2": gzip.compress(tex), "v3": b"<html>captcha</html>", "v4": None}
    redirects = {"v5": "https://evilarxiv.org/src/x", "v6": "https://export.arxiv.org/src/2506.12345v6"}
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.host != "arxiv.org":
            return httpx.Response(200, content=good)
        if "v" + request.url.path.rsplit("v", 1)[1] in redirects:
            return httpx.Response(302, headers={"location": redirects["v" + request.url.path.rsplit("v", 1)[1]]})
        body = responses["v" + request.url.path.rsplit("v", 1)[1]]
        if body is None:
            return httpx.Response(429)
        return httpx.Response(200, content=body)

    monkeypatch.setattr(latex_source, "_new_client", lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
    )

    def prepare(version: int):
        return asyncio.run(prepare_latex_source(_paper(version), tmp_path))

    first = prepare(1)
    assert "Hello" in first.text and first.manifest["main_file"] == "paper/main.tex"
    assert requests == ["https://arxiv.org/src/2506.12345v1"]
    prepare(1)
    assert len(requests) == 1  # Same version is served from the cache.
    assert "Hello" in prepare(2).text and requests[-1].endswith("v2")  # Single gzip tex, new version, new cache.
    assert (tmp_path / "latex_source/2506.12345v1/source").is_dir()
    assert (tmp_path / "latex_source/2506.12345v2/source").is_dir()

    for version in (3, 4):  # HTML fake response and HTTP 429, each requested exactly once.
        before = len(requests)
        with pytest.raises(SourceUnavailable):
            prepare(version)
        assert len(requests) == before + 1
        assert not (tmp_path / f"latex_source/2506.12345v{version}").exists()
    with pytest.raises(SourceUnavailable):  # Redirect to a look-alike host is not arxiv.org.
        prepare(5)
    assert "Hello" in prepare(6).text  # A real arXiv subdomain is accepted.
    assert not [p for p in (tmp_path / "latex_source").iterdir() if p.name.startswith(".tmp-")]

    dest = tmp_path / "unsafe"
    for members in (
        [("../escape.tex", tex, tarfile.REGTYPE)],
        [("/abs.tex", tex, tarfile.REGTYPE)],
        [("link.tex", None, tarfile.SYMTYPE)],
        [("hard.tex", None, tarfile.LNKTYPE)],
        [("dev", None, tarfile.CHRTYPE)],
    ):
        dest.mkdir(exist_ok=True)
        with pytest.raises(SourceUnavailable):
            latex_source._unpack(_tar_gz(members), dest)
    assert not (tmp_path / "escape.tex").exists()
    with pytest.raises(SourceUnavailable):
        latex_source._unpack(gzip.compress(b"%PDF-1.7 only a pdf"), dest)
    with pytest.raises(SourceUnavailable):  # A second gzip member must not be dropped silently.
        latex_source._unpack(gzip.compress(tex) + gzip.compress(tex), dest)
