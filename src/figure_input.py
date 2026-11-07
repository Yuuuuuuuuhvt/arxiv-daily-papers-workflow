"""Convert figures referenced by paper LaTeX sources into images a vision model can read.

Every returned image is a PNG. Raster images keep their original pixel grid, PDF figures are
rendered page by page at a fixed DPI, and no image is downscaled or merged into a contact
sheet, because shrinking a figure makes the text inside it unreadable. Inputs that exceed the
resource limits fail with FigureInputError instead of being dropped, so the caller decides how
to fall back.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Bump when the rendering pipeline changes output pixels, so stale cache entries are not reused.
RENDER_VERSION = 1

DEFAULT_PDF_DPI = 150
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PIXELS_PER_IMAGE = 40_000_000
DEFAULT_MAX_TOTAL_PIXELS = 200_000_000

_PDF_POINTS_PER_INCH = 72
_MANIFEST_NAME = "manifest.json"
_SUPPORTED_RASTER_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
# Only these modes have an unambiguous 8-bit conversion to RGB or RGBA. 16-bit and float modes
# are rejected because a plain conversion would clip or rescale intensities.
_ALPHA_MODES = {"RGBA", "LA", "PA"}
_OPAQUE_MODES = {"RGB", "L", "P", "CMYK", "1"}

# PDFium keeps process-wide state and is not thread-safe, so every call into it takes this lock.
_PDFIUM_LOCK = threading.Lock()


class FigureInputError(Exception):
    """Raised when a figure cannot be converted into a model-readable image."""


@dataclass(frozen=True)
class RenderedImage:
    """One output image. `page` is the 1-based PDF page or animation frame; 1 for still images."""

    source_path: Path
    page: int
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class _Limits:
    dpi: int
    max_file_bytes: int
    max_pixels_per_image: int
    max_total_pixels: int


def render_images(
    paths: list[Path],
    cache_dir: Path,
    *,
    dpi: int = DEFAULT_PDF_DPI,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_pixels_per_image: int = DEFAULT_MAX_PIXELS_PER_IMAGE,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
) -> list[RenderedImage]:
    """Convert figure files to PNG images, keeping input order and every page or frame.

    A path that appears more than once is converted once and listed once. Output files live in
    `cache_dir/<sha256 of content and render parameters>/`, so identical content is reused across
    calls and different DPI settings never share files.
    """
    limits = _Limits(dpi, max_file_bytes, max_pixels_per_image, max_total_pixels)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[RenderedImage] = []
    seen: set[Path] = set()
    total_pixels = 0
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        images = _render_one(path, cache_dir, limits)
        total_pixels += sum(image.width * image.height for image in images)
        if total_pixels > limits.max_total_pixels:
            raise FigureInputError(
                f"Figures exceed the total pixel limit of {limits.max_total_pixels} at {path.name}"
            )
        results.extend(images)
    return results


def _render_one(path: Path, cache_dir: Path, limits: _Limits) -> list[RenderedImage]:
    try:
        size = path.stat().st_size
        if size > limits.max_file_bytes:
            raise FigureInputError(f"{path.name} is {size} bytes, over the {limits.max_file_bytes} byte limit")
        data = path.read_bytes()
    except OSError as exc:
        raise FigureInputError(f"Cannot read figure {path}: {exc}") from exc

    kind = _sniff(data)
    if kind in ("eps", "svg"):
        raise FigureInputError(f"{path.name}: {kind.upper()} conversion is not supported, use the PDF instead")

    digest = hashlib.sha256(data)
    digest.update(f"|v{RENDER_VERSION}|dpi={limits.dpi}".encode())
    entry = cache_dir / digest.hexdigest()

    cached = _load_cached(entry)
    if cached is not None:
        # Limits are enforced on cached sizes too, so a tightened limit is never bypassed.
        _check_dimensions(path, [(w, h) for _, w, h in cached], limits)
        return [RenderedImage(path, page, entry / name, w, h) for page, (name, w, h) in enumerate(cached, 1)]

    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=cache_dir))
    try:
        if kind == "pdf":
            produced = _render_pdf(path, data, staging, limits)
        else:
            produced = _render_raster(path, data, staging, limits)
        (staging / _MANIFEST_NAME).write_text(
            json.dumps([{"name": n, "width": w, "height": h} for n, w, h in produced]),
            encoding="utf-8",
        )
        try:
            # Renaming the finished directory publishes all pages at once, so readers never see
            # a half-written entry. If another process published first, its identical entry wins.
            os.replace(staging, entry)
        except OSError:
            if _load_cached(entry) is None:
                raise
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return [RenderedImage(path, page, entry / name, w, h) for page, (name, w, h) in enumerate(produced, 1)]


def _sniff(data: bytes) -> str:
    """Identify the container from content, because LaTeX file extensions are often wrong."""
    # The PDF spec allows junk before the header within the first 1024 bytes.
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if data.startswith(b"%!PS") or data.startswith(b"\xc5\xd0\xd3\xc6"):
        return "eps"
    head = data[:2048].lstrip().lower()
    if head.startswith(b"<svg") or (head.startswith((b"<?xml", b"<!doctype svg")) and b"<svg" in data[:8192].lower()):
        return "svg"
    return "raster"


def _load_cached(entry: Path) -> list[tuple[str, int, int]] | None:
    try:
        items = json.loads((entry / _MANIFEST_NAME).read_text(encoding="utf-8"))
        pages = [(item["name"], int(item["width"]), int(item["height"])) for item in items]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not pages or not all((entry / name).is_file() for name, _, _ in pages):
        return None
    return pages


def _check_dimensions(path: Path, sizes: list[tuple[int, int]], limits: _Limits) -> None:
    total = 0
    for width, height in sizes:
        if width * height > limits.max_pixels_per_image:
            raise FigureInputError(
                f"{path.name}: a {width}x{height} page exceeds the per-image limit of "
                f"{limits.max_pixels_per_image} pixels"
            )
        total += width * height
    if total > limits.max_total_pixels:
        raise FigureInputError(f"{path.name}: {total} pixels exceed the total limit of {limits.max_total_pixels}")


def _page_name(index: int) -> str:
    return f"page-{index:04d}.png"


def _render_pdf(path: Path, data: bytes, staging: Path, limits: _Limits) -> list[tuple[str, int, int]]:
    scale = limits.dpi / _PDF_POINTS_PER_INCH
    with _PDFIUM_LOCK:
        try:
            document = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as exc:
            raise FigureInputError(f"{path.name}: cannot open PDF: {exc}") from exc
        try:
            count = len(document)
            if count == 0:
                raise FigureInputError(f"{path.name}: PDF has no pages")
            # Measure every page before rendering anything, so an oversized document fails
            # fast and no page is silently skipped.
            sizes = []
            for index in range(count):
                page = document[index]
                try:
                    width, height = page.get_size()
                finally:
                    page.close()
                sizes.append((max(1, math.ceil(width * scale)), max(1, math.ceil(height * scale))))
            _check_dimensions(path, sizes, limits)

            produced = []
            for index in range(count):
                page = document[index]
                try:
                    image = page.render(scale=scale).to_pil()
                except pdfium.PdfiumError as exc:
                    raise FigureInputError(f"{path.name}: cannot render page {index + 1}: {exc}") from exc
                finally:
                    page.close()
                name = _page_name(index + 1)
                image.save(staging / name, format="PNG")
                produced.append((name, image.width, image.height))
            return produced
        finally:
            document.close()


def _render_raster(path: Path, data: bytes, staging: Path, limits: _Limits) -> list[tuple[str, int, int]]:
    try:
        image = Image.open(io.BytesIO(data))
        fmt = image.format
        if fmt not in _SUPPORTED_RASTER_FORMATS:
            raise FigureInputError(f"{path.name}: unsupported image format {fmt}")
        frames = getattr(image, "n_frames", 1)
        # Header-only size check, run before any pixel decoding to stop decompression bombs.
        _check_dimensions(path, [image.size] * frames, limits)

        produced = []
        for index in range(frames):
            image.seek(index)
            frame = _to_png_ready(path, image)
            name = _page_name(index + 1)
            frame.save(staging / name, format="PNG")
            produced.append((name, frame.width, frame.height))
        return produced
    except FigureInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as exc:
        raise FigureInputError(f"{path.name}: cannot decode image: {exc}") from exc


def _to_png_ready(path: Path, image: Image.Image) -> Image.Image:
    # Honour the camera orientation tag, otherwise rotated photos would be sent sideways.
    image = ImageOps.exif_transpose(image)
    mode = image.mode
    has_transparency = mode in _ALPHA_MODES or "transparency" in image.info
    if has_transparency:
        return image.convert("RGBA")
    if mode in _OPAQUE_MODES:
        return image.convert("RGB")
    raise FigureInputError(f"{path.name}: unsupported pixel mode {mode}")
