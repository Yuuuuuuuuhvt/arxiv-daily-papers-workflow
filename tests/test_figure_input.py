from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from src.figure_input import FigureInputError, render_images


def _make_pdf(path: Path, page_sizes: list[tuple[int, int]]) -> Path:
    document = pdfium.PdfDocument.new()
    for width, height in page_sizes:
        document.new_page(width, height)
    document.save(path)
    document.close()
    return path


def test_render_preserves_pixels_pages_and_cache_isolation(tmp_path: Path) -> None:
    rgba = Image.new("RGBA", (4, 2), (0, 0, 0, 0))
    rgba.putpixel((1, 0), (255, 0, 0, 255))
    rgba.putpixel((2, 0), (0, 255, 0, 128))
    png = tmp_path / "alpha.png"
    rgba.save(png)

    gif = tmp_path / "anim.gif"
    frame_a = Image.new("RGB", (3, 3), (255, 0, 0))
    frame_b = Image.new("RGB", (3, 3), (0, 0, 255))
    frame_a.save(gif, save_all=True, append_images=[frame_b], duration=50, loop=0)

    pdf = _make_pdf(tmp_path / "multi.pdf", [(200, 100), (100, 300), (50, 50)])
    pdf_copy = tmp_path / "copy.pdf"
    pdf_copy.write_bytes(pdf.read_bytes())

    cache = tmp_path / "cache"
    results = render_images([png, pdf, png, gif, pdf_copy], cache)

    # The repeated png path is encoded once and listed once; the byte-identical copy is a
    # distinct path, so it is listed but reuses the same cached files.
    assert [(r.source_path.name, r.page) for r in results] == [
        ("alpha.png", 1),
        ("multi.pdf", 1), ("multi.pdf", 2), ("multi.pdf", 3),
        ("anim.gif", 1), ("anim.gif", 2),
        ("copy.pdf", 1), ("copy.pdf", 2), ("copy.pdf", 3),
    ]

    with Image.open(results[0].path) as out:
        assert out.mode == "RGBA" and out.size == (4, 2)
        assert out.getpixel((0, 0))[3] == 0
        assert out.getpixel((1, 0)) == (255, 0, 0, 255)
        assert out.getpixel((2, 0)) == (0, 255, 0, 128)

    # 150 DPI means 150/72 pixels per point, rounded up. Recorded sizes must match the files.
    assert [(r.width, r.height) for r in results[1:4]] == [(417, 209), (209, 625), (105, 105)]
    for r in results:
        with Image.open(r.path) as out:
            assert out.size == (r.width, r.height)

    with Image.open(results[4].path) as first, Image.open(results[5].path) as second:
        assert first.convert("RGB").getpixel((1, 1)) == (255, 0, 0)
        assert second.convert("RGB").getpixel((1, 1)) == (0, 0, 255)

    assert [r.path for r in results[1:4]] == [r.path for r in results[6:9]]

    # A different DPI must land in a different cache entry with different dimensions.
    other = render_images([pdf], cache, dpi=72)
    assert other[0].path.parent != results[1].path.parent
    assert (other[0].width, other[0].height) == (200, 100)
    assert not any(p.name.startswith(".staging-") for p in cache.iterdir())


def test_rejects_bad_unsupported_and_oversized_inputs(tmp_path: Path) -> None:
    cache = tmp_path / "cache"

    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage")
    with pytest.raises(FigureInputError):
        render_images([corrupt], cache)

    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4\nnot a real document")
    with pytest.raises(FigureInputError):
        render_images([bad_pdf], cache)

    tiff = tmp_path / "scan.tiff"
    Image.new("RGB", (4, 4)).save(tiff)
    with pytest.raises(FigureInputError, match="unsupported"):
        render_images([tiff], cache)

    svg = tmp_path / "vector.svg"
    svg.write_text('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>')
    eps = tmp_path / "vector.eps"
    eps.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n")
    for vector in (svg, eps):
        with pytest.raises(FigureInputError, match="not supported"):
            render_images([vector], cache)

    with pytest.raises(FigureInputError):
        render_images([tmp_path / "missing.png"], cache)

    big = tmp_path / "big.png"
    Image.new("RGB", (100, 100)).save(big)
    with pytest.raises(FigureInputError, match="per-image"):
        render_images([big], cache, max_pixels_per_image=9_999)

    # Every page is under the per-image limit but the document total is not: nothing is dropped.
    pdf = _make_pdf(tmp_path / "pages.pdf", [(72, 72)] * 3)
    with pytest.raises(FigureInputError, match="total"):
        render_images([pdf], cache, dpi=72, max_total_pixels=3 * 72 * 72 - 1)
    with pytest.raises(FigureInputError, match="total"):
        render_images([big], cache, max_total_pixels=9_999)
    with pytest.raises(FigureInputError, match="byte limit"):
        render_images([big], cache, max_file_bytes=10)
