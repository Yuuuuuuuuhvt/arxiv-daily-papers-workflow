from __future__ import annotations

import asyncio
import logging
import hashlib
import json
import os
import tempfile
from pathlib import Path

import httpx

from .models import ArxivPaper

logger = logging.getLogger(__name__)


def verified_pdf_path(paper: ArxivPaper, output_dir: Path) -> Path | None:
    dest = output_dir / f"{paper.arxiv_id}.pdf"
    try:
        record = json.loads(dest.with_suffix(".pdf.json").read_text())
        if record["version"] == paper.version and record["sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest():
            return dest
    except (OSError, ValueError, KeyError):
        pass
    return None


def _alphaxiv_pdf_url(paper: ArxivPaper) -> str:
    return f"https://fetcher.alphaxiv.org/v2/pdf/{paper.arxiv_id}v{paper.version}.pdf"


async def download_pdf(
    paper: ArxivPaper,
    output_dir: Path,
    sem: asyncio.Semaphore,
) -> Path | None:
    dest = output_dir / f"{paper.arxiv_id}.pdf"
    if verified_pdf_path(paper, output_dir):
        logger.debug("PDF already exists: %s", dest)
        return dest

    # A source archive and its fallback PDF must refer to the exact same version.
    urls = [f"https://arxiv.org/pdf/{paper.arxiv_id}v{paper.version}", _alphaxiv_pdf_url(paper)]

    async with sem:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            for url in urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    if not resp.content.startswith(b"%PDF-"):
                        raise ValueError("Downloaded response is not a PDF")
                    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as out:
                        out.write(resp.content)
                        temporary = out.name
                    os.replace(temporary, dest)
                    dest.with_suffix(".pdf.json").write_text(json.dumps({
                        "version": paper.version, "url": url,
                        "sha256": hashlib.sha256(resp.content).hexdigest()}) + "\n")
                    logger.info("Downloaded PDF: %s from %s (%.1f MB)",
                                paper.arxiv_id, url.split("/")[2], len(resp.content) / 1e6)
                    return dest
                except Exception:
                    logger.debug("PDF download failed from %s for %s", url.split("/")[2], paper.arxiv_id)
            logger.warning("Failed to download PDF for %s from all sources", paper.arxiv_id)
            return None


async def download_all_pdfs(
    papers: list[ArxivPaper],
    output_dir: Path,
    max_concurrent: int = 5,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrent)

    tasks = [download_pdf(paper, output_dir, sem) for paper in papers]
    results = await asyncio.gather(*tasks)

    downloaded: dict[str, Path] = {}
    for paper, path in zip(papers, results):
        if path is not None:
            downloaded[paper.arxiv_id] = path

    logger.info("PDFs downloaded: %d/%d", len(downloaded), len(papers))
    return downloaded
