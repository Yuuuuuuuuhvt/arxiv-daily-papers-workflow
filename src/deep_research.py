from __future__ import annotations

import base64
import logging
from pathlib import Path

from .config import load_prompt, DATA_DIR, ROOT_DIR
from .llm_client import call_llm
from .models import ArxivPaper

logger = logging.getLogger(__name__)

# PDFs larger than this (in bytes) use pdf-text instead of native to avoid
# Gemini processing timeouts on very large documents.
_MAX_NATIVE_PDF_BYTES = 15 * 1024 * 1024  # 15 MB


def _build_paper_content(paper: ArxivPaper) -> str:
    return (
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Categories: {', '.join(paper.categories)}\n"
        f"Abstract:\n{paper.abstract}\n"
        f"Comment: {paper.comment or 'None'}\n"
        f"Journal Ref: {paper.journal_ref or 'None'}\n"
        f"PDF URL: {paper.pdf_url}"
    )


def _read_pdf_base64(paper: ArxivPaper, pdf_dir: Path) -> tuple[str | None, int]:
    """Read a local PDF and return (base64_data_url, file_size_bytes).

    Returns (None, 0) if the file does not exist.
    """
    pdf_path = pdf_dir / f"{paper.arxiv_id}.pdf"
    if not pdf_path.exists():
        return None, 0
    raw = pdf_path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:application/pdf;base64,{b64}", len(raw)


async def generate_deep_research(
    paper: ArxivPaper,
    model_config: dict,
    api_key: str,
    pdf_config: dict | None = None,
) -> str:
    """Generate a rich 3-module scholarly analysis for a single paper.

    PDF handling strategy (3-tier fallback):
      1. If local PDF exists and <= 15 MB → base64 + native engine (Gemini sees
         figures, formulas, tables).
      2. If local PDF > 15 MB → base64 + pdf-text engine (avoid Gemini timeout).
      3. If PDF unavailable or all PDF attempts fail → text-only (abstract).
    """
    system_prompt = load_prompt("deep_research.txt")
    paper_content = _build_paper_content(paper)
    text_part = (
        "请对以下论文进行【模式一：论文解读】，严格按照三个模块输出。\n\n"
        f"{paper_content}"
    )

    use_pdf = pdf_config and pdf_config.get("download_enabled", False)
    if not use_pdf:
        return await _call_text_only(model_config, api_key, system_prompt, text_part, paper)

    # Resolve local PDF directory
    pdf_dir = Path(pdf_config.get("storage_dir", "data/pdfs"))
    if not pdf_dir.is_absolute():
        pdf_dir = ROOT_DIR / pdf_config.get("storage_dir", "storage/pdfs")

    b64_url, file_size = _read_pdf_base64(paper, pdf_dir)
    if not b64_url:
        logger.info("DeepResearch: no local PDF for %s, text-only", paper.arxiv_id)
        return await _call_text_only(model_config, api_key, system_prompt, text_part, paper)

    # Choose engine based on file size
    if file_size <= _MAX_NATIVE_PDF_BYTES:
        engine = "native"
    else:
        engine = "pdf-text"
        logger.info("DeepResearch: PDF too large (%.1f MB) for native, using pdf-text for %s",
                     file_size / 1e6, paper.arxiv_id)

    user_content: list[dict] = [
        {"type": "text", "text": text_part},
        {"type": "file", "file": {
            "filename": f"{paper.arxiv_id}.pdf",
            "file_data": b64_url,
        }},
    ]
    plugins = [{"id": "file-parser", "pdf": {"engine": engine}}]
    logger.info("DeepResearch with PDF for %s (engine=%s, %.1f MB, base64)",
                paper.arxiv_id, engine, file_size / 1e6)

    try:
        report = await call_llm(
            model=model_config["model_id"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=model_config.get("temperature", 0.2),
            api_key=api_key,
            plugins=plugins,
        )
        return report.strip()
    except Exception as exc:
        logger.warning("DeepResearch with PDF failed for %s: %s. Falling back to text-only",
                       paper.arxiv_id, exc)
        return await _call_text_only(model_config, api_key, system_prompt, text_part, paper)


async def _call_text_only(
    model_config: dict, api_key: str,
    system_prompt: str, text_part: str, paper: ArxivPaper,
) -> str:
    try:
        report = await call_llm(
            model=model_config["model_id"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_part},
            ],
            temperature=model_config.get("temperature", 0.2),
            api_key=api_key,
        )
        return report.strip()
    except Exception:
        logger.exception("DeepResearch text-only also failed for %s", paper.arxiv_id)
        return ""


async def generate_all_deep_research(
    papers: list[ArxivPaper],
    model_config: dict,
    api_key: str,
) -> dict[str, str]:
    """Generate deep research reports for all papers.

    Returns {arxiv_id: markdown_report}.
    """
    results: dict[str, str] = {}
    for paper in papers:
        report = await generate_deep_research(paper, model_config, api_key)
        if report:
            results[paper.arxiv_id] = report
            logger.info("DeepResearch done for %s (%d chars)", paper.arxiv_id, len(report))
        else:
            logger.warning("DeepResearch empty for %s", paper.arxiv_id)
    return results
