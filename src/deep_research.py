from __future__ import annotations

import base64
import logging
from pathlib import Path

from .config import load_prompt, ROOT_DIR
from .budget import BudgetExceeded
from .llm_client import call_llm
from .models import ArxivPaper

logger = logging.getLogger(__name__)

# PDFs larger than this use the free text parser instead of native to avoid
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
    """Analyze a PDF, or explicitly label an allowed abstract-only fallback.

    URL input is optional and bounded by the provider download size limit.
    require_pdf disables abstract fallback for runs that need full documents.
    """
    system_prompt = load_prompt("deep_research.txt")
    paper_content = _build_paper_content(paper)
    text_part = (
        "请对以下论文进行【模式一：论文解读】，严格按照三个模块输出。\n\n"
        f"{paper_content}"
    )

    require_pdf = bool(pdf_config and pdf_config.get("require_pdf", False))
    use_pdf = pdf_config and pdf_config.get("download_enabled", False)
    if not use_pdf:
        if require_pdf:
            raise ValueError("require_pdf needs PDF downloads enabled")
        return await _call_text_only(model_config, api_key, system_prompt, text_part, paper)

    # Resolve local PDF directory
    pdf_dir = Path(pdf_config.get("storage_dir", "data/pdfs"))
    if not pdf_dir.is_absolute():
        pdf_dir = ROOT_DIR / pdf_config.get("storage_dir", "storage/pdfs")

    pdf_path = pdf_dir / f"{paper.arxiv_id}.pdf"
    if not pdf_path.is_file():
        if require_pdf:
            raise FileNotFoundError(f"Required PDF missing: {paper.arxiv_id}")
        logger.info("DeepResearch: no local PDF for %s, abstract-only", paper.arxiv_id)
        return await _call_text_only(model_config, api_key, system_prompt, text_part, paper)
    with pdf_path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise ValueError(f"Invalid PDF header: {paper.arxiv_id}")
    file_size = pdf_path.stat().st_size
    engine = "native" if file_size <= _MAX_NATIVE_PDF_BYTES else "cloudflare-ai"
    # OpenRouter URL downloads reject files larger than 5 MiB. Inline bytes
    # retain the local document and avoid that separate download limit.
    transport = pdf_config.get("transport", "base64")
    if transport not in {"url", "base64"}:
        raise ValueError("pdf.transport must be url or base64")
    use_url = transport == "url"
    if use_url and file_size > 5 * 1024 * 1024:
        raise ValueError("PDF URL download exceeds the provider 5 MiB limit; use base64")
    file_data = paper.pdf_url if use_url else _read_pdf_base64(paper, pdf_dir)[0]

    user_content: list[dict] = [
        {"type": "text", "text": text_part},
        {"type": "file", "file": {
            "filename": f"{paper.arxiv_id}.pdf",
            "file_data": file_data,
        }},
    ]
    plugins = [{"id": "file-parser", "pdf": {"engine": engine}}]
    logger.info("DeepResearch with PDF for %s (engine=%s, %.1f MB)",
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
        source_note = "PDF 原文，包含图表输入" if engine == "native" else "PDF 全文解析，图像细节可能缺失"
        return f"> 输入依据：{source_note}。\n\n{report.strip()}"
    except BudgetExceeded:
        raise
    except Exception as exc:
        if require_pdf:
            raise
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
        return f"> 输入依据：仅论文摘要，尚未完成 PDF 全文解读。\n\n{report.strip()}"
    except BudgetExceeded:
        raise
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
