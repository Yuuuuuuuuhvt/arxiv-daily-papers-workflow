from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR, ROOT_DIR, load_config, get_env
from .dedup import dedup, load_index, save_index
from .deep_analysis import analyze_paper
from .deep_research import generate_deep_research
from .email_sender import send_digest
from .fetcher import hybrid_fetch
from .pdf_downloader import download_all_pdfs
from .git_ops import commit_and_push_data
from .models import ArxivPaper, AnalysisResult, paper_to_index_entry
from .relevance_filter import filter_relevance
from .report_generator import (
    generate_daily_report,
    generate_email_html,
    save_report,
)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stage = record.name.rsplit(".", 1)[-1] if "." in record.name else record.name
        return json.dumps({
            "ts": self.formatTime(record),
            "level": record.levelname,
            "stage": stage,
            "msg": record.getMessage(),
        }, ensure_ascii=False)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


async def run_pipeline() -> None:
    setup_logging()
    logger = logging.getLogger("pipeline")

    config = load_config()
    api_key = get_env("OPENROUTER_API_KEY")
    date_str = os.environ.get("REPORT_DATE", "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    index_path = DATA_DIR / "papers_index.json"
    index = load_index(index_path)

    # Stage 2: Fetch
    logger.info("=== Stage 2: Fetching papers ===")
    candidates = hybrid_fetch(config)
    if not candidates:
        logger.info("No candidates found. Exiting.")
        return

    # Stage 3: Dedup
    logger.info("=== Stage 3: Deduplication ===")
    new_papers = dedup(candidates, index)
    if not new_papers:
        logger.info("No new papers after dedup. Exiting.")
        return

    # Stage 4: Relevance filter (3-tier: core / peripheral / not_relevant)
    logger.info("=== Stage 4: Relevance filter ===")
    core_papers, peripheral_papers = await filter_relevance(
        new_papers,
        config["models"]["relevance_filter"],
        api_key,
    )
    all_relevant = core_papers + peripheral_papers
    if not all_relevant:
        logger.info("No relevant papers after filtering. Exiting.")
        return

    # Stage 5: Deep analysis for ALL relevant papers (core + peripheral)
    logger.info("=== Stage 5: Deep analysis (%d core + %d peripheral, concurrent) ===",
                len(core_papers), len(peripheral_papers))
    sem = asyncio.Semaphore(config.get("concurrency", {}).get("max_concurrent_llm", 10))

    async def _analyze_one(paper: ArxivPaper) -> tuple[str, AnalysisResult]:
        async with sem:
            logger.info("[deep_analysis] start %s", paper.arxiv_id)
            a = await analyze_paper(paper, config["models"]["deep_analysis"], config["scoring"], api_key)
            logger.info("[deep_analysis] done  %s score=%.2f", paper.arxiv_id, a.weighted_score)
            return paper.arxiv_id, a

    analysis_results = await asyncio.gather(*[_analyze_one(p) for p in all_relevant])
    analyses: dict[str, AnalysisResult] = dict(analysis_results)

    # PDF download (concurrent, for local archival) — core papers only
    pdf_config = config.get("pdf", {})
    if pdf_config.get("download_enabled", False):
        pdf_base = Path(pdf_config.get("storage_dir", "data/pdfs"))
        if not pdf_base.is_absolute():
            pdf_base = ROOT_DIR / pdf_config.get("storage_dir", "storage/pdfs")
        pdf_dir = pdf_base / date_str
        logger.info("=== Downloading PDFs to %s ===", pdf_dir)
        await download_all_pdfs(core_papers, pdf_dir)
        pdf_config = {**pdf_config, "storage_dir": str(pdf_dir)}

    # Stage 5b: DeepResearch — CORE papers only (with full PDF)
    skip_deep_research = os.environ.get("SKIP_DEEP_RESEARCH", "").strip().lower() in ("1", "true", "yes")
    deep_research_reports: dict[str, str] = {}

    if not skip_deep_research:
        pdf_concurrency = config.get("concurrency", {}).get("max_concurrent_pdf", 5)
        sem_pdf = asyncio.Semaphore(pdf_concurrency)
        logger.info("=== Stage 5b: DeepResearch (%d CORE papers, concurrent=%d, PDF=%s) ===",
                     len(core_papers), pdf_concurrency, pdf_config.get("download_enabled", False))

        async def _research_one(paper: ArxivPaper) -> tuple[str, str]:
            async with sem_pdf:
                logger.info("[deep_research] start %s", paper.arxiv_id)
                dr = await generate_deep_research(
                    paper, config["models"]["deep_analysis"], api_key,
                    pdf_config=pdf_config if pdf_config.get("download_enabled") else None,
                )
                logger.info("[deep_research] done  %s (%d chars)", paper.arxiv_id, len(dr))
                return paper.arxiv_id, dr

        research_results = await asyncio.gather(*[_research_one(p) for p in core_papers])
        deep_research_reports = {k: v for k, v in research_results if v}
    else:
        logger.info("=== Stage 5b: DeepResearch skipped (SKIP_DEEP_RESEARCH=1) ===")

    # Save index
    core_ids = {p.arxiv_id for p in core_papers}
    github_repo = os.environ.get("GITHUB_REPOSITORY", "Yuuuuuuuuhvt/arxiv-daily-papers-workflow")
    for paper in all_relevant:
        analysis = analyses.get(paper.arxiv_id)
        if analysis:
            rel_url = f"https://github.com/{github_repo}/releases/download/v{date_str}/{paper.arxiv_id}.pdf"
            entry = paper_to_index_entry(paper, analysis, release_pdf_url=rel_url)
            entry["relevance_tier"] = "core" if paper.arxiv_id in core_ids else "peripheral"
            index[paper.arxiv_id] = entry
    save_index(index, index_path)

    # Stage 6: Report generation
    logger.info("=== Stage 6: Report generation ===")
    report_md = generate_daily_report(
        date_str, core_papers, peripheral_papers, analyses, deep_research_reports, config
    )
    save_report(report_md, date_str)

    # Stage 7: Email
    logger.info("=== Stage 7: Email ===")
    qq_addr = get_env("QQ_MAIL_ADDRESS", required=False)
    qq_auth = get_env("QQ_MAIL_AUTH_CODE", required=False)
    if qq_addr and qq_auth:
        email_html = generate_email_html(date_str, core_papers, analyses, config)
        subject = f"arXiv Daily -- {date_str} -- {len(all_relevant)} papers"
        send_digest(
            subject=subject,
            html_body=email_html,
            to_addr=qq_addr,
            auth_code=qq_auth,
            smtp_server=config["email"]["smtp_server"],
            smtp_port=config["email"]["smtp_port"],
        )
    else:
        logger.info("Email skipped: QQ_MAIL_ADDRESS or QQ_MAIL_AUTH_CODE not set")

    # Stage 7: Git push (only in CI unless delegated to workflow)
    skip_git_push = os.environ.get("SKIP_GIT_PUSH", "").strip().lower() in ("1", "true", "yes")
    if os.environ.get("GITHUB_ACTIONS") and not skip_git_push:
        logger.info("=== Stage 7: Git push ===")
        commit_and_push_data(date_str)
    elif os.environ.get("GITHUB_ACTIONS"):
        logger.info("Git push skipped: SKIP_GIT_PUSH=1")
    else:
        logger.info("Not in CI, skipping git push")

    logger.info("=== Pipeline complete: %d core + %d peripheral papers ===",
                len(core_papers), len(peripheral_papers))


def main() -> None:
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
