from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .budget import BudgetExceeded, RunBudget, current_budget
from .config import DATA_DIR, ROOT_DIR, load_config, get_env, release_pdf_url
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


def save_run_summary(date_str: str, summary: dict, data_dir: Path | None = None) -> None:
    directory = (data_dir if data_dir is not None else DATA_DIR) / "run-metadata"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{date_str}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_empty_run(date_str: str, status: str, config: dict, summary: dict, data_dir: Path | None = None) -> None:
    # 重复补跑没有新论文时，已发布日报仍是当天的完整分析结果。
    # 已有摘要同样保留；只有旧日报缺少摘要时才记录本次空尝试。
    data_dir = data_dir if data_dir is not None else DATA_DIR
    report_path = data_dir / "reports" / f"{date_str}.md"
    if report_path.is_file():
        logging.getLogger("pipeline").info("Retaining existing report for %s after %s", date_str, status)
        if not (data_dir / "run-metadata" / f"{date_str}.json").is_file():
            save_run_summary(date_str, {**summary, "status": status, "retained_existing_report": True}, data_dir)
        return
    save_report(generate_daily_report(date_str, [], [], {}, {}, config), date_str, data_dir)
    summary["status"] = status
    save_run_summary(date_str, summary, data_dir)


async def _load_run_budget(config: dict, data_dir: Path, date_str: str) -> RunBudget | None:
    limit = os.environ.get("PAPERS_BUDGET_USD", "").strip()
    if not limit:
        return None
    settings = config.get("budget", {})
    policies = settings.get("models", {})
    model_ids = {config["models"][stage]["model_id"] for stage in ("relevance_filter", "deep_analysis")}
    if not model_ids.issubset(policies):
        raise ValueError("Budget price limits must cover every configured model")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        catalog = {model["id"]: model for model in response.json()["data"]}
    verified = {}
    for model_id in model_ids:
        model = catalog[model_id]
        context_length = model["context_length"]
        if type(context_length) is not int or context_length <= 0:
            raise ValueError(f"Missing context limit for {model_id}")
        policy = policies[model_id]
        prompt_cap = Decimal(str(policy["prompt_price_per_million"])) / 1000000
        completion_cap = Decimal(str(policy["completion_price_per_million"])) / 1000000
        pricing = model["pricing"]
        for tier in [pricing, *pricing.get("overrides", [])]:
            for field, cap in (("prompt", prompt_cap), ("completion", completion_cap),
                               ("internal_reasoning", completion_cap), ("input_cache_read", prompt_cap),
                               ("input_cache_write", prompt_cap)):
                price = Decimal(str(tier.get(field, pricing.get(field, "0"))))
                if not price.is_finite() or price < 0 or price > cap:
                    raise ValueError(f"Published {field} price exceeds the budget policy for {model_id}")
        if Decimal(str(pricing.get("request", "0"))) != 0:
            raise ValueError("Per-request fees are not supported by this budget policy")
        verified[model_id] = {**policy, "context_length": context_length}
    return RunBudget(limit, data_dir / "run-metadata" / f"{date_str}.budget.json", verified,
                     max_output_tokens=settings.get("max_output_tokens", 8192))


def _save_checkpoint(data_dir: Path, date_str: str, state: dict) -> None:
    directory = data_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    # A killed process must leave the last complete checkpoint available.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as file:
        json.dump(state, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, directory / f"{date_str}.json")


async def run_pipeline(*, quick_start: bool = False, check_only: bool = False) -> None:
    setup_logging()
    logger = logging.getLogger("pipeline")
    try:
        config = load_config()
        api_key = get_env("OPENROUTER_API_KEY")
        report_date = os.environ.get("REPORT_DATE", "").strip()
        if report_date and datetime.strptime(report_date, "%Y-%m-%d").strftime("%Y-%m-%d") != report_date:
            raise ValueError("REPORT_DATE must use YYYY-MM-DD")
        maximum = int(os.environ.get("MAX_PAPERS", "0"))
        if maximum < 0:
            raise ValueError("MAX_PAPERS must be non-negative")
        limit = os.environ.get("PAPERS_BUDGET_USD", "").strip()
        if limit and (not Decimal(limit).is_finite() or Decimal(limit) <= 0):
            raise ValueError("PAPERS_BUDGET_USD must be a positive finite amount")
    except (OSError, ValueError, ArithmeticError) as exc:
        logger.error("Setup failed: %s. See README.md for configuration and environment variables.", exc)
        raise SystemExit(2) from None
    data_dir = DATA_DIR / "trial" if quick_start else DATA_DIR
    if check_only:
        logger.info("Local setup check passed; output directory: %s. Network access, API credentials and write permissions have not been verified.", data_dir)
        return
    if quick_start:
        maximum = min(maximum, 2) if maximum else 2
        logger.info("Quick start: at most %d new papers; no PDF downloads, DeepResearch, email or git pushes", maximum)
    date_str = report_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary = {"date": date_str, "query_mode": "submitted-date-utc" if report_date else "hybrid-current", "email_sent": False}
    pdf_config = config.get("pdf", {})
    pdf_dir = None
    if not quick_start and pdf_config.get("download_enabled", False):
        pdf_base = Path(pdf_config.get("storage_dir", "storage/pdfs")).expanduser()
        pdf_dir = (pdf_base if pdf_base.is_absolute() else ROOT_DIR / pdf_base) / date_str
    index_path = data_dir / "papers_index.json"
    try:
        directories = [data_dir, data_dir / "reports", data_dir / "run-metadata"]
        if pdf_dir is not None:
            directories.append(pdf_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory):
                pass
        for path in (index_path, Path(str(index_path) + ".bak"), data_dir / "reports" / f"{date_str}.md", data_dir / "run-metadata" / f"{date_str}.json"):
            if path.exists() and (not path.is_file() or not os.access(path, os.W_OK)):
                raise PermissionError(f"Output file is not writable: {path}")
    except OSError as exc:
        logger.error("Output setup failed before collection: %s", exc)
        raise SystemExit(2) from None
    index = load_index(index_path)
    budget = await _load_run_budget(config, data_dir, date_str)
    budget_token = current_budget.set(budget)
    try:
        await _collect_and_report(config, api_key, report_date, date_str, maximum, quick_start,
                                  data_dir, pdf_dir, pdf_config, index_path, index, summary, budget)
    finally:
        current_budget.reset(budget_token)


async def _collect_and_report(config: dict, api_key: str, report_date: str, date_str: str,
                              maximum: int, quick_start: bool, data_dir: Path, pdf_dir: Path | None,
                              pdf_config: dict, index_path: Path, index: dict, summary: dict,
                              budget: RunBudget | None) -> None:
    logger = logging.getLogger("pipeline")
    logger.info("=== Fetching papers ===")
    candidates = hybrid_fetch(config, report_date or None)
    summary["candidates"] = len(candidates)
    if not candidates:
        save_empty_run(date_str, "no_candidates", config, summary, data_dir)
        return
    new_papers = dedup(candidates, index)
    if maximum:
        new_papers = new_papers[:maximum]
    summary["new_papers"] = len(new_papers)
    if not new_papers:
        save_empty_run(date_str, "already_indexed", config, summary, data_dir)
        return

    analyses: dict[str, AnalysisResult] = {}
    deep_reports: dict[str, str] = {}
    core_papers: list[ArxivPaper] = []
    peripheral_papers: list[ArxivPaper] = []
    budget_stopped = False
    classified = 0

    def checkpoint(stage: str) -> None:
        if budget is not None:
            _save_checkpoint(data_dir, date_str, {
                "date": date_str, "stage": stage, "candidates": [asdict(p) for p in new_papers],
                "classified": classified, "core_ids": [p.arxiv_id for p in core_papers],
                "peripheral_ids": [p.arxiv_id for p in peripheral_papers],
                "analyses": {key: asdict(value) for key, value in analyses.items()},
                "deep_research_reports": deep_reports, "budget": budget.snapshot(),
            })

    checkpoint("fetch_completed")
    try:
        if budget is None:
            core_papers, peripheral_papers = await filter_relevance(new_papers, config["models"]["relevance_filter"], api_key)
            classified = len(new_papers)
        else:
            # Sequential batches preserve classifications before a budget stop;
            # analysis and PDF research still use their configured concurrency.
            batch_size = config["models"]["relevance_filter"].get("batch_size", 10)
            for start in range(0, len(new_papers), batch_size):
                batch = new_papers[start:start + batch_size]
                core, peripheral = await filter_relevance(batch, config["models"]["relevance_filter"], api_key)
                core_papers.extend(core)
                peripheral_papers.extend(peripheral)
                classified += len(batch)
                checkpoint("filtering")
    except BudgetExceeded as exc:
        logger.warning("Budget stopped filtering: %s", exc)
        budget_stopped = True
    budget_stopped = budget_stopped or (budget is not None and budget.snapshot()["stopped"])
    summary["classified"] = classified
    checkpoint("filter_completed")
    all_relevant = core_papers + peripheral_papers
    if not all_relevant and not budget_stopped:
        save_empty_run(date_str, "no_relevant_papers", config, summary, data_dir)
        return

    sem = asyncio.Semaphore(config.get("concurrency", {}).get("max_concurrent_llm", 10))
    async def analyze_one(paper: ArxivPaper) -> None:
        async with sem:
            analyses[paper.arxiv_id] = await analyze_paper(paper, config["models"]["deep_analysis"], config["scoring"], api_key)
            logger.info("Analysis completed %s", paper.arxiv_id)
            checkpoint("analysis")

    if not budget_stopped:
        results = await asyncio.gather(*(analyze_one(p) for p in all_relevant), return_exceptions=True)
        for result in results:
            if isinstance(result, BudgetExceeded):
                budget_stopped = True
            elif isinstance(result, BaseException):
                raise result
    budget_stopped = budget_stopped or (budget is not None and budget.snapshot()["stopped"])
    completed_core = [p for p in core_papers if p.arxiv_id in analyses]
    completed_peripheral = [p for p in peripheral_papers if p.arxiv_id in analyses]
    downloaded = {}
    if pdf_dir is not None and not budget_stopped:
        downloaded = await download_all_pdfs(completed_core, pdf_dir)
        pdf_config = {**pdf_config, "storage_dir": str(pdf_dir)}
    checkpoint("analysis_completed")

    skip_research = quick_start or os.environ.get("SKIP_DEEP_RESEARCH", "").strip().lower() in ("1", "true", "yes")
    if not skip_research and not budget_stopped:
        sem_pdf = asyncio.Semaphore(config.get("concurrency", {}).get("max_concurrent_pdf", 5))
        async def research_one(paper: ArxivPaper) -> None:
            async with sem_pdf:
                report = await generate_deep_research(paper, config["models"]["deep_analysis"], api_key,
                    pdf_config=pdf_config if pdf_config.get("download_enabled") else None)
                if report:
                    deep_reports[paper.arxiv_id] = report
                logger.info("DeepResearch completed %s (%d chars)", paper.arxiv_id, len(report))
                checkpoint("deep_research")
        results = await asyncio.gather(*(research_one(p) for p in completed_core), return_exceptions=True)
        for result in results:
            if isinstance(result, BudgetExceeded):
                budget_stopped = True
            elif isinstance(result, BaseException):
                raise result

    budget_stopped = budget_stopped or (budget is not None and budget.snapshot()["stopped"])
    status = "budget_exhausted" if budget_stopped else "completed"
    # A partial run must not hide unfinished papers from a later full run.
    # Checkpoints and the partial report retain completed paid results separately.
    if not budget_stopped:
        core_ids = {p.arxiv_id for p in completed_core}
        for paper in completed_core + completed_peripheral:
            url = release_pdf_url(date_str, paper.arxiv_id) if paper.arxiv_id in downloaded else ""
            entry = paper_to_index_entry(paper, analyses[paper.arxiv_id], release_pdf_url=url)
            entry["relevance_tier"] = "core" if paper.arxiv_id in core_ids else "peripheral"
            index[paper.arxiv_id] = entry
        save_index(index, index_path)
    report_config = {**config, "downloaded_pdf_ids": list(downloaded), "run_status": status}
    save_report(generate_daily_report(date_str, completed_core, completed_peripheral, analyses, deep_reports, report_config), date_str, data_dir)

    skip_email = quick_start or budget_stopped or os.environ.get("SKIP_EMAIL", "").strip().lower() in ("1", "true", "yes")
    mail_keys = ("MAIL_SMTP_HOST", "MAIL_SMTP_PORT", "MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_FROM", "MAIL_TO")
    mail = {key: get_env(key, required=False) for key in mail_keys}
    if not skip_email and all(mail.values()):
        summary["email_sent"] = send_digest(
            subject=f"arXiv Daily -- {date_str} -- {len(analyses)} papers",
            html_body=generate_email_html(date_str, completed_core, analyses, config),
            to_addr=mail["MAIL_TO"], from_addr=mail["MAIL_FROM"], username=mail["MAIL_USERNAME"],
            password=mail["MAIL_PASSWORD"], smtp_host=mail["MAIL_SMTP_HOST"], smtp_port=mail["MAIL_SMTP_PORT"],
        )
    summary.update(status=status, core=len(completed_core), peripheral=len(completed_peripheral),
                   relevant=len(all_relevant), analyzed=len(analyses), pdf_downloaded=len(downloaded),
                   deep_research_completed=len(deep_reports),
                   deep_research_pending=0 if skip_research else len(core_papers) - len(deep_reports))
    if budget is not None:
        summary["budget"] = budget.snapshot()
    save_run_summary(date_str, summary, data_dir)
    checkpoint(status)
    skip_push = quick_start or budget_stopped or os.environ.get("SKIP_GIT_PUSH", "").strip().lower() in ("1", "true", "yes")
    if os.environ.get("GITHUB_ACTIONS") and not skip_push:
        commit_and_push_data(date_str)
    logger.info("Pipeline %s: %d analyzed, %d deep research", status, len(analyses), len(deep_reports))
    if budget_stopped:
        raise SystemExit(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an arXiv paper report locally or from a scheduler.")
    parser.add_argument("--check", action="store_true", help="Check local configuration without network requests or file writes")
    parser.add_argument("--quick-start", action="store_true", default=os.environ.get("PAPERS_QUICK_START", "").strip().lower() in ("1", "true", "yes"), help="Process at most two new papers in a separate trial directory without PDFs, DeepResearch, email or git pushes; model API calls still incur charges")
    args = parser.parse_args()
    asyncio.run(run_pipeline(quick_start=args.quick_start, check_only=args.check))


if __name__ == "__main__":
    main()
