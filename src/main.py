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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .budget import BudgetExceeded, RunBudget, current_budget
from .config import DATA_DIR, MODEL_STAGES, ROOT_DIR, load_config, get_env, provider_api_key, release_pdf_url
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
    save_reading_report,
    save_report_index,
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
    save_report_index(directory.parent)


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
    summary.update(status=status, analyzed=0, core=0, peripheral=0, pdf_downloaded=0,
                   deep_research_completed=0, deep_research_pending=0)
    budget = current_budget.get()
    if budget is not None:
        summary["budget"] = budget.snapshot()
    state = {"date": date_str, "stage": status, "candidates": [], "core_ids": [],
             "peripheral_ids": [], "analyses": {}, "deep_research_reports": {}}
    save_report(generate_daily_report(date_str, [], [], {}, {}, config), date_str, data_dir)
    save_reading_report(state, summary, data_dir)
    _save_checkpoint(data_dir, date_str, state)
    save_run_summary(date_str, summary, data_dir)


def resolve_report_date(now: datetime | None = None) -> str:
    mode = os.environ.get("PAPERS_DATE_MODE", "current").strip()
    if mode not in ("current", "previous-day"):
        raise ValueError("PAPERS_DATE_MODE must be current or previous-day")
    date = os.environ.get("REPORT_DATE", "").strip()
    if date:
        if datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d") != date:
            raise ValueError("REPORT_DATE must use YYYY-MM-DD")
        return date
    if mode == "previous-day":
        return ((now or datetime.now(timezone.utc)).astimezone(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    return ""


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
        supports_images = "image" in model.get("architecture", {}).get("input_modalities", [])
        image_cap = Decimal(str(policy.get("image_price_per_image", prompt_cap)))
        if not image_cap.is_finite() or image_cap < 0:
            raise ValueError("image_price_per_image must be finite and non-negative")
        for tier in [pricing, *pricing.get("overrides", [])]:
            image_price = Decimal(str(tier.get("image", pricing.get("image", "0"))))
            if not image_price.is_finite() or image_price < 0 or image_price > image_cap:
                supports_images = False
        # The token reservation covers the whole input context. Also reserve
        # per-image fees and enforce that cap at provider routing.
        verified[model_id] = {**policy, "context_length": context_length,
                              "supports_images": supports_images,
                              "image_price_per_image": image_cap}
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
        providers = {stage: config["models"][stage].get("provider", "openrouter") for stage in MODEL_STAGES}
        api_keys = {stage: provider_api_key(provider) for stage, provider in providers.items()}
        report_date = resolve_report_date()
        maximum = int(os.environ.get("MAX_PAPERS", "0"))
        if maximum < 0:
            raise ValueError("MAX_PAPERS must be non-negative")
        limit = os.environ.get("PAPERS_BUDGET_USD", "").strip()
        if limit and (not Decimal(limit).is_finite() or Decimal(limit) <= 0):
            raise ValueError("PAPERS_BUDGET_USD must be a positive finite amount")
        if limit and set(providers.values()) != {"openrouter"}:
            # A relay reports no request cost, so the ledger could not bound it.
            raise ValueError("PAPERS_BUDGET_USD only supports the openrouter provider")
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
    # A timer catch-up or manual duplicate must not create a second paid run.
    summary_path = data_dir / "run-metadata" / f"{date_str}.json"
    if not quick_start and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") in {"completed", "no_candidates", "no_relevant_papers", "already_indexed"} and all(
            (data_dir / "reports" / f"{date_str}.{suffix}").is_file() for suffix in ("md", "html")
        ):
            logger.info("Daily report for %s already completed; no requests made", date_str)
            return
    summary = {"date": date_str, "query_mode": "submitted-date-utc" if report_date else "announced-rss", "email_sent": False}
    pdf_config = config.get("pdf", {})
    pdf_dir = None
    if not quick_start and pdf_config.get("download_enabled", False):
        pdf_base = Path(pdf_config.get("storage_dir", "storage/pdfs")).expanduser()
        pdf_dir = (pdf_base if pdf_base.is_absolute() else ROOT_DIR / pdf_base) / date_str
    index_path = data_dir / "papers_index.json"
    try:
        directories = [data_dir, data_dir / "reports", data_dir / "run-metadata"]
        if os.environ.get("PAPERS_WEB_REPORTS", "").strip() == "1":
            directories.append(data_dir / "reports" / "papers")
        if pdf_dir is not None:
            directories.append(pdf_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory):
                pass
        for path in (index_path, Path(str(index_path) + ".bak"), data_dir / "reports" / f"{date_str}.md",
                     data_dir / "reports" / f"{date_str}.html", data_dir / "reports" / "latest.html",
                     data_dir / "run-metadata" / f"{date_str}.json"):
            if path.exists() and (not path.is_file() or not os.access(path, os.W_OK)):
                raise PermissionError(f"Output file is not writable: {path}")
    except OSError as exc:
        logger.error("Output setup failed before collection: %s", exc)
        raise SystemExit(2) from None
    index = load_index(index_path)
    budget = await _load_run_budget(config, data_dir, date_str)
    budget_token = current_budget.set(budget)
    try:
        await _collect_and_report(config, api_keys, report_date, date_str, maximum, quick_start,
                                  data_dir, pdf_dir, pdf_config, index_path, index, summary, budget)
    finally:
        current_budget.reset(budget_token)


async def _collect_and_report(config: dict, api_keys: dict[str, str], report_date: str, date_str: str,
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
    errors: list[dict] = []

    def report_state(stage: str) -> dict:
        return {
                "date": date_str, "stage": stage, "candidates": [asdict(p) for p in new_papers],
                "classified": classified, "core_ids": [p.arxiv_id for p in core_papers],
                "peripheral_ids": [p.arxiv_id for p in peripheral_papers],
                "analyses": {key: asdict(value) for key, value in analyses.items()},
                "deep_research_reports": deep_reports,
                "budget": budget.snapshot() if budget is not None else None,
            }

    def checkpoint(stage: str) -> None:
        _save_checkpoint(data_dir, date_str, report_state(stage))

    checkpoint("fetch_completed")
    try:
        if budget is None:
            core_papers, peripheral_papers = await filter_relevance(new_papers, config["models"]["relevance_filter"], api_keys["relevance_filter"])
            classified = len(new_papers)
        else:
            # Sequential batches preserve classifications before a budget stop;
            # analysis and PDF research still use their configured concurrency.
            batch_size = config["models"]["relevance_filter"].get("batch_size", 10)
            for start in range(0, len(new_papers), batch_size):
                batch = new_papers[start:start + batch_size]
                core, peripheral = await filter_relevance(batch, config["models"]["relevance_filter"], api_keys["relevance_filter"])
                core_papers.extend(core)
                peripheral_papers.extend(peripheral)
                classified += len(batch)
                checkpoint("filtering")
    except BudgetExceeded as exc:
        logger.warning("Budget stopped filtering: %s", exc)
        budget_stopped = True
    except Exception as exc:
        errors.append({"stage": "filter", "error": type(exc).__name__})
        logger.error("Filtering failed: %s", type(exc).__name__)
    budget_stopped = budget_stopped or (budget is not None and budget.snapshot()["stopped"])
    summary["classified"] = classified
    checkpoint("filter_completed")
    all_relevant = core_papers + peripheral_papers
    if not all_relevant and not budget_stopped and not errors:
        save_empty_run(date_str, "no_relevant_papers", config, summary, data_dir)
        return

    sem = asyncio.Semaphore(config.get("concurrency", {}).get("max_concurrent_llm", 10))
    async def analyze_one(paper: ArxivPaper) -> None:
        async with sem:
            analyses[paper.arxiv_id] = await analyze_paper(paper, config["models"]["deep_analysis"], config["scoring"], api_keys["deep_analysis"])
            logger.info("Analysis completed %s", paper.arxiv_id)
            checkpoint("analysis")

    if not budget_stopped:
        results = await asyncio.gather(*(analyze_one(p) for p in all_relevant), return_exceptions=True)
        for paper, result in zip(all_relevant, results):
            if isinstance(result, BudgetExceeded):
                budget_stopped = True
            elif isinstance(result, Exception):
                errors.append({"stage": "analysis", "arxiv_id": paper.arxiv_id, "error": type(result).__name__})
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
                report = await generate_deep_research(paper, config["models"]["deep_analysis"], api_keys["deep_analysis"],
                    pdf_config=pdf_config if pdf_config.get("download_enabled") else None,
                    source_config={"storage_dir": str(data_dir / "paper-inputs"),
                                   **config.get("paper_input", {})})
                if report:
                    deep_reports[paper.arxiv_id] = report
                logger.info("DeepResearch completed %s (%d chars)", paper.arxiv_id, len(report))
                checkpoint("deep_research")
        results = await asyncio.gather(*(research_one(p) for p in completed_core), return_exceptions=True)
        for paper, result in zip(completed_core, results):
            if isinstance(result, BudgetExceeded):
                budget_stopped = True
            elif isinstance(result, Exception):
                errors.append({"stage": "deep_research", "arxiv_id": paper.arxiv_id, "error": type(result).__name__})
            elif isinstance(result, BaseException):
                raise result

    budget_stopped = budget_stopped or (budget is not None and budget.snapshot()["stopped"])
    pending = 0 if skip_research else len(core_papers) - len(deep_reports)
    status = "budget_exhausted" if budget_stopped else "partial" if errors or pending else "completed"
    # A partial run must not hide unfinished papers from a later full run.
    # Checkpoints and the partial report retain completed paid results separately.
    if status == "completed":
        core_ids = {p.arxiv_id for p in completed_core}
        for paper in completed_core + completed_peripheral:
            url = release_pdf_url(date_str, paper.arxiv_id) if paper.arxiv_id in downloaded else ""
            entry = paper_to_index_entry(paper, analyses[paper.arxiv_id], release_pdf_url=url)
            entry["relevance_tier"] = "core" if paper.arxiv_id in core_ids else "peripheral"
            index[paper.arxiv_id] = entry
        save_index(index, index_path)
    report_config = {**config, "downloaded_pdf_ids": list(downloaded), "run_status": status}
    save_report(generate_daily_report(date_str, completed_core, completed_peripheral, analyses, deep_reports, report_config), date_str, data_dir)

    summary.update(status=status, core=len(completed_core), peripheral=len(completed_peripheral),
                   relevant=len(all_relevant), analyzed=len(analyses), pdf_downloaded=len(downloaded),
                   deep_research_completed=len(deep_reports), deep_research_pending=pending,
                   deep_research_skipped=skip_research, errors=errors)
    for field, prefix in (("pdf_native_count", "> 输入依据：PDF 原文"),
                          ("latex_image_count", "> 输入依据：LaTeX 正文与引用图片"),
                          ("pdf_page_image_count", "> 输入依据：PDF 全部页面图像"),
                          ("pdf_text_count", "> 输入依据：PDF 全部页面提取的正文"),
                          ("pdf_abstract_count", "> 输入依据：仅论文摘要")):
        summary[field] = sum(report.startswith(prefix) for report in deep_reports.values())
    if budget is not None:
        summary["budget"] = budget.snapshot()
    save_reading_report(report_state(status), summary, data_dir)
    save_run_summary(date_str, summary, data_dir)
    checkpoint(status)

    skip_email = quick_start or status != "completed" or os.environ.get("SKIP_EMAIL", "").strip().lower() in ("1", "true", "yes")
    mail_keys = ("MAIL_SMTP_HOST", "MAIL_SMTP_PORT", "MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_FROM", "MAIL_TO")
    mail = {key: get_env(key, required=False) for key in mail_keys}
    if not skip_email and all(mail.values()):
        summary["email_sent"] = send_digest(
            subject=f"arXiv Daily -- {date_str} -- {len(analyses)} papers",
            html_body=generate_email_html(date_str, completed_core, analyses, config),
            to_addr=mail["MAIL_TO"], from_addr=mail["MAIL_FROM"], username=mail["MAIL_USERNAME"],
            password=mail["MAIL_PASSWORD"], smtp_host=mail["MAIL_SMTP_HOST"], smtp_port=mail["MAIL_SMTP_PORT"],
        )
    save_run_summary(date_str, summary, data_dir)
    skip_push = quick_start or status != "completed" or os.environ.get("SKIP_GIT_PUSH", "").strip().lower() in ("1", "true", "yes")
    if os.environ.get("GITHUB_ACTIONS") and not skip_push:
        commit_and_push_data(date_str)
    logger.info("Pipeline %s: %d analyzed, %d deep research", status, len(analyses), len(deep_reports))
    if budget_stopped:
        raise SystemExit(3)
    if status == "partial":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an arXiv paper report locally or from a scheduler.")
    parser.add_argument("--check", action="store_true", help="Check local configuration without network requests or file writes")
    parser.add_argument("--quick-start", action="store_true", default=os.environ.get("PAPERS_QUICK_START", "").strip().lower() in ("1", "true", "yes"), help="Process at most two new papers in a separate trial directory without PDFs, DeepResearch, email or git pushes; model API calls still incur charges")
    args = parser.parse_args()
    asyncio.run(run_pipeline(quick_start=args.quick_start, check_only=args.check))


if __name__ == "__main__":
    main()
