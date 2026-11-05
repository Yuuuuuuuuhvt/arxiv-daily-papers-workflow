from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from src import main, deep_analysis, deep_research, relevance_filter
from src.budget import BudgetExceeded
from src.config import load_config
from src.models import AnalysisResult, ArxivPaper


def test_budget_stop_preserves_completed_work_and_escapes_fallbacks(monkeypatch, tmp_path):
    config = load_config()
    config["concurrency"]["max_concurrent_llm"] = 1
    config["pdf"]["storage_dir"] = str(tmp_path / "pdfs")
    monkeypatch.setattr(main, "load_config", lambda: config)
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("REPORT_DATE", "2026-11-04")
    monkeypatch.setenv("MAX_PAPERS", "0")
    monkeypatch.setenv("SKIP_DEEP_RESEARCH", "0")
    monkeypatch.delenv("PAPERS_BUDGET_USD", raising=False)
    day = datetime(2026, 11, 4, tzinfo=timezone.utc)
    papers = [ArxivPaper(arxiv_id=f"example-{i}", version=1, title=f"Synthetic example {i}",
        abstract="Synthetic input", authors=[], categories=["cs.RO"], primary_category="cs.RO",
        published=day, updated=day, pdf_url="https://example.test/paper.pdf", abs_url="https://example.test/abstract")
        for i in range(2)]
    class Budget:
        stopped = False

        def snapshot(self):
            return {"limit_usd": "1", "spent_usd": "0.9", "stopped": self.stopped}
    run_budget = Budget()
    async def budget(*args):
        return run_budget
    async def classify(*args):
        return papers, []
    async def analyze(paper, *args):
        if paper.arxiv_id == "example-1":
            run_budget.stopped = True
            raise BudgetExceeded("Remaining budget cannot cover the next request")
        return AnalysisResult(one_line_summary="已完成的分析", detailed_summary="保留的具体分析", direction="robotics")
    def unexpected(*args, **kwargs):
        raise AssertionError("No PDF, email or publishing after budget stop")
    monkeypatch.setattr(main, "_load_run_budget", budget)
    monkeypatch.setattr(main, "hybrid_fetch", lambda *args: papers)
    monkeypatch.setattr(main, "filter_relevance", classify)
    monkeypatch.setattr(main, "analyze_paper", analyze)
    for name in ("download_all_pdfs", "send_digest", "commit_and_push_data"):
        monkeypatch.setattr(main, name, unexpected)
    with pytest.raises(SystemExit) as stopped:
        asyncio.run(main.run_pipeline())
    assert stopped.value.code == 3
    report = (tmp_path / "reports/2026-11-04.md").read_text()
    assert "预算不足" in report and "已完成的分析" in report
    assert "Synthetic example 1" not in report
    checkpoint = json.loads((tmp_path / "checkpoints/2026-11-04.json").read_text())
    assert list(checkpoint["analyses"]) == ["example-0"]
    assert not (tmp_path / "papers_index.json").exists()
    assert json.loads((tmp_path / "run-metadata/2026-11-04.json").read_text())["status"] == "budget_exhausted"

    async def exhausted(**kwargs):
        raise BudgetExceeded("Budget exhausted before sending")
    for module in (relevance_filter, deep_analysis, deep_research):
        monkeypatch.setattr(module, "call_llm", exhausted)
    for operation in (
        lambda: relevance_filter.filter_relevance(papers[:1], config["models"]["relevance_filter"], "test-key"),
        lambda: deep_analysis.analyze_paper(papers[0], config["models"]["deep_analysis"], config["scoring"], "test-key"),
        lambda: deep_research.generate_deep_research(papers[0], config["models"]["deep_analysis"], "test-key"),
    ):
        with pytest.raises(BudgetExceeded):
            asyncio.run(operation())
