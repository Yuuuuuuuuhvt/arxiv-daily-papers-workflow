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


def test_pdf_transport_and_strict_source_handling(monkeypatch, tmp_path):
    day = datetime(2026, 11, 4, tzinfo=timezone.utc)
    paper = ArxivPaper("sample", 1, "Synthetic paper", "Abstract", [], ["cs.RO"], "cs.RO",
                       day, day, "https://arxiv.org/pdf/sample", "https://arxiv.org/abs/sample")
    config = {"download_enabled": True, "require_pdf": True, "storage_dir": str(tmp_path), "transport": "url"}
    model = {"model_id": "test"}
    async def inspect(**kwargs):
        assert kwargs["plugins"] == [{"id": "file-parser", "pdf": {"engine": "native"}}]
        assert kwargs["messages"][1]["content"][1]["file"]["file_data"] == paper.pdf_url
        return "Full document output"
    monkeypatch.setattr(deep_research, "call_llm", inspect)
    with pytest.raises(FileNotFoundError):
        asyncio.run(deep_research.generate_deep_research(paper, model, "test", config))
    (tmp_path / "sample.pdf").write_bytes(b"%PDF-synthetic")
    result = asyncio.run(deep_research.generate_deep_research(paper, model, "test", config))
    assert "PDF 原文" in result
    async def failing(**kwargs):
        if "plugins" in kwargs:
            raise RuntimeError("PDF unavailable")
        return "Abstract output"
    monkeypatch.setattr(deep_research, "call_llm", failing)
    with pytest.raises(RuntimeError, match="PDF unavailable"):
        asyncio.run(deep_research.generate_deep_research(paper, model, "test", config))
    result = asyncio.run(deep_research.generate_deep_research(paper, model, "test", {**config, "require_pdf": False}))
    assert "仅论文摘要" in result and "Abstract output" in result


def test_large_pdf_reads_every_page_and_rejects_empty_text(monkeypatch, tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    for text in ("First page evidence", "Last page evidence"):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica")})})})
        stream = DecodedStreamObject()
        placement = "0 1 -1 0 300 300 Tm" if text.startswith("Last") else "40 720 Td"
        stream.set_data(f"BT /F1 12 Tf {placement} ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    writer.write(path)
    day = datetime(2026, 11, 4, tzinfo=timezone.utc)
    paper = ArxivPaper("sample", 1, "Synthetic", "Abstract only", [], [], "cs.RO",
                       day, day, "https://arxiv.org/pdf/sample", "https://arxiv.org/abs/sample")
    async def inspect(**kwargs):
        assert "plugins" not in kwargs and kwargs["stream"] is True
        content = kwargs["messages"][1]["content"]
        assert "First page evidence" in content and "Last page evidence" in content
        return "Full text result"
    monkeypatch.setattr(deep_research, "call_llm", inspect)
    monkeypatch.setattr(deep_research, "_MAX_NATIVE_PDF_BYTES", 5)
    config = {"download_enabled": True, "require_pdf": True, "storage_dir": str(tmp_path)}
    result = asyncio.run(deep_research.generate_deep_research(paper, {"model_id": "test"}, "test", config))
    assert "PDF 全部页面" in result and "未直接输入图像像素" in result
    blank = PdfWriter()
    blank.add_blank_page(width=612, height=792)
    blank.write(path)
    with pytest.raises(ValueError, match="no extractable text"):
        asyncio.run(deep_research.generate_deep_research(paper, {"model_id": "test"}, "test", config))


def test_partial_pdf_failure_still_saves_readable_results(monkeypatch, tmp_path):
    config = load_config()
    config["pdf"]["download_enabled"] = False
    monkeypatch.setattr(main, "load_config", lambda: config)
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    for key, value in {"OPENROUTER_API_KEY": "test", "REPORT_DATE": "2026-11-04",
                       "MAX_PAPERS": "0", "SKIP_DEEP_RESEARCH": "0", "SKIP_EMAIL": "0",
                       "SKIP_GIT_PUSH": "0", "GITHUB_ACTIONS": "true"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PAPERS_BUDGET_USD", raising=False)
    day = datetime(2026, 11, 4, tzinfo=timezone.utc)
    paper = ArxivPaper("example", 1, "Synthetic", "Abstract", [], [], "cs.RO",
                       day, day, "https://example.test/a.pdf", "https://example.test/a")
    monkeypatch.setattr(main, "hybrid_fetch", lambda *args: [paper])
    async def classify(*args):
        return [paper], []
    async def analyze(*args):
        return AnalysisResult(one_line_summary="保存的摘要分析", direction="robotics")
    async def research(*args, **kwargs):
        raise FileNotFoundError("PDF not available")
    def unexpected(*args, **kwargs):
        raise AssertionError("Partial runs must not publish")
    monkeypatch.setattr(main, "filter_relevance", classify)
    monkeypatch.setattr(main, "analyze_paper", analyze)
    monkeypatch.setattr(main, "generate_deep_research", research)
    monkeypatch.setattr(main, "send_digest", unexpected)
    monkeypatch.setattr(main, "commit_and_push_data", unexpected)
    with pytest.raises(SystemExit) as stopped:
        asyncio.run(main.run_pipeline())
    assert stopped.value.code == 2
    summary = json.loads((tmp_path / "run-metadata/2026-11-04.json").read_text())
    assert summary["status"] == "partial" and summary["deep_research_pending"] == 1
    assert summary["errors"][0]["error"] == "FileNotFoundError"
    assert "保存的摘要分析" in (tmp_path / "reports/2026-11-04.html").read_text()
    assert not (tmp_path / "papers_index.json").exists()
    assert (tmp_path / "checkpoints/2026-11-04.json").is_file()
