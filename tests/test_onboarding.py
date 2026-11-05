from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest
import yaml

from src import config, main, report_generator
from src.models import AnalysisResult, ArxivPaper


def test_check_rejects_bad_setup_without_collection(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("Setup checks must not collect papers")
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    monkeypatch.setattr(main, "hybrid_fetch", unexpected)
    monkeypatch.setattr(main, "DATA_DIR", tmp_path / "missing")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MAX_PAPERS", "2")
    monkeypatch.delenv("REPORT_DATE", raising=False)
    asyncio.run(main.run_pipeline(check_only=True))
    assert not (tmp_path / "missing").exists()
    for key, value in [("OPENROUTER_API_KEY", ""), ("MAX_PAPERS", "-1"), ("REPORT_DATE", "2026-02-30")]:
        with monkeypatch.context() as case:
            case.setenv(key, value)
            with pytest.raises(SystemExit) as error:
                asyncio.run(main.run_pipeline(check_only=True))
            assert error.value.code == 2


def test_external_paths_and_invalid_yaml(monkeypatch, tmp_path):
    original = config.load_config()
    settings = tmp_path / "instance.yaml"
    settings.write_text(yaml.safe_dump(original), encoding="utf-8")
    env = {**os.environ, "PAPERS_CONFIG": str(settings), "PAPERS_DATA_DIR": str(tmp_path / "state")}
    result = subprocess.run(
        [sys.executable, "-c", "from src.config import DATA_DIR, load_config; print(DATA_DIR); print(load_config()['models']['relevance_filter']['model_id'])"],
        env=env, cwd=config.ROOT_DIR, capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == [str(tmp_path / "state"), original["models"]["relevance_filter"]["model_id"]]
    monkeypatch.setenv("PAPERS_CONFIG", str(settings))
    for text in ["- not-a-mapping", "models: [", "models: {}"]:
        settings.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            config.load_config()
    original["concurrency"]["max_concurrent_llm"] = 0
    settings.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(ValueError, match="max_concurrent_llm"):
        config.load_config()


def test_trial_writes_real_report_and_index_without_external_side_effects(monkeypatch, tmp_path):
    data_dir = tmp_path / "new-data-directory"
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(report_generator, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    for key, value in {
        "OPENROUTER_API_KEY": "test-key", "REPORT_DATE": "2026-11-04", "MAX_PAPERS": "0",
        "GITHUB_ACTIONS": "true", "SKIP_GIT_PUSH": "0", "SKIP_EMAIL": "0", "SKIP_DEEP_RESEARCH": "0",
        "MAIL_SMTP_HOST": "example.test", "MAIL_SMTP_PORT": "587", "MAIL_USERNAME": "test",
        "MAIL_PASSWORD": "test", "MAIL_FROM": "test@example.test", "MAIL_TO": "reader@example.test",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PDF_RELEASE_BASE_URL", raising=False)
    day = datetime(2026, 11, 4, tzinfo=timezone.utc)
    papers = [ArxivPaper(
        arxiv_id=f"fixture-{i}", version=1, title=f"Synthetic paper {i}", abstract="Test input",
        authors=["Test author"], categories=["cs.RO"], primary_category="cs.RO", published=day, updated=day,
        pdf_url=f"https://example.test/{i}.pdf", abs_url=f"https://example.test/{i}",
    ) for i in range(4)]
    monkeypatch.setattr(main, "hybrid_fetch", lambda *args: papers)
    async def relevance(candidates, *args):
        assert len(candidates) == 2
        return candidates, []
    async def analyze(paper, *args):
        return AnalysisResult(one_line_summary=paper.title, direction="embodied_ai", weighted_score=4.0)
    def unexpected(*args, **kwargs):
        raise AssertionError("Trial must suppress publishing, PDFs, extended research and email")
    monkeypatch.setattr(main, "filter_relevance", relevance)
    monkeypatch.setattr(main, "analyze_paper", analyze)
    for name in ("download_all_pdfs", "generate_deep_research", "send_digest", "commit_and_push_data"):
        monkeypatch.setattr(main, name, unexpected)
    asyncio.run(main.run_pipeline(quick_start=True))
    trial_dir = data_dir / "trial"
    report = (trial_dir / "reports/2026-11-04.md").read_text(encoding="utf-8")
    index = json.loads((trial_dir / "papers_index.json").read_text())
    summary = json.loads((trial_dir / "run-metadata/2026-11-04.json").read_text())
    assert "Synthetic paper 0" in report and "Synthetic paper 1" in report
    assert "Synthetic paper 2" not in report and "git.luonto.me" not in report
    assert index["_meta"]["total_papers"] == 2
    assert all(index[paper.arxiv_id]["release_pdf_url"] == "" for paper in papers[:2])
    assert summary["status"] == "completed" and summary["email_sent"] is False
    asyncio.run(main.run_pipeline(quick_start=True))
    assert json.loads((trial_dir / "papers_index.json").read_text())["_meta"]["total_papers"] == 4
    assert not (data_dir / "papers_index.json").exists()
    assert not (data_dir / "reports").exists()


def test_unwritable_output_fails_before_collection(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    monkeypatch.setattr(main, "DATA_DIR", tmp_path / "output")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MAX_PAPERS", "0")
    monkeypatch.delenv("REPORT_DATE", raising=False)
    def denied(*args, **kwargs):
        raise PermissionError("Read-only filesystem")
    def unexpected(*args, **kwargs):
        raise AssertionError("Storage must be writable before collection")
    monkeypatch.setattr(main.tempfile, "TemporaryFile", denied)
    monkeypatch.setattr(main, "hybrid_fetch", unexpected)
    with pytest.raises(SystemExit) as error:
        asyncio.run(main.run_pipeline(quick_start=True))
    assert error.value.code == 2
