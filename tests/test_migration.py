import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src import fetcher, report_generator
from src.config import release_pdf_url
from src.models import AnalysisResult


def test_backfill_uses_submission_date_and_all_pages(monkeypatch):
    queries = []
    class Search:
        def __init__(self, **kwargs):
            queries.append(kwargs)
    class Client:
        def __init__(self, **kwargs):
            pass
        def results(self, search):
            return iter(())
    monkeypatch.setattr(fetcher.arxiv, "Search", Search)
    monkeypatch.setattr(fetcher.arxiv, "Client", Client)
    monkeypatch.setattr(fetcher.time, "sleep", lambda delay: None)
    fetcher.fetch_api({"arxiv": {}, "research_directions": {"test": {"title_keywords": ["world model"]}}}, "2026-10-09")
    assert "submittedDate:[202610090000 TO 202610092359]" in queries[0]["query"]
    assert queries[0]["max_results"] is None


def test_backfill_rejects_wrong_date_and_never_uses_current_rss(monkeypatch):
    papers = {"one": SimpleNamespace(published=datetime(2026, 10, 9, tzinfo=timezone.utc)),
              "two": SimpleNamespace(published=datetime(2026, 10, 10, tzinfo=timezone.utc))}
    monkeypatch.setattr(fetcher, "fetch_api", lambda config, day: papers)
    def unexpected(config):
        raise AssertionError("Historical run must not read current RSS")
    monkeypatch.setattr(fetcher, "fetch_rss", unexpected)
    assert fetcher.hybrid_fetch({}, "2026-10-09") == [papers["one"]]


def test_backfill_failure_is_not_an_empty_day(monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass
        def results(self, search):
            raise ConnectionError("network unavailable")
    monkeypatch.setattr(fetcher.arxiv, "Client", Client)
    with pytest.raises(ConnectionError):
        fetcher.fetch_api({"arxiv": {}, "research_directions": {"test": {"title_keywords": ["robot"]}}}, "2026-10-09")


def test_release_url_uses_real_report_date():
    paper = SimpleNamespace(arxiv_id="2610.01234", title="Paper", authors=[], abs_url="https://arxiv.org/abs/2610.01234", pdf_url="https://arxiv.org/pdf/2610.01234")
    view = report_generator._paper_view(paper, AnalysisResult(), "https://example.test/{arxiv_id}", "2026-10-09", "research")
    assert view["release_pdf_url"] == release_pdf_url("2026-10-09", paper.arxiv_id)
    assert view["release_pdf_url"].startswith("https://git.luonto.me/")
    assert view["deep_research"] == "research"

    assert report_generator._paper_view(paper, AnalysisResult(), "https://example.test/{arxiv_id}", "2026-10-09", pdf_available=False)["release_pdf_url"] == ""


def test_empty_retry_preserves_existing_report_and_summary(monkeypatch, tmp_path):
    from src import main
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    reports = tmp_path / "reports"
    metadata = tmp_path / "run-metadata"
    reports.mkdir()
    metadata.mkdir()
    report = reports / "2026-10-09.md"
    summary = metadata / "2026-10-09.json"
    report.write_text("Existing complete analysis\n")
    summary.write_text('{"status":"completed","core":32}\n')
    main.save_empty_run("2026-10-09", "already_indexed", {}, {"new_papers": 0})
    assert report.read_text() == "Existing complete analysis\n"
    assert summary.read_text() == '{"status":"completed","core":32}\n'
    summary.unlink()
    main.save_empty_run("2026-10-09", "already_indexed", {}, {"new_papers": 0})
    assert report.read_text() == "Existing complete analysis\n"
    assert json.loads(summary.read_text()) == {
        "new_papers": 0, "status": "already_indexed", "retained_existing_report": True
    }
