from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from src import fetcher
from src.models import ArxivPaper


def config():
    return {"arxiv": {"api_delay_seconds": 0, "api_max_results_per_direction": 50},
            "research_directions": {"scene_reconstruction": {
                "title_keywords": ["Gaussian splatting"], "categories": ["cs.GR"],
                "abstract_combos": [["scene", "simulation"]]}}}


def announcement(number, **changes):
    values = dict(arxiv_id=f"2611.{number:05}", version=1, title="3D Gaussian splatting",
                  abstract="Scene reconstruction", authors=[], categories=["cs.GR"],
                  primary_category="cs.GR", published=datetime(2026, 11, 9, tzinfo=timezone.utc),
                  updated=datetime(2026, 11, 9, tzinfo=timezone.utc),
                  pdf_url="", abs_url="", announce_type="new", rss_pub_date="Monday")
    return ArxivPaper(**(values | changes))


def test_announced_batch_has_no_search_cap_and_keeps_source_identity(monkeypatch):
    papers = {p.arxiv_id: p for p in [announcement(i) for i in range(205)]}
    for p in [announcement(900, announce_type="replace"),
              announcement(901, title="Gaussian splatting", categories=["math.OC"]),
              announcement(902, title="Clinical reconstruction")]:
        papers[p.arxiv_id] = p
    papers["2611.00000"].announce_type = "cross"
    monkeypatch.setattr(fetcher, "fetch_rss", lambda cfg: (papers, {}))
    searches = []
    class Client:
        def __init__(self, **kwargs):
            pass
        def results(self, search):
            searches.append(search)
            for paper_id in reversed(search.id_list):
                yield SimpleNamespace(
                    entry_id=f"https://arxiv.org/abs/{paper_id}v2", title=papers[paper_id].title,
                    summary="Full API abstract", authors=[], categories=["cs.GR"],
                    primary_category="cs.GR", published=datetime(2026, 11, 5, tzinfo=timezone.utc),
                    updated=datetime(2026, 11, 6, tzinfo=timezone.utc),
                    pdf_url=f"https://arxiv.org/pdf/{paper_id}v2", comment=None, journal_ref=None)
    monkeypatch.setattr(fetcher.arxiv, "Client", Client)
    result = fetcher.hybrid_fetch(config())
    assert len(result) == 205
    assert [len(s.id_list) for s in searches] == [100, 100, 5]
    assert all(s.max_results == len(s.id_list) for s in searches)
    assert result[0].arxiv_id == "2611.00000" and result[0].announce_type == "cross"
    assert all(p.version == 2 and p.published.day == 5 for p in result)
    assert all(p.matched_direction == "scene_reconstruction" for p in result)


def test_missing_api_metadata_fails_instead_of_reporting_a_partial_batch(monkeypatch):
    paper = announcement(1)
    monkeypatch.setattr(fetcher, "fetch_rss", lambda cfg: ({paper.arxiv_id: paper}, {}))
    monkeypatch.setattr(fetcher.arxiv, "Client", lambda **kw: SimpleNamespace(results=lambda s: iter(())))
    with pytest.raises(ValueError, match="omitted"):
        fetcher.hybrid_fetch(config())


def test_rss_transport_and_invalid_feed_fail_closed(monkeypatch):
    for status, body in [(503, b"unavailable"), (200, b"<html>maintenance</html>")]:
        monkeypatch.setattr(fetcher.httpx, "get", lambda *a, **kw: httpx.Response(
            status, content=body, request=httpx.Request("GET", "https://rss.arxiv.org/rss/cs.GR")))
        with pytest.raises((httpx.HTTPStatusError, ValueError)):
            fetcher.fetch_rss(config())


def test_empty_rss_does_not_request_api_and_keyword_matching_is_bounded(monkeypatch):
    monkeypatch.setattr(fetcher, "fetch_rss", lambda cfg: ({}, {}))
    monkeypatch.setattr(fetcher.arxiv.Client, "results", lambda *a: pytest.fail("No API request expected"))
    assert fetcher.hybrid_fetch(config()) == []
    direction = config()["research_directions"]["scene_reconstruction"]
    assert fetcher._matches_direction(announcement(1, title="Gaussian-splatting"), direction)
    assert fetcher._matches_direction(announcement(1, title="Assets", abstract="Scene for simulation"), direction)
    assert not fetcher._matches_direction(announcement(1, title="Assets", abstract="Simulation alone"), direction)
