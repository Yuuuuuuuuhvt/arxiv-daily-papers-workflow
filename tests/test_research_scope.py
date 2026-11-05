from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.models import AnalysisResult, ArxivPaper
from src import relevance_filter, report_generator


def test_new_directions_reach_reports_and_email(monkeypatch):
    examples = json.loads((Path(__file__).parent / "data/research_scope_examples.json").read_text())["papers"]
    day = datetime(2026, 11, 5, tzinfo=timezone.utc)
    papers = [ArxivPaper(
        arxiv_id=e["id"], version=1, title=e["title"], abstract=e["abstract"], authors=["Synthetic example"],
        categories=e["categories"], primary_category=e["categories"][0], published=day, updated=day,
        abs_url="https://example.test/abstract", pdf_url="https://example.test/paper.pdf",
    ) for e in examples]
    async def classify(**kwargs):
        return json.dumps([{
            "id": e["id"], "verdict": e["expected_verdict"], "direction": e["expected_direction"],
            "confidence": 0.9, "reason": "Synthetic transport test",
        } for e in examples])
    monkeypatch.setattr(relevance_filter, "call_llm", classify)
    config = load_config()
    core, peripheral = asyncio.run(relevance_filter.filter_relevance(papers, config["models"]["relevance_filter"], "test-key"))
    assert len(core) == 5 and peripheral == []
    assert core[0].matched_direction == "agentic"
    assert core[1].matched_direction == "robotics"
    analyses = {p.arxiv_id: AnalysisResult(direction=p.matched_direction, weighted_score=4.0) for p in core}
    report = report_generator.generate_daily_report("2026-11-05", core, peripheral, analyses, {}, config)
    email = report_generator.generate_email_html("2026-11-05", core, analyses, config)
    for label in ("Agentic AI", "Robotics", "Embodied AI / VLA", "World Models", "Autonomous Driving"):
        assert label in report and label in email
    assert examples[-1]["title"] not in report
