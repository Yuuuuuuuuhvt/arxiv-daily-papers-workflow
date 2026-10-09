from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import TEMPLATES_DIR, DATA_DIR
from .models import AnalysisResult, ArxivPaper

logger = logging.getLogger(__name__)

DIRECTION_DISPLAY = {
    "embodied_ai": "Embodied AI",
    "world_models": "World Models",
    "autonomous_driving": "Autonomous Driving",
    "multiple": "Multiple",
}


def _paper_view(
    paper: ArxivPaper,
    analysis: AnalysisResult,
    hjfy_template: str,
    date_str: str = "",
    deep_research: str = "",
) -> dict:
    github_repo = os.environ.get("GITHUB_REPOSITORY", "Yuuuuuuuuhvt/arxiv-daily-papers-workflow")
    release_pdf_url = f"https://github.com/{github_repo}/releases/download/v{date_str}/{paper.arxiv_id}.pdf"
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors_str": ", ".join(paper.authors[:5]) + (" et al." if len(paper.authors) > 5 else ""),
        "abs_url": paper.abs_url,
        "pdf_url": paper.pdf_url,
        "release_pdf_url": release_pdf_url,
        "hjfy_url": hjfy_template.format(arxiv_id=paper.arxiv_id),
        "code_url": analysis.code_url,
        "one_line_summary": analysis.one_line_summary,
        "detailed_summary": analysis.detailed_summary,
        "direction_display": DIRECTION_DISPLAY.get(analysis.direction, analysis.direction),
        "tags_display": " ".join(analysis.tags),
        "weighted_score": analysis.weighted_score,
        "novelty": analysis.novelty_score,
        "impact": analysis.impact_score,
        "reproducibility": analysis.reproducibility_score,
        "affiliation_names_str": ", ".join(analysis.affiliation_names) if analysis.affiliation_names else "Unknown",
        "affiliation_tier": analysis.affiliation_tier,
        "key_terms_str": ", ".join(analysis.key_terms),
        "deep_research": deep_research,
    }


def generate_daily_report(
    date_str: str,
    core_papers: list[ArxivPaper],
    peripheral_papers: list[ArxivPaper],
    analyses: dict[str, AnalysisResult],
    deep_research_reports: dict[str, str],
    config: dict,
) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("daily_report.md.j2")

    hjfy_template = config.get("hjfy", {}).get("link_template", "https://hjfy.top/arxiv/{arxiv_id}")
    hot_threshold = config.get("scoring", {}).get("hot_threshold", 4.0)

    core_views = []
    for paper in core_papers:
        a = analyses.get(paper.arxiv_id, AnalysisResult())
        dr = deep_research_reports.get(paper.arxiv_id, "")
        core_views.append(_paper_view(paper, a, hjfy_template, dr))
    core_views.sort(key=lambda v: v["weighted_score"], reverse=True)

    peripheral_views = []
    for paper in peripheral_papers:
        a = analyses.get(paper.arxiv_id, AnalysisResult())
        peripheral_views.append(_paper_view(paper, a, hjfy_template))
    peripheral_views.sort(key=lambda v: v["weighted_score"], reverse=True)

    highlights = [v for v in core_views if v["weighted_score"] >= hot_threshold]

    direction_counts = {}
    for v in core_views + peripheral_views:
        d = v["direction_display"]
        direction_counts[d] = direction_counts.get(d, 0) + 1

    ctx = {
        "date": date_str,
        "core_count": len(core_views),
        "peripheral_count": len(peripheral_views),
        "paper_count": len(core_views) + len(peripheral_views),
        "highlights": highlights,
        "core_papers": core_views,
        "peripheral_papers": peripheral_views,
        "direction_counts": direction_counts,
        "top_score": core_views[0]["weighted_score"] if core_views else 0,
        "top_paper_title": core_views[0]["title"] if core_views else "N/A",
    }

    return template.render(**ctx)


def save_report(content: str, date_str: str) -> Path:
    reports_dir = DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{date_str}.md"
    path.write_text(content, encoding="utf-8")
    logger.info("Report saved: %s", path)
    return path


def generate_email_html(
    date_str: str,
    papers: list[ArxivPaper],
    analyses: dict[str, AnalysisResult],
    config: dict,
) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("email_digest.html.j2")

    hjfy_template = config.get("hjfy", {}).get("link_template", "https://hjfy.top/arxiv/{arxiv_id}")
    hot_threshold = config.get("scoring", {}).get("hot_threshold", 4.0)
    max_highlights = config.get("email", {}).get("max_highlights_in_email", 3)

    all_views = []
    for paper in papers:
        a = analyses.get(paper.arxiv_id, AnalysisResult())
        all_views.append(_paper_view(paper, a, hjfy_template))

    all_views.sort(key=lambda v: v["weighted_score"], reverse=True)
    highlights = all_views[:max_highlights]

    direction_counts = {}
    for v in all_views:
        d = v["direction_display"]
        direction_counts[d] = direction_counts.get(d, 0) + 1

    return template.render(
        date=date_str,
        paper_count=len(all_views),
        highlights=highlights,
        direction_counts=direction_counts,
    )
