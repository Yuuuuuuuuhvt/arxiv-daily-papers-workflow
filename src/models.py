from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ArxivPaper:
    arxiv_id: str
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    pdf_url: str
    abs_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    matched_direction: Optional[str] = None
    announce_type: Optional[str] = None
    rss_pub_date: Optional[str] = None


@dataclass
class RelevanceResult:
    arxiv_id: str
    verdict: str  # "core" | "peripheral" | "not_relevant"
    direction: str  # "embodied_ai" | "world_models" | "autonomous_driving" | "multiple" | "none"
    confidence: float
    reason: str = ""
    error: Optional[str] = None


@dataclass
class AnalysisResult:
    one_line_summary: str = ""
    detailed_summary: str = ""
    direction: str = ""
    tags: list[str] = field(default_factory=list)
    affiliation_tier: int = 0
    affiliation_names: list[str] = field(default_factory=list)
    novelty_score: int = 3
    impact_score: int = 3
    reproducibility_score: int = 3
    focus_relevance_score: int = 3
    has_code: bool = False
    code_url: Optional[str] = None
    has_dataset: bool = False
    has_demo: bool = False
    key_terms: list[str] = field(default_factory=list)
    related_work_context: str = ""
    weighted_score: float = 0.0


def paper_to_index_entry(
    paper: ArxivPaper,
    analysis: AnalysisResult,
    release_pdf_url: str = "",
) -> dict:
    return {
        "title": paper.title,
        "authors": paper.authors,
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "published": paper.published.isoformat() if isinstance(paper.published, datetime) else str(paper.published),
        "updated": paper.updated.isoformat() if isinstance(paper.updated, datetime) else str(paper.updated),
        "abs_url": paper.abs_url,
        "pdf_url": paper.pdf_url,
        "release_pdf_url": release_pdf_url,
        "first_seen": datetime.utcnow().strftime("%Y-%m-%d"),
        "announce_type": paper.announce_type,
        "direction": analysis.direction,
        "one_line_summary": analysis.one_line_summary,
        "tags": analysis.tags,
        "affiliation_tier": analysis.affiliation_tier,
        "affiliation_names": analysis.affiliation_names,
        "scores": {
            "novelty": analysis.novelty_score,
            "impact": analysis.impact_score,
            "reproducibility": analysis.reproducibility_score,
            "focus_relevance": analysis.focus_relevance_score,
            "weighted": round(analysis.weighted_score, 2),
        },
        "has_code": analysis.has_code,
        "code_url": analysis.code_url,
        "has_dataset": analysis.has_dataset,
        "has_demo": analysis.has_demo,
        "key_terms": analysis.key_terms,
        "journal_ref": paper.journal_ref,
        "comment": paper.comment,
    }
