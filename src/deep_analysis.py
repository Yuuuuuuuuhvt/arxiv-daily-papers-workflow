from __future__ import annotations

import json
import logging

from .config import load_affiliations, load_config, load_prompt
from .budget import BudgetExceeded
from .llm_client import call_llm
from .models import AnalysisResult, ArxivPaper

logger = logging.getLogger(__name__)

AFFILIATION_TIER_SCORES = {1: 5, 2: 4, 3: 3, 0: 2}
DIRECTION_MATCH_SCORES = {"multiple": 5, "exact": 4, "tangential": 3}


def _compute_weighted_score(analysis: AnalysisResult, config: dict) -> float:
    weights = config["scoring"]["weights"]
    aff_score = AFFILIATION_TIER_SCORES.get(analysis.affiliation_tier, 2)
    dir_score = DIRECTION_MATCH_SCORES.get(
        "multiple" if analysis.direction == "multiple" else "exact", 4
    )
    score = (
        analysis.novelty_score * weights["novelty"]
        + analysis.impact_score * weights["impact"]
        + analysis.reproducibility_score * weights["reproducibility"]
        + aff_score * weights["affiliation"]
        + dir_score * weights["direction_match"]
    )
    if "focus_relevance" in weights:
        score += analysis.focus_relevance_score * weights["focus_relevance"]
    return score


def _assign_tags(analysis: AnalysisResult, hot_threshold: float) -> list[str]:
    tags = []
    if analysis.has_code:
        tags.append("\U0001f7e2 Code")
    else:
        tags.append("\U0001f534 No Code")
    if analysis.has_dataset:
        tags.append("\U0001f4ca Dataset")
    if analysis.has_demo:
        tags.append("\U0001f3ae Demo")
    if analysis.weighted_score >= hot_threshold:
        tags.append("\U0001f525 Hot")
    return tags


def _parse_analysis(raw: str) -> dict:
    text = raw.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)


def _dict_to_analysis(data: dict) -> AnalysisResult:
    return AnalysisResult(
        one_line_summary=data.get("one_line_summary", ""),
        detailed_summary=data.get("detailed_summary", ""),
        direction=data.get("direction", ""),
        tags=data.get("tags", []),
        affiliation_tier=int(data.get("affiliation_tier", 0)),
        affiliation_names=data.get("affiliation_names", []),
        novelty_score=int(data.get("novelty_score", 3)),
        impact_score=int(data.get("impact_score", 3)),
        reproducibility_score=int(data.get("reproducibility_score", 3)),
        focus_relevance_score=int(data.get("focus_relevance_score", 3)),
        has_code=bool(data.get("has_code", False)),
        code_url=data.get("code_url"),
        has_dataset=bool(data.get("has_dataset", False)),
        has_demo=bool(data.get("has_demo", False)),
        key_terms=data.get("key_terms", []),
        related_work_context=data.get("related_work_context", ""),
    )


async def analyze_paper(
    paper: ArxivPaper,
    model_config: dict,
    scoring_config: dict,
    api_key: str,
) -> AnalysisResult:
    system_prompt = load_prompt("deep_analysis.txt")
    affiliations = load_affiliations()
    focus_config = load_config().get("research_focus", {})
    focus_desc = focus_config.get("description", "").strip()

    user_msg = (
        f"## Paper to Analyze\n\n"
        f"**Title**: {paper.title}\n"
        f"**Authors**: {', '.join(paper.authors)}\n"
        f"**Categories**: {', '.join(paper.categories)}\n"
        f"**Abstract**: {paper.abstract}\n"
        f"**Comment**: {paper.comment or 'None'}\n"
        f"**Journal Ref**: {paper.journal_ref or 'None'}\n"
        f"**PDF URL**: {paper.pdf_url}\n\n"
        f"## Affiliation Reference\n"
        f"{json.dumps(affiliations, ensure_ascii=False, indent=2)}\n\n"
    )
    if focus_desc:
        user_msg += (
            f"## Reader's Research Focus\n"
            f"{focus_desc}\n\n"
            f"Score `focus_relevance_score` based on how closely this paper "
            f"aligns with the reader's focus described above.\n\n"
        )
    user_msg += "Produce the analysis JSON."

    try:
        raw = await call_llm(
            model=model_config["model_id"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=model_config.get("temperature", 0.2),
            api_key=api_key,
        )
        data = _parse_analysis(raw)
        analysis = _dict_to_analysis(data)
    except BudgetExceeded:
        raise
    except Exception:
        logger.exception("Deep analysis failed for %s, using defaults", paper.arxiv_id)
        analysis = AnalysisResult(
            one_line_summary=paper.title,
            direction=paper.matched_direction or "",
        )

    analysis.weighted_score = round(
        _compute_weighted_score(analysis, {"scoring": scoring_config}), 2
    )

    hot_threshold = scoring_config.get("hot_threshold", 4.0)
    extra_tags = _assign_tags(analysis, hot_threshold)
    for t in extra_tags:
        if t not in analysis.tags:
            analysis.tags.append(t)

    logger.info(
        "Analyzed %s: score=%.2f, direction=%s",
        paper.arxiv_id, analysis.weighted_score, analysis.direction,
    )
    return analysis


async def analyze_all(
    papers: list[ArxivPaper],
    model_config: dict,
    scoring_config: dict,
    api_key: str,
) -> dict[str, AnalysisResult]:
    results: dict[str, AnalysisResult] = {}
    for paper in papers:
        results[paper.arxiv_id] = await analyze_paper(
            paper, model_config, scoring_config, api_key
        )
    return results
