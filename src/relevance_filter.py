from __future__ import annotations

import asyncio
import json
import logging

from .config import load_prompt
from .budget import BudgetExceeded
from .llm_client import call_llm
from .models import ArxivPaper, RelevanceResult

logger = logging.getLogger(__name__)


def _build_papers_block(papers: list[ArxivPaper]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"[Paper {i}]")
        lines.append(f"ID: {p.arxiv_id}")
        lines.append(f"Title: {p.title}")
        lines.append(f"Categories: {' '.join(p.categories)}")
        lines.append(f"Abstract: {p.abstract}")
        lines.append("")
    return "\n".join(lines)


FEW_SHOT_BLOCK = """## Illustrative synthetic calibration examples

CORE:
- A VLA architecture that improves language-conditioned robot manipulation
  -> core, embodied_ai
- A latent world model benchmark evaluating action-conditioned future prediction
  -> core, world_models
- An end-to-end autonomous driving model evaluated on closed-loop driving tasks
  -> core, autonomous_driving
- A software agent memory system evaluated on long-horizon tool-use tasks
  -> core, agentic; no physical robot is required
- Coordinating coding agents through shared plans, with repository task evaluation
  -> core, agentic; software-only evaluation is sufficient
- Model predictive control for dexterous manipulation evaluated on real robot hands
  -> core, robotics; no foundation model is required
- A SLAM method evaluated for mobile robot navigation
  -> core, robotics; classical estimation is eligible
- An open robot hand with tactile sensing and evaluated manipulation capabilities
  -> core, robotics; hardware can be the main contribution

PERIPHERAL:
- A generic storage optimization with only a brief discussion of agent memory
  -> peripheral, agentic; agent behavior is not evaluated
- A materials study motivated by robot grippers without a robot method or evaluation
  -> peripheral, robotics; it provides supporting context

NOT RELEVANT:
- A contrast agent for medical imaging with no intelligent-agent or robot method
  -> not_relevant, none
- Generic stock-price prediction with no action-taking agent contribution
  -> not_relevant, none
- A video generator for visual entertainment without a world-model evaluation
  -> not_relevant, none"""


VALID_VERDICTS = {"core", "peripheral", "not_relevant"}


def _parse_response(raw: str, expected_ids: list[str]) -> list[RelevanceResult]:
    text = raw.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        results = json.loads(text)
        result_map = {r["id"]: r for r in results}
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Failed to parse relevance response, defaulting all to peripheral")
        return [
            RelevanceResult(arxiv_id=pid, verdict="peripheral", direction="unknown",
                            confidence=0.0, error="parse_failure")
            for pid in expected_ids
        ]

    output = []
    for pid in expected_ids:
        if pid in result_map:
            r = result_map[pid]
            verdict = r.get("verdict", "peripheral").lower().strip()
            # Backwards compat: map old "relevant" -> "core"
            if verdict == "relevant":
                verdict = "core"
            if verdict not in VALID_VERDICTS:
                verdict = "peripheral"
            output.append(RelevanceResult(
                arxiv_id=pid,
                verdict=verdict,
                direction=r.get("direction", "unknown"),
                confidence=float(r.get("confidence", 0.5)),
                reason=r.get("reason", ""),
            ))
        else:
            output.append(RelevanceResult(
                arxiv_id=pid, verdict="peripheral", direction="unknown",
                confidence=0.0, error="missing_from_response",
            ))
    return output


async def filter_relevance(
    papers: list[ArxivPaper],
    model_config: dict,
    api_key: str,
) -> tuple[list[ArxivPaper], list[ArxivPaper]]:
    """Returns (core_papers, peripheral_papers)."""
    if not papers:
        return [], []

    system_prompt = load_prompt("relevance_filter.txt")
    batch_size = model_config.get("batch_size", 10)
    model_id = model_config["model_id"]
    temperature = model_config.get("temperature", 0.1)
    max_tokens = model_config.get("max_tokens")

    paper_map = {p.arxiv_id: p for p in papers}
    core_ids: set[str] = set()
    peripheral_ids: set[str] = set()

    batches = []
    for i in range(0, len(papers), batch_size):
        batches.append(papers[i:i + batch_size])

    async def _classify_batch(batch: list[ArxivPaper]) -> list[RelevanceResult]:
        papers_block = _build_papers_block(batch)
        expected_ids = [p.arxiv_id for p in batch]
        user_msg = (
            f"{FEW_SHOT_BLOCK}\n\n---\n\n"
            f"Classify the following {len(batch)} papers. "
            f"Respond with ONLY the JSON array.\n\n{papers_block}"
        )
        try:
            raw = await call_llm(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
            return _parse_response(raw, expected_ids)
        except BudgetExceeded:
            raise
        except Exception:
            logger.exception("Relevance filter LLM call failed, defaulting batch to peripheral")
            return [
                RelevanceResult(arxiv_id=pid, verdict="peripheral", direction="unknown",
                                confidence=0.0, error="llm_failure")
                for pid in expected_ids
            ]

    all_results = await asyncio.gather(*[_classify_batch(b) for b in batches])

    for batch_results in all_results:
        for r in batch_results:
            if r.direction != "unknown" and r.direction != "none":
                paper_map[r.arxiv_id].matched_direction = r.direction
            if r.verdict == "core":
                core_ids.add(r.arxiv_id)
                logger.info("  CORE:       %s | %s | %s", r.arxiv_id, r.direction, r.reason)
            elif r.verdict == "peripheral":
                peripheral_ids.add(r.arxiv_id)
                logger.info("  PERIPHERAL: %s | %s | %s", r.arxiv_id, r.direction, r.reason)
            else:
                logger.info("  EXCLUDED:   %s | %s", r.arxiv_id, r.reason)

    core_papers = [p for p in papers if p.arxiv_id in core_ids]
    peripheral_papers = [p for p in papers if p.arxiv_id in peripheral_ids]
    logger.info("Relevance filter: %d total -> %d core + %d peripheral (%d excluded)",
                len(papers), len(core_papers), len(peripheral_papers),
                len(papers) - len(core_papers) - len(peripheral_papers))
    return core_papers, peripheral_papers
