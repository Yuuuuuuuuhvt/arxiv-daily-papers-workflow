from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .models import ArxivPaper

logger = logging.getLogger(__name__)


def load_index(path: Path) -> dict:
    if not path.exists():
        return {"_meta": {"version": 1, "last_updated": "", "total_papers": 0}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = Path(str(path) + ".bak")
        shutil.copy2(path, backup)

    index["_meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    index["_meta"]["total_papers"] = len(index) - 1  # exclude _meta

    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    logger.info("Index saved: %d papers", index["_meta"]["total_papers"])


def dedup(candidates: list[ArxivPaper], index: dict) -> list[ArxivPaper]:
    new_papers = []
    for paper in candidates:
        if paper.arxiv_id not in index:
            new_papers.append(paper)
        else:
            logger.debug("Skipping known paper: %s (%s)", paper.arxiv_id, paper.title[:60])

    logger.info("Dedup: %d candidates -> %d new papers", len(candidates), len(new_papers))
    return new_papers
