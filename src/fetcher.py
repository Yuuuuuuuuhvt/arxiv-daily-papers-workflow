from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

import arxiv
import feedparser

from .models import ArxivPaper

logger = logging.getLogger(__name__)


def _strip_version(arxiv_id: str) -> tuple[str, int]:
    m = re.match(r"^(\d+\.\d+)(v(\d+))?$", arxiv_id)
    if m:
        return m.group(1), int(m.group(3)) if m.group(3) else 1
    return arxiv_id, 1


def _extract_id_from_url(url: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
    return m.group(1) if m else ""


def build_query(direction_config: dict) -> str:
    clauses = []

    if direction_config.get("title_keywords"):
        ti_parts = [f'ti:"{kw}"' for kw in direction_config["title_keywords"]]
        clauses.append(f'({" OR ".join(ti_parts)})')

    if direction_config.get("abstract_keywords"):
        abs_parts = [f'abs:"{kw}"' for kw in direction_config["abstract_keywords"]]
        clauses.append(f'({" OR ".join(abs_parts)})')

    for combo in direction_config.get("abstract_combos", []):
        combo_parts = [f'abs:"{term}"' for term in combo]
        clauses.append(f'({" AND ".join(combo_parts)})')

    keyword_query = " OR ".join(clauses)

    cats = direction_config.get("categories", [])
    if cats:
        cat_query = " OR ".join([f"cat:{c}" for c in cats])
        return f"({keyword_query}) AND ({cat_query})"

    return keyword_query


def _result_to_paper(result: arxiv.Result, direction_id: str) -> ArxivPaper:
    raw_id = result.entry_id.split("/")[-1]
    base_id, version = _strip_version(raw_id)
    return ArxivPaper(
        arxiv_id=base_id,
        version=version,
        title=result.title.replace("\n", " ").strip(),
        abstract=result.summary.replace("\n", " ").strip(),
        authors=[a.name for a in result.authors],
        categories=[c for c in result.categories],
        primary_category=result.primary_category,
        published=result.published,
        updated=result.updated,
        pdf_url=result.pdf_url,
        abs_url=result.entry_id,
        comment=result.comment,
        journal_ref=result.journal_ref,
        matched_direction=direction_id,
    )


def fetch_api(config: dict, report_date: str | None = None) -> dict[str, ArxivPaper]:
    papers: dict[str, ArxivPaper] = {}
    day = datetime.strptime(report_date, "%Y-%m-%d") if report_date else None
    max_results = config["arxiv"].get("api_max_results_per_direction", 50)
    delay = config["arxiv"].get("api_delay_seconds", 3)
    num_retries = config["arxiv"].get("api_num_retries", 5)

    for direction_id, direction_config in config["research_directions"].items():
        query = build_query(direction_config)
        if day:
            stamp = day.strftime("%Y%m%d")
            query = f"({query}) AND submittedDate:[{stamp}0000 TO {stamp}2359]"
        logger.info("Fetching API for %s: %s", direction_id, query[:120])

        client = arxiv.Client(
            delay_seconds=delay,
            num_retries=num_retries,
        )
        search = arxiv.Search(
            query=query,
            max_results=None if day else max_results,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        count = 0
        try:
            for result in client.results(search):
                paper = _result_to_paper(result, direction_id)
                if paper.arxiv_id not in papers:
                    papers[paper.arxiv_id] = paper
                count += 1
            logger.info("API returned %d papers for %s", count, direction_id)
        except arxiv.HTTPError as e:
            if day:
                raise
            logger.warning("arXiv API rate-limited for %s after %d retries: %s",
                           direction_id, num_retries, e)
        except Exception as e:
            if day:
                raise
            logger.warning("arXiv API failed for %s: %s", direction_id, e)

        time.sleep(delay)

    return papers


def _parse_rss_entry(entry: dict) -> tuple[str, ArxivPaper | None, dict]:
    """Parse a single RSS entry into (arxiv_id, ArxivPaper | None, metadata)."""
    arxiv_id = _extract_id_from_url(entry.get("link", ""))
    if not arxiv_id:
        return "", None, {}

    announce_type = "new"
    primary_category = ""
    for tag in entry.get("tags", []):
        term = tag.get("term", "")
        if term in ("new", "replace", "cross", "replace-cross"):
            announce_type = term
        elif term:
            primary_category = primary_category or term

    if hasattr(entry, "arxiv_announce_type"):
        announce_type = entry.arxiv_announce_type

    metadata = {
        "announce_type": announce_type,
        "pub_date": entry.get("published", ""),
    }

    # Extract abstract from summary (format: "arXiv:XXXX.XXXXvN Announce Type: ...\nAbstract: ...")
    summary = entry.get("summary", "")
    abstract = summary
    abs_marker = "Abstract: "
    idx = summary.find(abs_marker)
    if idx != -1:
        abstract = summary[idx + len(abs_marker):].strip()

    # Parse authors (RSS puts all authors in one name field, comma-separated)
    authors = []
    for a in entry.get("authors", []):
        authors.extend([name.strip() for name in a.get("name", "").split(",") if name.strip()])

    # Parse published date
    pub_parsed = entry.get("published_parsed")
    if pub_parsed:
        published = datetime(*pub_parsed[:6], tzinfo=timezone.utc)
    else:
        published = datetime.now(timezone.utc)

    categories = [tag.get("term", "") for tag in entry.get("tags", [])
                  if tag.get("term", "") not in ("new", "replace", "cross", "replace-cross")]

    paper = ArxivPaper(
        arxiv_id=arxiv_id,
        version=1,
        title=entry.get("title", "").replace("\n", " ").strip(),
        abstract=abstract.replace("\n", " ").strip(),
        authors=authors,
        categories=categories or [primary_category] if primary_category else [],
        primary_category=primary_category,
        published=published,
        updated=published,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
        announce_type=announce_type,
        rss_pub_date=entry.get("published", ""),
    )

    return arxiv_id, paper, metadata


def fetch_rss(config: dict) -> tuple[dict[str, ArxivPaper], dict[str, dict]]:
    """Fetch RSS and return both full papers and metadata dicts."""
    categories = config["arxiv"].get("rss_categories", "cs.CV+cs.RO+cs.AI+cs.LG+cs.MA")
    url = f"https://rss.arxiv.org/rss/{categories}"
    logger.info("Fetching RSS: %s", url)

    feed = feedparser.parse(url)
    papers: dict[str, ArxivPaper] = {}
    metadata: dict[str, dict] = {}

    for entry in feed.entries:
        arxiv_id, paper, meta = _parse_rss_entry(entry)
        if not arxiv_id:
            continue
        metadata[arxiv_id] = meta
        if paper:
            papers[arxiv_id] = paper

    logger.info("RSS returned %d entries", len(metadata))
    return papers, metadata


def hybrid_fetch(config: dict, report_date: str | None = None) -> list[ArxivPaper]:
    api_papers = fetch_api(config, report_date)
    if report_date:
        # 补收按 UTC 首次提交日期查询，不能把当前 RSS 当作历史公告。
        return [paper for paper in api_papers.values()
                if paper.published.astimezone(timezone.utc).strftime("%Y-%m-%d") == report_date]
    _rss_papers, rss_metadata = fetch_rss(config)

    for arxiv_id, paper in api_papers.items():
        if arxiv_id in rss_metadata:
            paper.announce_type = rss_metadata[arxiv_id]["announce_type"]
            paper.rss_pub_date = rss_metadata[arxiv_id]["pub_date"]
        else:
            paper.announce_type = None
            paper.rss_pub_date = None

    candidates = []
    now = datetime.now(timezone.utc)
    staleness_hours = config["arxiv"].get("staleness_threshold_hours", 24)

    for paper in api_papers.values():
        if paper.announce_type in ("replace", "replace-cross"):
            continue

        if paper.announce_type in ("new", "cross"):
            candidates.append(paper)
            continue

        # Not in RSS — apply safety net
        updated = paper.updated
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age = now - updated
        if age <= timedelta(hours=staleness_hours):
            candidates.append(paper)
            logger.info("Keeping API-only paper (age %s): %s", age, paper.arxiv_id)
        else:
            logger.debug("Discarding stale API-only paper (age %s): %s", age, paper.arxiv_id)

    logger.info("Hybrid fetch: %d API papers + %d RSS entries -> %d candidates",
                len(api_papers), len(rss_metadata), len(candidates))
    return candidates
