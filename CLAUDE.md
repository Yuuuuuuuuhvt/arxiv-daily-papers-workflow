# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install runtime dependencies.
python -m pip install -r requirements.txt

# Install the test runner. It is not listed in requirements.txt.
python -m pip install pytest

# Run the full pipeline. Requires OPENROUTER_API_KEY.
python -m src.main
python -m src

# Skip the expensive DeepResearch stage.
SKIP_DEEP_RESEARCH=1 python -m src.main

# Override report date. The default is today's UTC date.
REPORT_DATE=2026-03-28 python -m src.main

# Prevent the in-process CI git push path if GITHUB_ACTIONS is set.
SKIP_GIT_PUSH=1 python -m src.main

# Run all tests.
python -m pytest tests/ -v

# Run one test file or one test case.
python -m pytest tests/test_fetcher.py -v
python -m pytest tests/test_relevance_filter.py::test_parse_failure_defaults_to_peripheral -v
```

On PowerShell, set runtime flags with `$env:NAME = 'value'`, then run `python -m src.main`. There is no configured build or lint command in this repository.

## Architecture

This repository is a fully automated arXiv paper discovery pipeline for Embodied AI, World Models, and Autonomous Driving. The scheduled GitHub Actions workflow runs on weekdays at UTC 05:30.

The pipeline is sequentially orchestrated by `src/main.py::run_pipeline()`:

```text
Fetch API and RSS -> Dedup -> Relevance Filter -> Deep Analysis -> PDF Download
  -> DeepResearch for core papers -> Report -> Email -> Data commit and push
```

### Pipeline stages

- Fetching lives in `src/fetcher.py`. It combines arXiv API keyword search with arXiv RSS metadata. RSS announce types are used to prefer new and cross submissions, ignore replacement-only entries, and keep recent API-only papers as a safety net.
- Deduplication lives in `src/dedup.py`. `data/papers_index.json` is the cumulative source of truth, and `save_index()` writes a `.bak` backup before replacing the index.
- Relevance filtering lives in `src/relevance_filter.py`. The LLM classifies papers as `core`, `peripheral`, or `not_relevant`. Parse failures and missing results default to `peripheral` so potentially relevant papers are not dropped silently.
- Deep analysis lives in `src/deep_analysis.py`. It runs for both core and peripheral papers, reads affiliation tiers from `config/affiliations.json`, adds focus relevance from `config/config.yaml`, computes weighted scores, and assigns report tags.
- DeepResearch lives in `src/deep_research.py`. It runs only for core papers unless `SKIP_DEEP_RESEARCH=1` is set. It prefers local PDF input, switches from native PDF parsing to `pdf-text` above 15 MB, and falls back to text-only analysis if PDF processing fails.
- Report and email rendering live in `src/report_generator.py` with Jinja2 templates under `templates/`. Markdown rendering uses `autoescape=False`; HTML email rendering uses `autoescape=True`. Keep report shaping logic in `_paper_view()` rather than in templates.
- Git push support exists in `src/git_ops.py`, but the current workflow sets `SKIP_GIT_PUSH=1` during pipeline execution and performs the data submodule commit and parent submodule-pointer commit in `.github/workflows/daily-run.yml`.

### Configuration and prompts

- Runtime configuration lives in `config/config.yaml`: arXiv categories, research directions, model IDs, scoring weights, email settings, concurrency, PDF settings, and hjfy link settings.
- Prompt text lives in `prompts/` and is loaded with `config.load_prompt()`. Change LLM behavior through prompt files and model settings before changing Python code.
- The OpenRouter client is centralized in `src/llm_client.py`. It retries 429 and 5xx or transport failures three times with 2, 4, and 8 second backoff.
- Data models are dataclasses in `src/models.py`. Keep dependencies minimal and avoid adding Pydantic unless the project direction changes.

### Data layout

`data/` is a git submodule backed by a separate data repository. It contains `papers_index.json`, generated reports under `data/reports/YYYY-MM-DD.md`, and downloaded PDFs under `data/pdfs/<date>/`. Local runs skip remote git pushes unless CI-related environment variables are set.

## Coding Conventions

- Put `from __future__ import annotations` at the top of Python files.
- Prefer `X | None` over `Optional[X]` in new or edited code.
- Use async I/O for network and LLM work. Pipeline concurrency is controlled with `asyncio.Semaphore` from `config.config.yaml`.
- Use `logging.getLogger(__name__)`; do not use `print()`. The root logger emits JSON-shaped records with `ts`, `level`, `stage`, and `msg`.
- Keep tunable values in `config/config.yaml`. Do not hardcode model IDs, thresholds, keywords, batch sizes, or concurrency limits in stage code.
- Use `pathlib.Path` for filesystem paths to keep Windows and Linux behavior consistent.

## Extension Patterns

- Add a research direction by adding a block under `research_directions` in `config/config.yaml`, updating `prompts/relevance_filter.txt`, and adding a display label to `DIRECTION_DISPLAY` in `src/report_generator.py`.
- Add a pipeline stage by creating a focused `src/<stage>.py` module, wiring it into `src/main.py::run_pipeline()`, adding a semaphore for network or LLM fanout when needed, and moving new knobs into `config/config.yaml`.
- Change output structure by updating `src/report_generator.py::_paper_view()` and the relevant template together. Avoid putting business logic directly into Jinja2 templates.
- Change model behavior by editing prompt files and `models.*` entries in `config/config.yaml`. If response structure changes, update the relevant parser and preserve malformed JSON fallbacks.

## Known Constraints

- `papers_index.json` controls deduplication. If it is deleted or corrupted, previously seen papers can be reprocessed.
- `analyze_all()` and `generate_all_deep_research()` are legacy serial wrappers for standalone use. The main pipeline uses concurrent calls inside `run_pipeline()`.
- `download_all_pdfs()` has a `max_concurrent` argument, but `run_pipeline()` currently calls it with the function default of 5.
- arXiv API calls use `api_delay_seconds` from config between direction searches to reduce rate limiting.
- Email requires `QQ_MAIL_ADDRESS` and `QQ_MAIL_AUTH_CODE`. Missing email variables skip email without failing report generation.
