# AGENTS.md -- arXiv Daily Papers Agent

## Project Overview

全自动 arXiv 论文日报系统。每个工作日抓取 Embodied AI / World Models / Autonomous Driving 三个方向的新论文，经 LLM 过滤、分析、解读后生成 Markdown 日报并可选邮件推送。

- **技术栈**: Python 3.11+, asyncio, httpx, Jinja2, PyYAML, feedparser, arxiv
- **LLM 网关**: OpenRouter (openrouter.ai) — 所有 LLM 调用经此路由
- **部署**: GitHub Actions (定时 cron) + 本地调试

---

## Architecture

```text
Pipeline stages (sequential):
  Fetch -> Dedup -> Relevance Filter -> Deep Analysis -> PDF Download
    -> DeepResearch -> Report -> Email -> Git Push

Data flow:
  arXiv API/RSS -> list[ArxivPaper] -> dedup against papers_index.json
    -> LLM batch classify (core / peripheral / not_relevant)
    -> LLM per-paper analysis (scores, tags, summary)
    -> Download PDFs -> LLM per-paper scholarly analysis (with PDF)
    -> Jinja2 render report -> save to data/reports/YYYY-MM-DD.md
```

### Key Design Decisions

1. **Hybrid fetch**: API (keyword search) + RSS (announce type metadata) combined. RSS provides `new`/`replace`/`cross` classification; API provides deep keyword matching.
2. **3-tier relevance**: `core` (foundation model papers) / `peripheral` (domain-relevant but traditional) / `not_relevant`. Only core papers get DeepResearch.
3. **PDF via base64**: Local PDFs are base64-encoded and sent directly to OpenRouter, avoiding server-side URL fetch failures (OpenRouter 502 when downloading arXiv URLs). Files <=15 MB use `native` engine (Gemini sees figures/tables); larger files degrade to `pdf-text`.
4. **Concurrency**: `asyncio.Semaphore` controls parallel LLM calls (`max_concurrent_llm` for analysis, `max_concurrent_pdf` for DeepResearch).
5. **Fail-safe**: Every LLM stage has fallback — parse failures default to `peripheral`, analysis failures use defaults, DeepResearch falls back from PDF to text-only.

---

## Directory Structure

```text
src/                        # Python package, run via: python -m src.main
  main.py                   # Pipeline orchestrator — the ONLY entry point
  config.py                 # Path constants (ROOT_DIR, DATA_DIR, etc.) + YAML/env loading
  models.py                 # Dataclasses: ArxivPaper, RelevanceResult, AnalysisResult
  fetcher.py                # arXiv API + RSS hybrid fetching
  dedup.py                  # ID-based deduplication against papers_index.json
  llm_client.py             # OpenRouter API wrapper (retry, backoff, error handling)
  relevance_filter.py       # Batch LLM classification with few-shot examples
  deep_analysis.py          # Per-paper structured analysis (scores, tags, affiliations)
  deep_research.py          # Per-paper 3-module scholarly analysis (with PDF support)
  pdf_downloader.py         # Async PDF download from arXiv
  report_generator.py       # Jinja2 report + email HTML rendering
  email_sender.py           # 通用 SMTP STARTTLS 发信，返回实际提交结果
  git_ops.py                # Data submodule commit + push (CI only)
config/
  config.yaml               # ALL configuration: keywords, models, scoring, concurrency
  affiliations.json         # Institution tier whitelist for scoring
prompts/
  relevance_filter.txt      # System prompt for classification
  deep_analysis.txt         # System prompt for structured analysis
  deep_research.txt         # System prompt for scholarly deep-dive (Chinese output)
templates/
  daily_report.md.j2        # Markdown report template
  email_digest.html.j2      # Email HTML template
tests/                      # pytest unit tests
data/                       # Runtime data (gitignored, or git submodule)
  papers_index.json         # Cumulative paper index (dedup source)
  reports/                  # Generated daily reports
  pdfs/                     # Downloaded PDFs (local cache)
.github/workflows/
  daily-run.yml             # GitHub Actions: cron + manual trigger
```

---

## Coding Conventions

### Python Style

- `from __future__ import annotations` at top of every file (PEP 604 union types).
- Type hints everywhere: `dict[str, X]`, `list[X]`, `X | None` (not `Optional[X]`).
- Dataclasses for all data models (not Pydantic — keep dependencies minimal).
- Async functions for all I/O (LLM calls, PDF downloads). Pipeline orchestration in `main.py` uses `asyncio.gather` + `Semaphore`.
- Logging via `logging.getLogger(__name__)` — all output is JSON-formatted (see `JSONFormatter` in `main.py`).
- No comments unless logic is non-obvious. Docstrings on public functions only.

### Error Handling Philosophy

- **LLM calls**: 3 retries with exponential backoff (2s, 4s, 8s) for retryable errors (429, 5xx, timeout). Non-retryable errors (KeyError, ValueError) break immediately.
- **Parsing LLM output**: Always handle malformed JSON gracefully — default to safe values (`peripheral` for relevance, default scores for analysis, empty string for DeepResearch).
- **PDF processing**: 3-tier fallback: base64+native -> base64+pdf-text -> text-only. Never crash the pipeline over a PDF failure.

### Configuration

- **All tunable parameters live in `config/config.yaml`** — never hardcode model IDs, thresholds, batch sizes, or keyword lists in source code.
- **Prompts are plain text files** in `prompts/` — loaded at runtime via `config.load_prompt()`. Edit prompts to adjust LLM behavior without touching code.
- **Environment variables**: `OPENROUTER_API_KEY` (required), `MAIL_SMTP_HOST`、`MAIL_SMTP_PORT`、`MAIL_USERNAME`、`MAIL_PASSWORD`、`MAIL_FROM`、`MAIL_TO`，六项齐全时通过 STARTTLS 发信，缺项只跳过邮件, `SKIP_DEEP_RESEARCH` (optional flag).

### Templates

- Jinja2 with `autoescape=False` for Markdown, `autoescape=True` for HTML email.
- Template variables are pre-computed in `report_generator.py` via `_paper_view()` — templates should not contain business logic.

---

## How to Run

```powershell
# Local run (requires OPENROUTER_API_KEY)
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
python -m src.main

# Skip DeepResearch stage (faster, cheaper)
$env:SKIP_DEEP_RESEARCH = "1"
python -m src.main

# Run tests
pip install pytest
pytest tests/ -v
```

---

## Key Patterns to Follow

### Adding a New Research Direction

1. Add keyword block in `config/config.yaml` under `research_directions`.
2. Update `prompts/relevance_filter.txt` to describe the new direction for LLM classification.
3. Add display name in `report_generator.py` `DIRECTION_DISPLAY` dict.

### Adding a New Pipeline Stage

1. Create `src/new_stage.py` with async function(s).
2. Wire it into `main.py::run_pipeline()` at the appropriate position.
3. Add concurrency control via `asyncio.Semaphore` if it makes LLM/network calls.
4. Add config keys to `config/config.yaml` if needed.

### Modifying LLM Behavior

- **Change what the LLM outputs**: Edit the corresponding file in `prompts/`.
- **Change which model is used**: Edit `models.*.model_id` in `config/config.yaml`.
- **Change parsing logic**: Edit `_parse_response()` or `_parse_analysis()` in the relevant module. Always handle malformed JSON.

### Modifying Report Format

- Edit `templates/daily_report.md.j2` (Markdown) or `templates/email_digest.html.j2` (email).
- If you need new template variables, add them in `report_generator.py::_paper_view()` or the `ctx` dict in `generate_daily_report()`.

---

## Known Constraints and Gotchas

1. **OpenRouter native PDF engine**: Intermittent 502 when OpenRouter fetches PDF URLs server-side. Solved by sending base64-encoded local PDFs instead. Files >15 MB may still timeout with native engine — auto-degrades to `pdf-text`.

2. **arXiv rate limiting**: The `arxiv` library respects rate limits, but `api_delay_seconds` (default 3s) adds extra safety. PDF downloads from arxiv.org may get 429'd under heavy concurrency — `pdf_downloader.py` uses a semaphore (default 5).

3. **`papers_index.json` is the single source of truth for dedup**: If this file is corrupted or deleted, all papers will be re-processed. The file has a `.bak` backup created on every save.

4. **Git push only runs in CI** (`GITHUB_ACTIONS` env var check). The workflow YAML also has its own git push step — both exist for redundancy but the YAML step is authoritative.

5. **`analyze_all()` and `generate_all_deep_research()`** are legacy serial wrappers — unused by `main.py` which implements its own concurrent versions. They exist for potential standalone/testing use but are effectively dead code in the pipeline.

6. **JSON logging format**: All log output is single-line JSON (`ts`, `level`, `stage`, `msg`). Do not use `print()` for debugging — use `logger.info()`.

7. **Windows compatibility**: The project runs on both Windows (PowerShell) and Linux (GitHub Actions). Use `pathlib.Path` for all file paths. Never use hardcoded `/` or `\` separators.

---

## Testing Guidelines

- Tests live in `tests/` and use plain `pytest` (no fixtures framework, no mocks library).
- Test helper `_make_paper()` creates minimal `ArxivPaper` instances for unit tests.
- Current coverage: dedup logic, query building, relevance response parsing. 邮件测试覆盖 STARTTLS、认证顺序、失败处理与报告保留。
- **Missing test coverage**: `deep_analysis`, `deep_research`, `report_generator`, `llm_client`. These involve LLM calls and should be tested with mocked `call_llm`.
- Run: `pytest tests/ -v`

---

## Dependencies

All in `requirements.txt` — keep this list minimal:

| Package      | Purpose                                    |
|--------------|--------------------------------------------|
| `arxiv`      | arXiv Search API client                    |
| `feedparser` | RSS feed parsing                           |
| `httpx`      | Async HTTP (OpenRouter API, PDF downloads) |
| `jinja2`     | Template rendering                         |
| `pyyaml`     | YAML config loading                        |

Do NOT add new dependencies without strong justification. Standard library (`asyncio`, `json`, `base64`, `smtplib`, `logging`, `subprocess`, `pathlib`) is preferred.
