# AGENTS.md -- arXiv Daily Papers Agent

## Project Overview

全自动 arXiv 论文日报系统。每个工作日抓取 Embodied AI 与 VLA、World Models、Autonomous Driving、Agentic AI、Robotics 和 3DGS / 三维重建与仿真场景六个方向的新论文，经 LLM 过滤、分析、解读后生成 Markdown 日报并可选邮件推送。

- **技术栈**: Python 3.11+, asyncio, httpx, Jinja2, PyYAML, feedparser, arxiv
- **LLM 网关**: 默认经 OpenRouter (openrouter.ai) 路由。每个阶段的 `models.*.provider` 也可以选择 `codex-relay`（OpenAI Responses 协议）或 `gemini-relay`（Anthropic Messages 协议）。
- **部署**: 独立 Python 命令、Linux systemd 用户服务，以及可选的 GitHub Actions。维护者的生产发布配置位于 `.forgejo/`。

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
    -> Jinja2 render report -> save dated Markdown and offline HTML, plus reports/latest.html
```

### Key Design Decisions

1. **Hybrid fetch**: Current daily collection uses RSS new/cross announcements, local category and keyword matching, then API metadata lookup for every selected ID. Explicit REPORT_DATE uses historical UTC submission-date API search.
2. **3-tier relevance**: `core` means a direct contribution to a tracked direction; `peripheral` means supporting context or an indirect connection; `not_relevant` means outside scope. Agentic and Robotics papers do not need a foundation model to qualify as core. Only core papers get DeepResearch.
3. **Legacy PDF mode**: Local PDFs are base64-encoded and sent directly to OpenRouter, avoiding server-side URL fetch failures (OpenRouter 502 when downloading arXiv URLs). Files up to 5 MiB use native document input. Larger PDFs use local pypdf extraction of every page, without image pixels or OCR.
4. **Concurrency**: `asyncio.Semaphore` controls parallel LLM calls (`max_concurrent_llm` for analysis, `max_concurrent_pdf` for DeepResearch).
5. **Resume**: `--resume YYYY-MM-DD` 从 `checkpoints/YYYY-MM-DD.json` 继续未完成的运行，不重新抓取，只补做缺失的分类、分析与 DeepResearch。当天公告的 RSS 批次在之后无法再次取得，所以检查点是补完该批次的唯一依据。
6. **Fail-safe**: Every LLM stage has fallback — parse failures default to `peripheral`, analysis failures use defaults, DeepResearch only falls back to abstract input when require_pdf is false; strict failures produce a partial report.

---

## Directory Structure

```text
src/                        # Python package, run via: python -m src.main
  main.py                   # Pipeline orchestrator — the ONLY entry point
  config.py                 # Path constants (ROOT_DIR, DATA_DIR, etc.) + YAML/env loading
  models.py                 # Dataclasses: ArxivPaper, RelevanceResult, AnalysisResult
  fetcher.py                # arXiv API + RSS hybrid fetching
  dedup.py                  # ID-based deduplication against papers_index.json
  llm_client.py             # call_llm entry: OpenRouter wrapper (retry, backoff, budget) and relay dispatch
  relay_client.py           # Responses and Messages request building and SSE parsing for relay providers
  relevance_filter.py       # Batch LLM classification with few-shot examples
  deep_analysis.py          # Per-paper structured analysis (scores, tags, affiliations)
  deep_research.py          # LaTeX+images first, complete PDF-page fallback, input manifests
  latex_source.py           # Versioned source download and conservative static include/image parsing
  figure_input.py           # Raster and all-page PDF figure conversion with pixel limits
  pdf_downloader.py         # Async PDF download from arXiv
  report_generator.py       # Markdown/email rendering and report persistence
  html_report.py             # Offline daily HTML with sanitized Markdown and MathML
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
  daily_report.html.j2      # Offline daily reading page
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
- **PDF processing**: Small PDFs use native input; large PDFs use full-page local text extraction. require_pdf=true rejects missing or unusable PDFs. The pipeline preserves completed results and returns partial status for per-paper failures.

### Configuration

- **Tunable parameters live in `config/config.yaml`**. `PAPERS_CONFIG` selects an instance-specific YAML file; `PAPERS_DATA_DIR` selects persistent report and index storage. The `--quick-start` preset deliberately caps trials at two candidates and suppresses external side effects other than model calls.
- **Prompts are plain text files** in `prompts/` — loaded at runtime via `config.load_prompt()`. Edit prompts to adjust LLM behavior without touching code.
- **Environment variables**: `OPENROUTER_API_KEY` (required while a stage uses the `openrouter` provider), `PAPERS_RELEVANCE_FILTER_MODEL` 与 `PAPERS_DEEP_ANALYSIS_MODEL`（可选，格式为 `provider:model_id`，覆盖对应阶段的模型）, `CODEX_RELAY_BASE_URL`、`CODEX_RELAY_API_KEY`、`GEMINI_RELAY_BASE_URL`、`GEMINI_RELAY_API_KEY`（阶段使用对应中转时必填，地址以 `/v1` 结尾）, `MAIL_SMTP_HOST`、`MAIL_SMTP_PORT`、`MAIL_USERNAME`、`MAIL_PASSWORD`、`MAIL_FROM`、`MAIL_TO`，六项齐全时通过 STARTTLS 发信，缺项只跳过邮件, `SKIP_DEEP_RESEARCH` (optional flag).

### Templates

- Jinja2 with `autoescape=False` for Markdown, `autoescape=True` for HTML email.
- Template variables are pre-computed in `report_generator.py` via `_paper_view()` — templates should not contain business logic.

---

## How to Run

```powershell
# Local run (requires OPENROUTER_API_KEY)
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
python -m src.main --check
python -m src.main --quick-start
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
3. Add display names in `report_generator.py` `DIRECTION_DISPLAY` and `html_report.py` `DIRECTIONS`.

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

1. **OpenRouter native PDF engine**: Intermittent 502 when OpenRouter fetches PDF URLs server-side. Solved by sending base64-encoded local PDFs instead. Files above 5 MiB use local pypdf extraction before model input.

2. **arXiv rate limiting**: The `arxiv` library respects rate limits, but `api_delay_seconds` (default 3s) adds extra safety. PDF downloads from arxiv.org may get 429'd under heavy concurrency — `pdf_downloader.py` uses a semaphore (default 5).

3. **`papers_index.json` is the single source of truth for dedup**: If this file is corrupted or deleted, all papers will be re-processed. The file has a `.bak` backup created on every save.

4. **Git push only runs when `GITHUB_ACTIONS` is set and `SKIP_GIT_PUSH` is not enabled**. Publishing workflows set `SKIP_GIT_PUSH=1` and own their Git steps. The trial preset always suppresses Git pushes. Independent systemd deployment is described in `docs/self-hosting.md`.

5. **`analyze_all()` and `generate_all_deep_research()`** are legacy serial wrappers — unused by `main.py` which implements its own concurrent versions. They exist for potential standalone/testing use but are effectively dead code in the pipeline.

6. **JSON logging format**: All log output is single-line JSON (`ts`, `level`, `stage`, `msg`). Do not use `print()` for debugging — use `logger.info()`.

7. **Relay providers**: 中转不返回请求费用，所以任一阶段使用中转时 `PAPERS_BUDGET_USD` 会在启动检查中被拒绝。中转只接受文本与内联图片，`file` 内容在本地直接报错。Codex 中转拒绝 `temperature` 参数，请求体不发送该字段；Anthropic Messages 协议必须带 `max_tokens`，缺省值见 `relay_client.py`。中转地址属于私有信息，只从环境变量读取，不写入仓库。

8. **Windows compatibility**: The project runs on both Windows (PowerShell) and Linux (GitHub Actions). Use `pathlib.Path` for all file paths. Never use hardcoded `/` or `\` separators.

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

## LaTeX input contract

Full pipeline runs default to paper_input.prefer_latex=true. All resolved source images are sent with document text and captions; unknown or missing source content triggers PDF-page fallback. Do not replace missing images with captions, truncate oversized payloads, or reuse PDFs from another version. Paper input manifests are separate from budget ledgers and must never contain credentials. Images require a verified model image policy, per-image fee reservation and provider price cap. Tests for parser, rendering and multimodal pipeline are targeted; do not run unrelated test suites.
