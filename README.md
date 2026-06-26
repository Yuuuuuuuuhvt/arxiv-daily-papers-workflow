# arXiv Daily Papers Workflow

This repository contains the automation code for the daily arXiv paper workflow. It fetches papers related to embodied AI, world models, and autonomous driving, deduplicates them against a data submodule, filters relevance with an LLM, generates structured Markdown reports, optionally sends an email digest, and updates the data repository.

## Pipeline

| Stage | Module | Output |
|---|---|---|
| Configuration loading | `src/config.py` | Parsed research directions, model settings, scoring weights, email settings, and PDF settings |
| Paper fetching | `src/fetcher.py` | Candidate papers from arXiv RSS and API searches |
| Deduplication | `src/dedup.py` | Papers not already present in `data/papers_index.json` |
| Relevance filtering | `src/relevance_filter.py` | Direction labels and relevance decisions from the filter prompt |
| Deep analysis | `src/deep_analysis.py` | Structured paper scores and analysis fields |
| Deep research | `src/deep_research.py` | Longer academic interpretation generated from `prompts/deep_research.txt` |
| Report generation | `src/report_generator.py` | Markdown daily report rendered from `templates/daily_report.md.j2` |
| Email digest | `src/email_sender.py` | Optional HTML digest rendered from `templates/email_digest.html.j2` |
| Git update | `src/git_ops.py` | Data submodule commit and root submodule pointer update |

The command entry points are `python -m src.main` and `python -m src`.

## Repository Layout

| Path | Purpose |
|---|---|
| `.github/workflows/daily-run.yml` | Weekday GitHub Actions schedule and manual dispatch workflow |
| `config/config.yaml` | arXiv categories, research directions, keyword rules, OpenRouter model choices, scoring weights, email settings, concurrency, PDF behavior, and hjfy link behavior |
| `config/affiliations.json` | Institution ranking and affiliation metadata used during scoring |
| `prompts/` | LLM prompts for relevance filtering, deep analysis, and deep research interpretation |
| `src/` | Python implementation of the workflow |
| `templates/` | Markdown and HTML Jinja templates |
| `tests/` | Unit tests for deduplication, fetching, and relevance filtering |
| `data/` | Git submodule that stores generated reports, PDFs, and `papers_index.json` |

## Local Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

PowerShell users can set the required OpenRouter key locally.

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

macOS and Linux users can set the same value through the shell.

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Email delivery is optional. Set `QQ_MAIL_ADDRESS` and `QQ_MAIL_AUTH_CODE` only when the run should send the digest through QQ Mail SMTP.

## Run Locally

```bash
python -m src.main
```

Local runs can generate reports and update local data files. When the workflow runs in GitHub Actions, it checks out the `data` submodule, uses `PAT_TOKEN` for write access, runs the pipeline, commits data changes, and then updates the root submodule pointer when needed.

## GitHub Actions

The workflow runs on weekdays at UTC 05:30, which corresponds to Beijing time 13:30. Manual dispatch is also enabled.

Required repository secrets are:

| Secret | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM calls through OpenRouter |
| `PAT_TOKEN` | Push access to both this repository and the data submodule repository |
| `QQ_MAIL_ADDRESS` | Optional QQ Mail sender address |
| `QQ_MAIL_AUTH_CODE` | Optional QQ Mail SMTP authorization code |

The workflow sets `SKIP_GIT_PUSH=1` during the Python pipeline and handles commits explicitly in later shell steps. This keeps Git side effects visible in the workflow file.

## Configuration

Research directions are defined in `config/config.yaml`. Each direction can use title keywords, abstract keywords, abstract keyword combinations, and arXiv categories. The current focus favors world models that support embodied AI downstream tasks, including manipulation, locomotion, planning, VLA models, diffusion policies, model based reinforcement learning, latent dynamics, and sim to real work.

Model settings are also in `config/config.yaml`. The relevance filter currently uses a fast model, while deep analysis uses a stronger model one paper at a time. Concurrency limits for LLM and PDF work are configured separately.

## Tests

```bash
pip install pytest
pytest tests -v
```

The unit tests cover deterministic parts of the pipeline. End to end behavior still depends on network access, arXiv availability, model API responses, configured secrets, and the state of the `data` submodule.

## Data Policy

The code repository should stay small. Generated reports, downloaded PDFs, and the cumulative `papers_index.json` belong in the `data/` submodule. API keys, mail authorization codes, and personal access tokens must stay in local environment variables or GitHub Actions secrets.
