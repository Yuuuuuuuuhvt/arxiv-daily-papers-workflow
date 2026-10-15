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

邮件推送为可选功能。六项 SMTP 环境变量齐全时通过 STARTTLS 发信，参数见下表。认证用户名、发件地址与收件地址分别配置，不要求三者相同。

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
| `MAIL_SMTP_HOST` | SMTP 主机，当前为 `mail.luonto.me` |
| `MAIL_SMTP_PORT` | STARTTLS 端口，当前为 `587` |
| `MAIL_USERNAME` | SMTP 登录用户名，当前为 `eai-papers@luonto.me` |
| `MAIL_PASSWORD` | SMTP 登录口令，仅保存到私有 secret 与 Actions secrets |
| `MAIL_FROM` | 发件地址，当前为 `eai-papers@luonto.me` |
| `MAIL_TO` | 单个收件地址，当前为 `eai-papers@luonto.me` |

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

## Forgejo production workflow and backfill

Production now uses `.forgejo/workflows/daily-run.yml` on the TX Linux Runner.
The `PAPERS_WRITE_TOKEN` secret supplies `FORGEJO_PAPERS_TOKEN`, which grants write access only to this code repository and the
`embodied-daily-papers` data repository. The pipeline, PDF release attachments,
data push and submodule pointer push all use Forgejo. The retained GitHub workflow
stays disabled during migration.

生产邮件使用上表的六个 Forgejo Actions secrets，来源为 recitativo 的
`secret/mail/eai-papers.env`。任一变量缺失时仅跳过邮件，采集、报告、PDF 发布和数据更新继续执行。
`SKIP_EMAIL=1` 显式禁用邮件。无效端口、TLS、认证或发送失败会记录错误并令
`data/run-metadata` 中的 `email_sent` 为 false，不删除已生成的报告。
SMTP 使用系统证书校验和 30 秒连接及套接字操作超时，不自动重试以避免重复投递。
TX 的 Runner 使用 host 网络，邮件域名由 `DOMAIN,mail.luonto.me,DIRECT` 规则直连。
邮箱配额为 1024 MiB，本域收发不经过 SES 外部中继。

SMTP 调用依据 [Python smtplib 文档](https://docs.python.org/3/library/smtplib.html#smtplib.SMTP.starttls)，
先建立 TLS，再重新 EHLO 并认证。`email_sent` 仅表示 SMTP 接受提交，实际收件仍须经邮箱回读验证。

Manual input `report_date` selects a UTC first-submission date in `YYYY-MM-DD`
format. Historical runs query every page for that date and do not combine the
current RSS feed with historical results. API failures fail the run instead of
being recorded as empty days. The resulting report and `data/run-metadata` record
include the requested date and processing outcome. A weekday run without an
explicit date retains the current API/RSS feed behavior.
An empty retry preserves an existing report and its corresponding run summary.
A legacy report without a summary receives a `retained_existing_report` marker
in the newly recorded attempt summary.

The `validation` input limits analysis to two new papers, skips DeepResearch,
email, release uploads and production pushes. It verifies real collection and
analysis in an isolated job checkout before production scheduling is accepted.

Historical collection selects the first UTC submission date and analyzes the
currently available arXiv version. It does not reconstruct past RSS entries or
past versions. Backfill uses skip_email to prevent delayed notification bursts.
PDF attachments are verified against existing files and published to both
Forgejo and the GitHub fallback before production data commits.


## 日报邮件独立验证

手动运行 `.forgejo/workflows/mail-check.yml`，输入已有日报日期 `report_date`。
作业只读取该日报并发送一封验证邮件，不调用模型，不发布附件，不写入生产分支。
当前收发使用同一邮箱，作业通过 IMAP 993 回读 INBOX，核对完整正文并输出认证邮件头。
SMTP 提交成功与收件确认分别验证，域内认证结果按实际邮件头记录，不等同于外域投递认证。
