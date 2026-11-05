# arXiv Daily Papers Workflow

Collect arXiv papers, filter them against research interests, and generate scored Markdown reports with optional PDF analysis and email digests. The supplied profile focuses on embodied AI, world models, and autonomous driving. Analysis prompts currently produce Chinese text.

The pipeline runs as a normal Python command. You can run it locally, schedule it on a Linux server such as a Jetson AGX, or try it through GitHub Actions. A GitHub account, Git write token, email account, and the maintainer's data repository are not required for local or server use.

## 1. First local run

You need Python 3.11 or newer, access to arXiv and OpenRouter, and your own [OpenRouter API key](https://openrouter.ai/settings/keys). Model calls incur charges. Check the model IDs in `config/config.yaml` against your account's available models before running.

Clone the code without initializing submodules:

```bash
git clone https://github.com/Yuuuuuuuuhvt/arxiv-daily-papers-workflow.git
cd arxiv-daily-papers-workflow
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY='your-openrouter-key'
export PAPERS_DATA_DIR="$PWD/storage/data"
python -m src.main --check
python -m src.main --quick-start
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENROUTER_API_KEY = 'your-openrouter-key'
$env:PAPERS_DATA_DIR = Join-Path $PWD 'storage/data'
.\.venv\Scripts\python.exe -m src.main --check
.\.venv\Scripts\python.exe -m src.main --quick-start
```

`--check` validates local configuration and the presence of the API key without making network requests or writing data. It does not verify API authentication, model availability, or filesystem write permissions.

`--quick-start` processes at most two new candidate papers. It skips PDF downloads, DeepResearch, email, and Git pushes, even if those features are configured. Filtering and structured analysis still call OpenRouter. A smaller nonzero `MAX_PAPERS` limit is respected. The candidate limit is not a currency budget, and a run can produce an empty report if neither candidate is relevant.

With the commands above, trial results are stored under `storage/data/trial`. A full run writes directly under `storage/data`, so the trial cannot overwrite production reports or mark production papers as already processed.

| Path | Content |
| --- | --- |
| `reports/YYYY-MM-DD.md` | Daily report |
| `papers_index.json` | Persistent index used to avoid processing papers again |
| `run-metadata/YYYY-MM-DD.json` | Collection counts and processing outcome |

The default report date uses UTC. Keep the data directory between runs to preserve deduplication. An empty retry preserves an existing report for the same date. Rejected papers are not indexed, so repeated limited trials can select the same rejected candidates; trial mode is a setup check, not a continuous subscription. Model stages can fall back to default analysis after API failures, so inspect the logs and report content before treating a run as successful model analysis.

## 2. Run on your own server

Follow [the self-hosting guide](docs/self-hosting.md) to configure the supplied systemd service and weekday timer. The service can use a separate disk for configuration, reports, the index, and PDF storage. It does not need a GitHub Actions runner or GPU access.

The same command runs one full collection locally or from another scheduler:

```bash
python -m src.main
```

Full runs follow the PDF settings in YAML and enable DeepResearch unless disabled through the environment. Email is sent only when all six SMTP settings are present and `SKIP_EMAIL` is not enabled. Local runs do not push to Git. Set `SKIP_GIT_PUSH=1` when using an external CI scheduler that sets `GITHUB_ACTIONS`.

## 3. Configure an instance

`config/config.yaml` contains research directions, keyword rules, model IDs, scoring weights, PDF settings, and concurrency. Set `PAPERS_CONFIG` to use an instance-specific YAML copy without editing the checkout. Changing keywords within an existing direction only requires YAML changes; adding a new direction also requires updating `prompts/relevance_filter.txt` and `DIRECTION_DISPLAY` in `src/report_generator.py`.

| Variable | Purpose and default |
| --- | --- |
| `OPENROUTER_API_KEY` | Required for collection and `--check` |
| `PAPERS_CONFIG` | YAML file; defaults to the checkout's `config/config.yaml` |
| `PAPERS_DATA_DIR` | Report and index directory; defaults to the checkout's `data` directory |
| `PAPERS_QUICK_START` | Set to `1` to use the isolated trial mode from a service environment file |
| `MAX_PAPERS` | Maximum new candidates per run; `0` means no limit |
| `REPORT_DATE` | Optional UTC first-submission date in `YYYY-MM-DD` format |
| `SKIP_DEEP_RESEARCH` | Set to `1` to skip extended PDF-based analysis |
| `SKIP_EMAIL` | Set to `1` to suppress email |
| `SKIP_GIT_PUSH` | Set to `1` to suppress the in-process CI Git push path |
| `PDF_RELEASE_BASE_URL` | Optional repository URL for externally published PDF attachments; unset means no release links |

Relative `PAPERS_CONFIG` and `PAPERS_DATA_DIR` paths resolve from the process working directory. Use absolute paths for a scheduled service. PDF downloads use the separate `pdf.storage_dir` YAML setting, which resolves relative paths from the code checkout; use an absolute path to place PDFs on another disk.

Real collection prepares the output directories and checks that they can be written before making network requests. The read-only `--check` command intentionally does not create these directories.

Email is optional. Configure all of `MAIL_SMTP_HOST`, `MAIL_SMTP_PORT`, `MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_FROM`, and `MAIL_TO` to use SMTP with STARTTLS. Missing settings skip email. The username, sender, and recipient may differ. An `email_sent` result means SMTP accepted the message, not that it reached the inbox.

Keep secrets in your shell environment or a private service environment file. This project does not automatically load `.env` files. Do not put keys in YAML or commit private environment files.

## 4. Try through GitHub Actions

1. Fork this repository and enable Actions in your fork.
2. Add `OPENROUTER_API_KEY` under Settings > Secrets and variables > Actions.
3. Select Actions > Try Papers > Run workflow.
4. Open the completed run and download the `paper-trial` artifact to read the report and metadata.

The [Try Papers workflow](.github/workflows/quick-start.yml) uses `--quick-start`, needs only that secret, and has read-only repository permissions. It does not initialize the data submodule, send email, publish releases, or push commits. Its files are retained for seven days. Each run starts with a new index, so it is a trial rather than a persistent daily subscription.

GitHub requires a manually dispatched workflow to exist on the default branch. See [manual workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow) and [workflow artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data).

The older `daily-run.yml`, `.forgejo/` workflows, and `.gitmodules` describe the maintainer's publishing setup. They are not ready-made deployment settings for a new fork. See [maintainer deployment](docs/maintainer-deployment.md) before using them.

## 5. Development and verification

The pipeline is implemented in `src/main.py`: fetch, deduplicate, filter, analyze, download PDFs, run DeepResearch, render reports, optionally email, and optionally publish through a scheduler. Prompts live in `prompts/`; Jinja templates live in `templates/`.

Install pytest into the same virtual environment, then run tests for the area you change:

```bash
python -m pip install pytest
python -m pytest tests/test_onboarding.py tests/test_migration.py -q
```

These tests cover local startup, isolated storage, trial side effects, report links, and historical collection behavior. They do not establish real API authentication, model quality, email delivery, or hosted scheduler operation.
