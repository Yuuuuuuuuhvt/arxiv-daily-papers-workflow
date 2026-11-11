# arXiv Daily Papers Workflow

Collect arXiv papers, filter them against research interests, and generate scored HTML and Markdown reports with optional PDF analysis and email digests. The supplied profile covers embodied AI and VLA, world models, autonomous driving, agentic AI, robotics, and 3DGS, 3D reconstruction and simulation scenes. Analysis prompts currently produce Chinese text.

Agentic AI includes software, web, GUI and coding agents, tool use, memory, planning, multi-agent coordination and evaluation. Robotics includes manipulation, locomotion, navigation, SLAM, perception, control, hardware and simulation. 3DGS, 3D reconstruction and simulation scenes cover Gaussian splatting, NeRF, scene reconstruction, novel view synthesis, simulation-ready assets, scene generation and editing, physical properties and rendering efficiency. Each direction is independently relevant: software agents do not need physical robot experiments, and robotics papers do not need a foundation model to qualify as core research. Scene reconstruction papers need a 3D scene contribution, not a robot experiment.

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
| `reports/index.html` | Browser entry point with latest and historical reports |
| `reports/YYYY-MM-DD.html` | Offline reading page with formulas, search and direction filters |
| `reports/latest.html` | Latest dated HTML report; historical backfills do not replace it |
| `reports/YYYY-MM-DD.md` | Markdown report |
| `checkpoints/YYYY-MM-DD.json` | Saved classifications and completed model outputs |
| `papers_index.json` | Persistent index used to avoid processing papers again |
| `run-metadata/YYYY-MM-DD.json` | Collection counts and processing outcome |

The default report date uses UTC. Keep the data directory between runs to preserve deduplication. An empty retry preserves an existing report for the same date. Rejected papers are not indexed, so repeated limited trials can select the same rejected candidates; trial mode is a setup check, not a continuous subscription. Model stages can fall back to default analysis after API failures, so inspect the logs and report content before treating a run as successful model analysis.

## 2. Run on your own server

Follow [the self-hosting guide](docs/self-hosting.md) to configure the supplied systemd service and weekday timer. The service can use a separate disk for configuration, reports, the index, and PDF storage. It does not need a GitHub Actions runner or GPU access.

The same command runs one full collection locally or from another scheduler:

```bash
python -m src.main
```

Full runs prefer LaTeX source with the images referenced by the document. `paper_input.prefer_latex` defaults to `true`. Sources are pinned to the arXiv version and cached under `<PAPERS_DATA_DIR>/paper-inputs`; the parser expands supported local includes and records figure references and captions. Raster images retain their pixels; PDF figures contribute every page rendered at 150 DPI. The input manifest records file coverage, image hashes, dimensions, fallback reasons and request preparation status. Static TeX reading does not execute arbitrary macros, so it must reject unsupported document structures rather than promise compilation-equivalent output.

If source preparation fails, the pipeline sends every page of the version-verified PDF as an image alongside its extracted text. Missing files, unsupported images and resource limits never cause silent truncation. One request carries at most 50 MiB and 200 million image pixels. A paper above that is read in up to `paper_input.max_segments` requests: its images are split in order, every request repeats the complete text and records what its images show, and a final request writes the report from the text and all records. The report header states the number of segments. A paper that needs more requests than that fails and remains pending. OpenRouter input transforms are disabled so context overflow cannot silently remove middle sections. A provider rejection does not trigger a second paid analysis through a different input route.

Set `paper_input.prefer_latex: false` to use the previous PDF mode: files up to 5 MiB use native PDF input and larger files use local `pypdf` text extraction without image pixels. That legacy mode permits abstract fallback only when `pdf.require_pdf` is false. The default LaTeX mode requires complete source preparation or a usable PDF fallback regardless of that legacy switch. Email is sent only when all six SMTP settings are present and `SKIP_EMAIL` is not enabled. Local runs do not push to Git.

## 3. Configure an instance

`config/config.yaml` contains research directions, keyword rules, model IDs, scoring weights, PDF settings, and concurrency. Set `PAPERS_CONFIG` to use an instance-specific YAML copy without editing the checkout. Changing keywords within an existing direction only requires YAML changes; adding a new direction also requires updating `prompts/relevance_filter.txt`, `DIRECTION_DISPLAY` in `src/report_generator.py`, and `DIRECTIONS` in `src/html_report.py`.

| Variable | Purpose and default |
| --- | --- |
| `OPENROUTER_API_KEY` | Required for collection and `--check` while a stage uses the default `openrouter` provider |
| `PAPERS_RELEVANCE_FILTER_MODEL`, `PAPERS_DEEP_ANALYSIS_MODEL` | Optional `provider:model_id` override for one stage, replacing `models.*` in the YAML file |
| `CODEX_RELAY_BASE_URL`, `CODEX_RELAY_API_KEY` | Required by stages using the `codex-relay` provider |
| `GEMINI_RELAY_BASE_URL`, `GEMINI_RELAY_API_KEY` | Required by stages using the `gemini-relay` provider |
| `PAPERS_CONFIG` | YAML file; defaults to the checkout's `config/config.yaml` |
| `PAPERS_DATA_DIR` | Report and index directory; defaults to the checkout's `data` directory |
| `PAPERS_WEB_REPORTS` | Set to `1` for HTTP reading with per-paper detail loading; default keeps standalone HTML |
| `PAPERS_QUICK_START` | Set to `1` to use the isolated trial mode from a service environment file |
| `PAPERS_BUDGET_USD` | Optional per-run model budget in USD; requires matching model price policies under `budget.models` |
| `MAX_PAPERS` | Maximum new candidates per run; `0` means no limit |
| `PAPERS_DATE_MODE` | `current` by default; `previous-day` selects yesterday in UTC for a scheduled full-date query |
| `REPORT_DATE` | Optional UTC first-submission date in `YYYY-MM-DD` format |
| `SKIP_DEEP_RESEARCH` | Set to `1` to skip extended PDF-based analysis |
| `SKIP_EMAIL` | Set to `1` to suppress email |
| `SKIP_GIT_PUSH` | Set to `1` to suppress the in-process CI Git push path |
| `PDF_RELEASE_BASE_URL` | Optional repository URL for externally published PDF attachments; unset means no release links |

Relative `PAPERS_CONFIG` and `PAPERS_DATA_DIR` paths resolve from the process working directory. Use absolute paths for a scheduled service. PDF downloads use the separate `pdf.storage_dir` YAML setting, which resolves relative paths from the code checkout; use an absolute path to place PDFs on another disk.

Each stage under `models` names a `provider`. `openrouter` is the default and the only provider with cost accounting. `codex-relay` sends OpenAI Responses requests and `gemini-relay` sends Anthropic Messages requests to a relay you operate; both base URLs end in `/v1`. A relay reports no request cost, so `PAPERS_BUDGET_USD` is refused when any stage uses one. A relay receives papers as text and inline images only: keep `paper_input.prefer_latex` enabled, because the legacy native PDF input exists only on OpenRouter and otherwise falls back to the abstract. The Codex relay does not accept a temperature, so that setting is ignored there.

Real collection prepares the output directories and checks that they can be written before making network requests. The read-only `--check` command intentionally does not create these directories.

For a budgeted full run, disable trial mode and set `PAPERS_BUDGET_USD`, for example `30`. The program checks current model context limits and pricing, restricts provider prices and output length, and reserves the maximum possible cost before each request, including retries. The reservation covers the full advertised input context, so the run can stop conservatively before spending the entire budget. Unknown request costs keep their full reservation. This is a local run budget, not an OpenRouter account or key limit.

Budget records live in `run-metadata/YYYY-MM-DD.budget.json`. An existing ledger is never silently reset. Image requests additionally reserve per-image fees and set the provider image-price cap. The optional per-model `image_price_per_image` setting defaults to the prompt price per token; published image prices and model input modalities are checked before use. Use a new data directory for a separately authorized run. Checkpoints retain classifications, completed analyses and deep research. A budget stop writes a partial report with status `budget_exhausted`, suppresses email and publishing, leaves the deduplication index unchanged, and exits with status 3. `python -m src.main --resume YYYY-MM-DD` finishes a `partial` run from its checkpoint: it keeps the saved candidates, classifications, analyses and deep research, repeats only the missing steps, and then writes the report and index as a normal run does. It is refused together with `PAPERS_BUDGET_USD`, because the unfinished run already holds that date's ledger; a `budget_exhausted` run can therefore be resumed only with relay providers or without a budget. Other per-paper failures save available results with status `partial` and exit code 2, without updating the index, emailing, or publishing.

For daily scheduling, set `PAPERS_DATE_MODE=previous-day`; an explicit `REPORT_DATE` takes priority. Completed non-trial daily reports are reused without network requests when the dated HTML, Markdown and successful metadata already exist. Keep any existing budget ledger after a failed run: do not delete it to retry. The HTML embeds its styles, scripts and rendered MathML, so reading it does not require external assets.

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
