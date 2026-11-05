# 1. Independent Linux deployment

The pipeline runs once and exits. A systemd user service starts that command, and a timer schedules it. GitHub Actions, a Git write token, a separate data repository, and GPU access are not required. Model inference continues to use OpenRouter.

Use Python 3.11 or newer, Git, GNU `env` with `--chdir`, and a Linux host with a working systemd user manager. The commands below assume a dedicated checkout and data directory owned by the service user. Replace `/opt/arxiv-daily-papers` and `/var/lib/arxiv-daily-papers` with writable paths on your host. A Jetson can place both on its NVMe disk. Install Python in a separate environment rather than replacing JetPack's system Python.

# 2. Prepare code and configuration

Clone without `--recurse-submodules`, then create the environment:

```bash
git clone https://github.com/Yuuuuuuuuhvt/arxiv-daily-papers-workflow.git /opt/arxiv-daily-papers
cd /opt/arxiv-daily-papers
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
install -d -m 700 "$HOME/.config/arxiv-daily-papers"
```

On a new installation, copy `deploy/systemd/environment.example` to `~/.config/arxiv-daily-papers/environment` and `config/config.yaml` to `~/.config/arxiv-daily-papers/config.yaml`. Do not overwrite existing instance settings when updating the code. Set the environment file's permissions to `600`.

Edit the environment file using these settings:

| Setting | Value |
| --- | --- |
| `PAPERS_APP_DIR` | Absolute checkout path containing `.venv` |
| `PAPERS_CONFIG` | Absolute path to the instance YAML file |
| `PAPERS_DATA_DIR` | Absolute path for persistent reports, index, and run metadata |
| `OPENROUTER_API_KEY` | Your own API key |

The file uses `NAME=value` lines without `export`. systemd does not expand `$HOME`, `~`, or other variables in these values. Keep secrets out of the checkout. In the instance YAML, set `pdf.storage_dir` to an absolute path on the data disk. To avoid PDF downloads during initial validation, set `pdf.download_enabled` to `false`.

The sample sets `PAPERS_QUICK_START=1`: it limits each run to two new candidates, skips PDFs, DeepResearch and email, and disables Git pushes. Trial reports and the trial index go under `<PAPERS_DATA_DIR>/trial`, separately from production. These settings still make paid filtering and analysis requests. They are trial settings, not a complete daily collection.

# 3. Install and verify the service

For a new service installation:

```bash
install -d "$HOME/.config/systemd/user"
install -d -m 700 "$HOME/.local/state/arxiv-daily-papers"
install -m 644 deploy/systemd/arxiv-daily-papers.service "$HOME/.config/systemd/user/"
install -m 644 deploy/systemd/arxiv-daily-papers.timer "$HOME/.config/systemd/user/"
systemd-analyze --user verify "$HOME/.config/systemd/user/arxiv-daily-papers.service" "$HOME/.config/systemd/user/arxiv-daily-papers.timer"
systemctl --user daemon-reload
systemctl --user start arxiv-daily-papers.service
systemctl --user show arxiv-daily-papers.service -p Result -p ExecMainStatus
tail -n 100 "$HOME/.local/state/arxiv-daily-papers/service.log"
```

Starting the service runs the local setup check first, then one real collection. The service is `oneshot`: it normally becomes inactive after completing. Check `Result=success`, `ExecMainStatus=0`, the generated report, and the run metadata rather than expecting it to remain active. The pipeline can use fallback analysis when a model fails; service exit success alone is not proof of valid model output.

With the sample trial settings, read `<PAPERS_DATA_DIR>/trial/reports/YYYY-MM-DD.md` and `<PAPERS_DATA_DIR>/trial/run-metadata/YYYY-MM-DD.json`. Full runs use the same paths without the `trial` component. Keep each mode's `papers_index.json` between runs. Use this service for manual runs after installation; systemd will not launch overlapping instances of the same service. Do not simultaneously run the Python command or another scheduler against the same data directory. Actual collection creates and checks its output directories before making network requests, so a storage permission error fails before model calls.

# 4. Enable scheduled collection

After validating the result, set `PAPERS_QUICK_START=0` and keep `MAX_PAPERS=0` for complete collection. Set `SKIP_DEEP_RESEARCH=0` if extended analysis is desired. Enable PDF downloads in the instance YAML when needed. Set all six SMTP variables and `SKIP_EMAIL=0` only when email delivery is desired. The service reads the environment file again on its next start. Do not use a capped trial as a continuous subscription: rejected candidates are not indexed and can be selected again.

The supplied timer runs at 05:30 UTC on weekdays. Its explicit UTC schedule does not change with the server timezone or daylight saving time. `Persistent=true` performs one catch-up activation if a scheduled run was missed while the user manager was stopped; it does not backfill every missed date.

To keep the user manager running across logout and boot, an administrator can enable lingering for the service account:

```bash
sudo loginctl enable-linger "$USER"
systemctl --user enable --now arxiv-daily-papers.timer
systemctl --user list-timers arxiv-daily-papers.timer
```

Before transferring an existing production subscription, preserve its report data and index, verify a separate trial on the new host, and stop the old scheduler before enabling this timer. Otherwise, two instances can analyze the same papers and send duplicate notifications. This repository does not automatically migrate or disable an existing GitHub or Forgejo deployment.

# 5. Operation and recovery

Use `systemctl --user start arxiv-daily-papers.service` for an extra run. Stop future scheduled runs with `systemctl --user disable --now arxiv-daily-papers.timer`; this does not stop a collection that is already running. A running collection can be stopped separately with `systemctl --user stop arxiv-daily-papers.service`.

Back up the whole data directory and the private instance configuration before upgrading. Stop scheduling and wait for any active collection to finish before changing code or dependencies. Keep the checkout commit with the backup so the code can be restored. An interrupted write is not transactional across the index, report, and email, so inspect those files and logs before rerunning a failed job.

Missing keys or invalid local settings fail before collection. The service appends stdout and stderr to `~/.local/state/arxiv-daily-papers/service.log`, which remains readable on hosts without a user journal. Rotate or archive this file as part of regular host maintenance. Network and model failures require inspecting this log. No automatic whole-job restart is configured because replaying a job can repeat external side effects.

Local PDF downloads stay on the host. Set `PDF_RELEASE_BASE_URL` only when a separate publishing process uploads the matching files. The service itself does not publish releases; arXiv PDF links remain available in the reports.

The service relies on documented systemd behavior for [environment substitution](https://manpages.ubuntu.com/manpages/jammy/man5/systemd.service.5.html), [environment files and log output](https://manpages.ubuntu.com/manpages/jammy/man5/systemd.exec.5.html), and [persistent timers](https://manpages.ubuntu.com/manpages/jammy/man5/systemd.timer.5.html).
