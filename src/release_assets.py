from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT_DIR

BASE = "https://git.luonto.me/api/v1/repos/yurun/arxiv-daily-papers-workflow/releases"
logger = logging.getLogger(__name__)


def publish_pdfs() -> None:
    date_str = os.environ.get("REPORT_DATE", "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    datetime.strptime(date_str, "%Y-%m-%d")
    directory = ROOT_DIR / "storage/pdfs" / date_str
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        logger.info("No PDF attachments for %s", date_str)
        return
    token = os.environ["FORGEJO_PAPERS_TOKEN"]
    tag = "v" + date_str

    def request(url: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": "token " + token, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    try:
        release = request(BASE + "/tags/" + urllib.parse.quote(tag, safe=""))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        release = request(BASE, {"tag_name": tag, "name": "arXiv Papers " + date_str,
                                 "body": "Daily papers for " + date_str, "draft": False})
    existing = {asset["name"] for asset in release.get("assets", [])}
    # curl 在文件上传时直接流式读取 PDF，令牌只进入受限临时配置，不出现在命令参数。
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as config:
        config.write('header = "Authorization: token ' + token + '"\n')
        config.flush()
        for pdf in pdfs:
            if pdf.name in existing:
                continue
            subprocess.run(["curl", "--silent", "--show-error", "--fail", "--max-time", "600",
                            "--config", config.name, "--form", "attachment=@" + str(pdf),
                            BASE + "/" + str(release["id"]) + "/assets"],
                           check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    publish_pdfs()
