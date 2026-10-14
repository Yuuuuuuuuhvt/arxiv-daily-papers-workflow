from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT_DIR

FORGEJO = 'https://git.luonto.me'
GITHUB = 'https://api.github.com'
logger = logging.getLogger(__name__)


class ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        old = urllib.parse.urlsplit(request.full_url)
        new = urllib.parse.urlsplit(new_url)
        if new.scheme != 'https' or old.hostname == 'git.luonto.me' and new.hostname != old.hostname:
            raise RuntimeError('Untrusted release redirect')
        result = super().redirect_request(request, response, code, message, headers, new_url)
        if old.hostname != new.hostname:
            result.remove_header('Authorization')
        return result


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def publish_pdfs() -> None:
    date_str = os.environ.get('REPORT_DATE', '').strip() or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    datetime.strptime(date_str, '%Y-%m-%d')
    pdfs = sorted((ROOT_DIR / 'storage/pdfs' / date_str).glob('*.pdf'))
    if not pdfs:
        logger.info('No PDF attachments for %s', date_str)
        return
    providers = [
        (FORGEJO + '/api/v1/repos/yurun/arxiv-daily-papers-workflow', 'token ' + os.environ['FORGEJO_PAPERS_TOKEN']),
        (GITHUB + '/repos/Yuuuuuuuuhvt/arxiv-daily-papers-workflow', 'Bearer ' + os.environ['GITHUB_RELEASE_SYNC_TOKEN']),
    ]
    opener = urllib.request.build_opener(ReleaseRedirect())
    tag = 'v' + date_str
    releases = []
    for base, authorization in providers:
        def request(path: str, method: str = 'GET', body: dict | None = None):
            req = urllib.request.Request(base + path, method=method,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Authorization': authorization, 'Content-Type': 'application/json'})
            with opener.open(req, timeout=60) as response:
                return json.load(response)

        # 列表包含前一次失败留下的草稿；重试不能创建相同日期的第二个发布。
        release = next((r for r in request('/releases?per_page=100&limit=100') if r['tag_name'] == tag), None)
        if release is None:
            release = request('/releases', 'POST', {'tag_name': tag, 'name': 'arXiv Papers ' + date_str,
                'body': 'Daily papers for ' + date_str, 'draft': True, 'target_commitish': os.environ.get('GITHUB_SHA', 'main')})
        existing = {asset['name']: asset for asset in release.get('assets', [])}
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as config:
            config.write('header = "Authorization: ' + authorization + '"\n')
            config.flush()
            for pdf in pdfs:
                asset = existing.get(pdf.name)
                if asset:
                    if asset.get('digest'):
                        actual = asset['digest'].removeprefix('sha256:')
                    else:
                        h = hashlib.sha256()
                        req = urllib.request.Request(asset['browser_download_url'], headers={'Authorization': authorization})
                        with opener.open(req, timeout=600) as response:
                            for chunk in iter(lambda: response.read(1 << 20), b''):
                                h.update(chunk)
                        actual = h.hexdigest()
                    if actual != digest(pdf):
                        raise RuntimeError('Existing PDF differs: ' + pdf.name)
                    continue
                if base.startswith(FORGEJO):
                    args = ['--form', 'attachment=@' + str(pdf), base + '/releases/' + str(release['id']) + '/assets']
                else:
                    url = release['upload_url'].split('{', 1)[0] + '?name=' + urllib.parse.quote(pdf.name, safe='')
                    args = ['--header', 'Content-Type: application/pdf', '--data-binary', '@' + str(pdf), url]
                subprocess.run(['curl', '--silent', '--show-error', '--fail', '--max-time', '600', '--config', config.name, *args],
                    check=True, stdout=subprocess.DEVNULL)
        releases.append((base, authorization, release))
    # 回退附件先公开，之后主仓发布才可用于正式报告中的下载链接。
    for base, authorization, release in reversed(releases):
        if release.get('draft'):
            req = urllib.request.Request(base + '/releases/' + str(release['id']), method='PATCH',
                data=b'{"draft":false}', headers={'Authorization': authorization, 'Content-Type': 'application/json'})
            with opener.open(req, timeout=60) as response:
                response.read()


if __name__ == '__main__':
    publish_pdfs()
