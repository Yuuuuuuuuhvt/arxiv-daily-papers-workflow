import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from src import release_assets as release


class ReleaseTests(unittest.TestCase):
    def test_empty_day_needs_no_release_credentials(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(release, 'ROOT_DIR', Path(folder)), patch.object(release.urllib.request, 'build_opener') as opener:
            release.publish_pdfs()
            opener.assert_not_called()

    def test_different_existing_pdf_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'storage/pdfs/2026-10-09'
            path.mkdir(parents=True)
            (path / 'one.pdf').write_bytes(b'new')
            class Opener:
                def open(self, request, timeout):
                    return io.BytesIO(json.dumps([{'id': 1, 'tag_name': 'v2026-10-09', 'assets': [{'name': 'one.pdf', 'digest': 'sha256:wrong'}]}]).encode())
            env = {'REPORT_DATE': '2026-10-09', 'FORGEJO_PAPERS_TOKEN': 'test', 'GITHUB_RELEASE_SYNC_TOKEN': 'test'}
            with patch.dict(os.environ, env), patch.object(release, 'ROOT_DIR', root), patch.object(release.urllib.request, 'build_opener', return_value=Opener()), patch.object(release.subprocess, 'run') as upload:
                with self.assertRaisesRegex(RuntimeError, 'Existing PDF differs'):
                    release.publish_pdfs()
                upload.assert_not_called()

    def test_fallback_failure_keeps_primary_draft(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'storage/pdfs/2026-10-09'
            path.mkdir(parents=True)
            pdf = path / 'one.pdf'
            pdf.write_bytes(b'pdf')
            class Opener:
                def open(self, request, timeout):
                    calls.append((request.full_url, request.get_method()))
                    if request.get_method() == 'PATCH':
                        raise urllib.error.HTTPError(request.full_url, 503, 'unavailable', {}, None)
                    return io.BytesIO(json.dumps([{'id': 1, 'tag_name': 'v2026-10-09', 'draft': True, 'assets': [{'name': 'one.pdf', 'digest': 'sha256:' + release.digest(pdf)}]}]).encode())
            env = {'REPORT_DATE': '2026-10-09', 'FORGEJO_PAPERS_TOKEN': 'test', 'GITHUB_RELEASE_SYNC_TOKEN': 'test'}
            with patch.dict(os.environ, env), patch.object(release, 'ROOT_DIR', root), patch.object(release.urllib.request, 'build_opener', return_value=Opener()):
                with self.assertRaises(urllib.error.HTTPError):
                    release.publish_pdfs()
            patches = [url for url, method in calls if method == 'PATCH']
            self.assertEqual(patches, [release.GITHUB + '/repos/Yuuuuuuuuhvt/arxiv-daily-papers-workflow/releases/1'])
