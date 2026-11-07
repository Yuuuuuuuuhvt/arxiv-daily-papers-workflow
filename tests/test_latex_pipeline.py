from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from datetime import datetime, timezone

import pytest
from PIL import Image
from pypdf import PdfWriter

from src import deep_research, latex_source
from src.models import ArxivPaper


def test_latex_payload_keeps_body_captions_and_every_image(monkeypatch, tmp_path):
    day = datetime(2026, 11, 5, tzinfo=timezone.utc)
    paper = ArxivPaper('sample', 2, 'Synthetic', 'Abstract', [], [], 'cs.RO', day, day, '', '')
    figures = []
    for name, color in [('method', 'red'), ('results', 'blue')]:
        path = tmp_path / f'{name}.png'
        Image.new('RGB', (20, 15), color).save(path)
        figures.append(latex_source.SourceFigure(path, name, name + ' caption'))
    async def prepare(*args):
        return latex_source.LatexSource('Main formula $x_1$\nAppendix final evidence', figures, {})
    calls = []
    async def inspect(**kwargs):
        calls.append(kwargs)
        content = kwargs['messages'][1]['content']
        assert 'Appendix final evidence' in content[0]['text']
        for i, figure in enumerate(figures):
            assert figure.caption in content[1 + 2*i]['text']
            image = base64.b64decode(content[2 + 2*i]['image_url']['url'].split(',')[1])
            assert image == figure.path.read_bytes()
        assert kwargs['stream'] is True and 'plugins' not in kwargs
        return 'Verified complete synthetic input'
    monkeypatch.setattr(latex_source, 'prepare_latex_source', prepare)
    monkeypatch.setattr(deep_research, 'call_llm', inspect)
    result = asyncio.run(deep_research.generate_deep_research(
        paper, {'model_id': 'test'}, 'test', source_config={'storage_dir':str(tmp_path / 'cache')}))
    assert result.startswith('> 输入依据：LaTeX') and len(calls) == 1
    manifest = json.loads((tmp_path / 'cache/inputs/samplev2.json').read_text())
    assert manifest['image_count'] == 2 and manifest['version'] == 2
    assert manifest['status'] == 'response_received'
    monkeypatch.setattr(deep_research, '_MAX_MULTIMODAL_BYTES', 1)
    with pytest.raises(ValueError, match='no content was truncated'):
        asyncio.run(deep_research.generate_deep_research(
            paper, {'model_id':'test'}, 'test', source_config={'storage_dir':str(tmp_path / 'cache')}))
    assert len(calls) == 1


def test_pdf_fallback_keeps_all_pages_and_checks_version(monkeypatch, tmp_path):
    day = datetime(2026, 11, 5, tzinfo=timezone.utc)
    paper = ArxivPaper('sample', 2, 'Synthetic', 'Abstract', [], [], 'cs.RO', day, day, '', '')
    pdf = tmp_path / 'sample.pdf'
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.add_blank_page(120, 100)
    writer.write(pdf)
    sidecar = pdf.with_suffix('.pdf.json')
    def version_record(version):
        sidecar.write_text(json.dumps({'version':version, 'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}))
    version_record(2)
    async def unavailable(*args):
        raise latex_source.SourceUnavailable('Missing referenced figure')
    calls = []
    async def inspect(**kwargs):
        calls.append(kwargs)
        assert sum(p['type']=='image_url' for p in kwargs['messages'][1]['content']) == 2
        return 'PDF visual analysis'
    monkeypatch.setattr(latex_source, 'prepare_latex_source', unavailable)
    monkeypatch.setattr(deep_research, 'call_llm', inspect)
    async def run():
        return await deep_research.generate_deep_research(paper, {'model_id':'test'}, 'test',
            pdf_config={'storage_dir':str(tmp_path)}, source_config={'storage_dir':str(tmp_path/'cache')})
    assert 'PDF 全部页面图像' in asyncio.run(run())
    version_record(1)
    with pytest.raises(FileNotFoundError, match='version-verified'):
        asyncio.run(run())
    assert len(calls)==1
