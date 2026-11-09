from __future__ import annotations

import json
import re
from html.parser import HTMLParser

from src.html_report import generate_html_report, render_markdown


class Contents(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[tuple[str, str | None]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)

    def handle_data(self, value):
        self.text.append(value)


def embedded_papers(page: str) -> list[dict]:
    raw = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
    return json.loads(raw)["papers"]


def test_markdown_math_lists_and_cjk_bold():
    nested = render_markdown("1. 父项\n   - 子项一\n   - 子项二")
    assert re.search(r"<ol>\s*<li>父项\s*<ul>", nested)
    assert nested.count("<li>") == 3
    assert "<strong>模型预测控制（MPC）</strong>的系统" in render_markdown("**模型预测控制（MPC）**的系统")
    assert "<msubsup>" in render_markdown("矩阵 $W_i^{2}$ 的范数")
    display = render_markdown("公式：\n$$x=1$$\n继续说明。")
    assert '<span class="math display">' in display and "<div" not in display
    # 价格不能被当成公式分隔符, 带 LaTeX 命令的邻接算子仍是公式.
    prices = render_markdown(r"价格 $5 和 $10，变化 13$\rightarrow$29。")
    assert "$5 和 $10" in prices and "<mo>→</mo>" in prices.replace("&#8594;", "→")


def test_code_and_untrusted_content_are_inert():
    source = (
        "```python\nif x < 0: print(\"$x_1$\", \"<tag>\")\n```\n\n"
        "`$x_1$` <script>alert(1)</script> <img src=\"https://invalid.test/x\" onerror=\"alert(1)\">\n"
        "[链接](javascript:alert(1)) ![图片](https://invalid.test/a.png) $\\text{<img src=x onerror=alert(1)>}$\n"
    )
    parser = Contents()
    parser.feed(render_markdown(source))
    assert 'if x < 0: print("$x_1$", "<tag>")' in "".join(parser.text)
    assert not {"script", "img", "iframe", "style"} & set(parser.tags)
    assert not any(k.startswith("on") or k == "style" for k, _ in parser.attributes)
    assert all(v.startswith(("http://", "https://", "#")) for k, v in parser.attributes if k == "href")

    state = {
        "date": "2026-01-02",
        "candidates": [{"arxiv_id": "0000.00001", "title": "</script><b>x</b>", "abstract": "a\u2028b",
                        "abs_url": "javascript:alert(1)", "pdf_url": "https://example.test/p.pdf"}],
        "core_ids": ["0000.00001"], "peripheral_ids": [], "analyses": {}, "deep_research_reports": {},
        "editorial_notes": {"0000.00001": "</script>"},
    }
    page = generate_html_report(state, {"status": "completed"})
    assert page.count("</script>") == 2  # 只有模板自带的两个脚本结束标签
    assert "\u2028" not in page
    paper = embedded_papers(page)[0]
    assert paper["abs_url"] is None and paper["editorial_note"] == "</script>"


def test_run_status_counts_and_budget_visibility():
    state = {"date": "2026-01-02", "candidates": [], "core_ids": [], "peripheral_ids": [],
             "analyses": {}, "deep_research_reports": {}}
    for status, text in (("no_candidates", "没有抓取到候选论文"), ("no_relevant_papers", "没有与关注方向相关"),
                         ("already_indexed", "都已收录"), ("budget_exhausted", "剩余额度不足"),
                         ("partial", "只完成了一部分")):
        page = generate_html_report(state, {"status": status})
        assert text in page
        assert "本次没有可展示的论文" in page and 'id="data"' not in page
        assert "预算按三种状态分开记录" not in page

    legacy = {"status": "completed", "pdf_research_completed": 7, "pdf_research_pending": 1,
              "aggregate_budget": {"spent_usd": "1.20", "limit_usd": "5"}}
    page = generate_html_report(state, legacy)
    assert "<dt>完成扩展解读</dt><dd>7</dd>" in page and "<dt>扩展解读未完成</dt><dd>1</dd>" in page
    assert "<dt>已结算</dt><dd>$1.20</dd>" in page
    assert "原生 PDF 输入" not in page and "跳过扩展解读" not in page

    modern = {"status": "partial", "deep_research_completed": 3, "pdf_abstract_count": 2, "deep_research_skipped": False}
    page = generate_html_report(state, modern)
    assert "<dt>完成扩展解读</dt><dd>3</dd>" in page and "仅摘要输入 2 篇" in page
    assert "<dt>跳过扩展解读</dt><dd>否</dd>" in page
    announcement = generate_html_report(state, {**modern, "query_mode": "announced-rss"})
    assert "最新 arXiv 公告" in announcement
    assert "论文按 arXiv 首次提交的 UTC 日期归入" not in announcement


def test_report_index_preserves_empty_days_and_recent_papers(tmp_path):
    from src.report_generator import save_report_index
    reports = tmp_path / 'reports'
    metadata = tmp_path / 'run-metadata'
    reports.mkdir()
    metadata.mkdir()
    for day, summary in [('2026-11-04', {'status':'completed', 'analyzed':108, 'pdf_research_completed':97}),
                         ('2026-11-05', {'status':'completed', 'analyzed':121, 'deep_research_completed':104}),
                         ('2026-11-06', {'status':'no_candidates', 'analyzed':0, 'deep_research_completed':0}),
                         ('2026-11-07', {})]:
        (reports / f'{day}.html').write_text('<html>saved</html>')
        (metadata / f'{day}.json').write_text(json.dumps(summary))
    (reports/'latest.html').write_text('keep existing latest report')
    (reports/'2026-99-99.html').write_text('invalid date')
    (reports/'2026-11-08.html').symlink_to(metadata/'2026-11-05.json')
    page=save_report_index(tmp_path).read_text()
    assert 'href="2026-99-99.html"' not in page and 'href="2026-11-08.html"' not in page
    assert '未抓取到候选论文' in page and '缺少运行记录' in page
    assert '2026-11-05.html' in page and '2026-11-07.html' in page
    assert '97' in page and '104' in page
    assert (reports/'latest.html').read_text()=='keep existing latest report'
    assert 'budget' not in page and 'requests' not in page
