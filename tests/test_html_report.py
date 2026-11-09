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


def test_code_and_untrusted_content_are_inert(monkeypatch):
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
    links = render_markdown('[原文](https://example.test/paper) [章节](#method)')
    assert 'href="https://example.test/paper" target="_blank" rel="noopener noreferrer"' in links
    assert 'href="#method" target=' not in links

    state = {
        "date": "2026-01-02",
        "candidates": [{"arxiv_id": "0000.00001", "title": "</script><b>x</b>", "abstract": "a\u2028b",
                        "abs_url": "javascript:alert(1)", "pdf_url": "https://example.test/p.pdf"}],
        "core_ids": ["0000.00001"], "peripheral_ids": [], "analyses": {}, "deep_research_reports": {},
        "editorial_notes": {"0000.00001": "</script>"},
    }
    monkeypatch.setenv("PAPERS_FAVICON_URL", "icons/paper.gif?v=1")
    page = generate_html_report(state, {"status": "completed"})
    assert '<link rel="icon" href="icons/paper.gif?v=1">' in page
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

    def write_days(root, days):
        reports, metadata = root / 'reports', root / 'run-metadata'
        reports.mkdir(parents=True)
        metadata.mkdir()
        for day, summary in days.items():
            (reports / f'{day}.html').write_text(f'<html>report {day}</html>')
            (metadata / f'{day}.json').write_text(json.dumps(summary))
        return reports, metadata

    reports, metadata = write_days(tmp_path, {
        '2026-10-18': {'status': 'budget_exhausted', 'analyzed': 0},   # Sunday, abnormal
        '2026-10-24': {},                                              # Saturday, no run record
        '2026-10-25': {'status': 'completed', 'analyzed': 3},          # Sunday with papers
        '2026-10-31': {'status': 'partial', 'analyzed': 0},            # Saturday, abnormal
        '2026-11-01': {'status': 'running', 'analyzed': 0},            # Sunday, abnormal
        '2026-11-04': {'status': 'completed', 'analyzed': 108, 'pdf_research_completed': 97},
        '2026-11-05': {'status': 'completed', 'analyzed': 121, 'deep_research_completed': 104},
        '2026-11-06': {'status': 'no_candidates', 'analyzed': 0, 'deep_research_completed': 0},  # weekday
        '2026-11-07': {'status': 'no_candidates', 'analyzed': 0},      # Saturday, confirmed empty
        '2026-11-08': {'status': 'already_indexed', 'analyzed': 0},    # Sunday, confirmed empty
    })
    (reports / 'latest.html').write_text('stale latest report')
    (reports / '2026-99-99.html').write_text('invalid date')
    (reports / '2026-11-14.html').symlink_to(metadata / '2026-11-05.json')
    page = save_report_index(tmp_path).read_text()
    for hidden in ('2026-99-99', '2026-11-14', '2026-11-07', '2026-11-08'):
        assert f'href="{hidden}.html"' not in page
    for shown in ('2026-10-18', '2026-10-24', '2026-10-25', '2026-10-31', '2026-11-01', '2026-11-04', '2026-11-06'):
        assert f'href="{shown}.html"' in page
    assert '未抓取到候选论文' in page and '缺少运行记录' in page
    assert '97' in page and '104' in page
    assert 'budget' not in page and 'requests' not in page
    # The main entry is the newest report with papers; the newest visible date is the secondary entry.
    assert 'class="entry" href="2026-11-05.html"' in page
    assert 'class="entry-secondary" href="2026-11-06.html"' in page
    assert (reports / '2026-11-08.html').read_text() == '<html>report 2026-11-08</html>'
    assert (reports / 'latest.html').read_text() == '<html>report 2026-11-06</html>'

    empty_root = tmp_path / 'empty'
    reports, _ = write_days(empty_root, {'2026-11-07': {'status': 'no_relevant_papers', 'analyzed': 0}})
    index = save_report_index(empty_root).read_text()
    assert 'href="2026-11-07.html"' not in index
    assert (reports / 'latest.html').read_text() == index

def test_web_reader_publishes_immutable_details_before_index(monkeypatch, tmp_path):
    from src.html_report import generate_web_report
    from src.report_generator import save_reading_report
    state = {"date": "2026-11-09", "candidates": [{"arxiv_id": "2611.12345", "title": "Paper",
             "abstract": "Abstract", "authors": ["Author"], "matched_direction": "robotics"}],
             "core_ids": ["2611.12345"], "analyses": {"2611.12345": {"direction": "robotics",
             "one_line_summary": "searchable summary", "key_terms": ["robot"]}},
             "deep_research_reports": {"2611.12345": "# Method\nFull analysis $x^2$."}}
    summary = {"status": "completed"}
    standalone = embedded_papers(generate_html_report(state, summary))
    page, assets = generate_web_report(state, summary)
    index = embedded_papers(page)
    assert "report_html" not in index[0] and "abstract_html" not in index[0]
    assert index[0]["analysis"] == {"terms": ["robot"], "one_line": "searchable summary"}
    assert json.loads(assets[index[0]["detail_url"]]) == standalone[0]
    monkeypatch.setenv("PAPERS_WEB_REPORTS", "1")
    path = save_reading_report(state, summary, tmp_path)
    old_index = path.read_text()
    payload_path = tmp_path / "reports" / index[0]["detail_url"]
    assert json.loads(payload_path.read_text()) == standalone[0]
    payload_path.write_text("damaged")
    import pytest
    with pytest.raises(ValueError, match="damaged"):
        save_reading_report(state, summary, tmp_path)
    assert path.read_text() == old_index
    payload_path.write_text(assets[index[0]["detail_url"]])
    state["deep_research_reports"]["2611.12345"] = "# Updated analysis"
    new_path = save_reading_report(state, summary, tmp_path)
    new_url = embedded_papers(new_path.read_text())[0]["detail_url"]
    assert new_url != index[0]["detail_url"] and payload_path.is_file()
