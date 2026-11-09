from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from xml.etree import ElementTree

from jinja2 import Environment, FileSystemLoader
from latex2mathml.converter import convert
from markdown_it import MarkdownIt
from markdown_it.rules_inline.emphasis import tokenize as emphasis_tokenize
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.dollarmath.index import math_inline_dollar

from .config import TEMPLATES_DIR

MISSING = "未记录"

# 方向标识到页面名称的映射, 顺序即过滤按钮的顺序. 新增研究方向时需要同步这里.
DIRECTIONS = [
    ("world_models", "World Models"),
    ("embodied_ai", "VLA 与具身智能"),
    ("autonomous_driving", "自动驾驶"),
    ("agentic", "Agentic AI"),
    ("robotics", "Robotics"),
    ("scene_reconstruction", "3DGS / 三维重建与仿真场景"),
    ("multiple", "多方向"),
]

# 每项为 (摘要字段名列表, 页面名称). 主流程字段在前, 旧验证摘要字段在后作为别名.
SUMMARY_FIELDS = [
    (("candidates",), "候选论文"),
    (("analyzed",), "完成摘要级分析"),
    (("core",), "核心"),
    (("peripheral",), "外围"),
    (("pdf_downloaded",), "已下载 PDF"),
    (("deep_research_completed", "pdf_research_completed"), "完成扩展解读"),
    (("deep_research_pending", "pdf_research_pending"), "扩展解读未完成"),
]
# 只有摘要给出这些字段时才显示, 缺失说明本次运行没有这项统计.
OPTIONAL_SUMMARY_FIELDS = [
    (("deep_research_skipped",), "跳过扩展解读"),
]
INPUT_MODE_FIELDS = [
    ("latex_image_count", "LaTeX 正文与引用图片"),
    ("pdf_page_image_count", "PDF 整页图像输入"),
    ("pdf_native_count", "原生 PDF 输入"),
    ("pdf_text_count", "PDF 文本输入"),
    ("pdf_abstract_count", "仅摘要输入"),
]

BUDGET_FIELDS = [
    ("spent_usd", "已结算"),
    ("uncertain_usd", "未决预留"),
    ("reserved_usd", "请求执行中预留"),
    ("available_usd", "剩余可用"),
    ("limit_usd", "预算上限"),
]

STATUS_TEXT = {
    "completed": "本次运行已经完成。",
    "partial": "本次运行只完成了一部分，页面展示已经保存的结果。",
    "budget_exhausted": "本次运行因剩余额度不足而停止，尚未完成的论文没有继续请求。",
    "no_candidates": "本次没有抓取到候选论文。",
    "no_relevant_papers": "本次候选论文中没有与关注方向相关的论文。",
    "already_indexed": "本次候选论文都已收录在历史索引中，没有新论文需要处理。",
    "running": "本次运行仍在进行，当前页面只展示已经保存的结果。",
    "pdf_incomplete": "部分扩展解读尚未完成。",
}

SAFE_URL = re.compile(r"^https?://", re.IGNORECASE)
HREF = re.compile(r'href="([^"]*)"')
IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
MATHML_TAGS = set("math mrow mi mn mo mtext mspace ms mfrac mroot msqrt msub msup msubsup munder mover munderover mmultiscripts mprescripts none mfenced menclose mstyle mtable mtr mlabeledtr mtd mphantom mpadded semantics annotation".split())
MATHML_ATTRIBUTES = set("display mathvariant stretchy fence separator accent accentunder form lspace rspace columnalign rowalign columnspacing rowspacing displaystyle scriptlevel linebreak open close separators linethickness notation rowspan columnspan voffset width height depth encoding mathcolor mathbackground minsize maxsize".split())
ElementTree.register_namespace("", "http://www.w3.org/1998/Math/MathML")


def _text_or_missing(value: object) -> str:
    if value is None or value == "" or value == "None":
        return MISSING
    return str(value)


def _utc_text(value: object) -> str:
    """把带时区的时间转成 UTC 文本, 无法解析时原样返回."""
    if not value or value == "None":
        return MISSING
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if moment.tzinfo is None:
        return f"{moment:%Y-%m-%d %H:%M:%S}"
    return f"{moment.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S} UTC"


def _safe_url(value: object) -> str | None:
    return value if isinstance(value, str) and SAFE_URL.match(value) else None


def _keep_safe_href(match: re.Match[str]) -> str:
    url = html.unescape(match.group(1))
    if SAFE_URL.match(url):
        return match.group(0) + ' target="_blank" rel="noopener noreferrer"'
    return match.group(0) if url.startswith("#") else 'href="#"'


@lru_cache(maxsize=4096)
def _mathml(latex: str, display: bool) -> str:
    """保留公式的上下标与转义, 只允许没有外部资源的数学标记."""
    # 转换库缺少标准 arg 算子, 且 text 的内容未做 XML 转义.
    # 只兼容等价记法和字面文本, 不修改公式的数学含义.
    latex = re.sub(r"\\arg\b", lambda _: r"\operatorname{arg}", latex)
    latex = re.sub(r"(\\text\{)([^{}]*)(\})", lambda m:
                   m[1] + html.escape(m[2], quote=False) + m[3], latex)
    tree = ElementTree.fromstring(convert(latex, display="block" if display else "inline"))
    for node in tree.iter():
        if node.tag.rsplit("}", 1)[-1] not in MATHML_TAGS:
            raise ValueError("Unsupported MathML element")
        if node.tag.endswith("}mi") and re.match(r"\\[A-Za-z]", node.text or ""):
            raise ValueError("Unsupported LaTeX command")
        if node.tag.endswith("}mtext") and node.text:
            node.text = re.sub(r"\\+([_#$%&{}])", r"\1", node.text)
        for name in list(node.attrib):
            if name not in MATHML_ATTRIBUTES:
                del node.attrib[name]
    return ElementTree.tostring(tree, encoding="unicode")


def _render_math(latex: str, options: dict) -> str:
    try:
        return _mathml(latex, options["display_mode"])
    except Exception:
        # 无法解释的公式保留原始文本, 不自动改写可能影响数学含义的符号.
        return '<code class="math-error">' + html.escape(latex) + '</code>'


def _cjk_emphasis(state, silent: bool) -> bool:
    """沿用标准分隔符配对, 补足中文标点紧邻汉字时的加粗边界."""
    start, first = state.pos, len(state.delimiters)
    matched = emphasis_tokenize(state, silent)
    if not matched or state.src[start] != "*" or state.pos - start < 2:
        return matched
    before = state.src[start - 1] if start else " "
    after = state.src[state.pos] if state.pos < state.posMax else " "
    for delimiter in state.delimiters[first:]:
        if unicodedata.east_asian_width(before) in {"W", "F"} and not after.isspace():
            delimiter.open = True
        if unicodedata.east_asian_width(after) in {"W", "F"} and not before.isspace():
            delimiter.close = True
    return matched


# CommonMark 允许列表紧接段落, 并按列表标记宽度处理嵌套缩进.
# 禁用原始 HTML, 避免先转义整篇文本破坏引用, 代码和公式语法.
MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table").use(
    dollarmath_plugin, allow_labels=False, allow_space=False, allow_digits=False,
    double_inline=True, renderer=_render_math,
)
MARKDOWN.inline.ruler.at("emphasis", _cjk_emphasis)
ADJACENT_OPERATOR = math_inline_dollar(False, True, True)
STRICT_INLINE_MATH = math_inline_dollar(False, False, True)


def _guarded_inline_math(state, silent: bool) -> bool:
    # 价格后紧接中文标点时, 后面的公式起始 $ 不能充当价格的结束符.
    candidate = re.match(r"\$(?!\$)((?:\\.|[^$\\\n])+)\$", state.src[state.pos:])
    if candidate and re.match(r"\d+(?:\.\d+)?\s*[\u3400-\u9fff（，。；]", candidate[1]):
        return False
    return STRICT_INLINE_MATH(state, silent)


def _adjacent_math_operator(state, silent: bool) -> bool:
    # 13$\rightarrow$29 是公式, $13 与 $29 是价格. 仅在有明确 LaTeX
    # 命令时放宽数字邻接限制, 避免把两项价格合成一个数学表达式.
    candidate = re.match(r"\$(?!\$)((?:\\.|[^$\\\n])+)\$", state.src[state.pos:])
    if candidate and re.search(r"\\[A-Za-z]+", candidate[1]):
        return ADJACENT_OPERATOR(state, silent)
    return False


def _render_inline_display(renderer, tokens, index, options, env) -> str:
    # 正文内的 $$ 仍属于段落, div 会让浏览器提前关闭 p 并破坏列表层级.
    formula = _render_math(tokens[index].content.strip(), {"display_mode": True})
    return '<span class="math display">' + formula + '</span>'


MARKDOWN.inline.ruler.at("math_inline", _guarded_inline_math)
MARKDOWN.inline.ruler.before("math_inline", "adjacent_math_operator", _adjacent_math_operator)
MARKDOWN.add_render_rule("math_inline_double", _render_inline_display)


def render_markdown(text: str) -> str:
    """数学 token 先于普通 Markdown 解析, 代码块中的公式源码保持原样."""
    rendered = MARKDOWN.render(text)
    rendered = IMG_TAG.sub("", rendered)
    return HREF.sub(_keep_safe_href, rendered)


def _render_title(text: str) -> str:
    # 标题会放进选择按钮, 不允许嵌套链接或图片等交互元素.
    rendered = IMG_TAG.sub("", MARKDOWN.renderInline(text))
    return re.sub(r"</?a\b[^>]*>", "", rendered)


def _display_number(value: object) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _first_present(summary: dict, keys: tuple[str, ...]) -> object:
    return next((summary[key] for key in keys if summary.get(key) is not None), None)


def _build_papers(state: dict, notes: dict) -> list[dict]:
    core = set(state.get("core_ids", []))
    peripheral = set(state.get("peripheral_ids", []))
    analyses = state.get("analyses", {})
    reports = state.get("deep_research_reports", {})
    papers = []
    for item in state.get("candidates", []):
        pid = item["arxiv_id"]
        if pid not in core and pid not in peripheral:
            continue
        analysis = analyses.get(pid)
        report = reports.get(pid)
        abstract = _text_or_missing(item.get("abstract"))
        papers.append({
            "id": pid,
            "version": _text_or_missing(item.get("version")),
            "title": item.get("title", MISSING),
            "title_html": _render_title(item.get("title", MISSING)),
            "authors": item.get("authors") or [],
            "categories": item.get("categories") or [],
            "primary_category": _text_or_missing(item.get("primary_category")),
            "published": _utc_text(item.get("published")),
            "updated": _utc_text(item.get("updated")),
            "abs_url": _safe_url(item.get("abs_url")),
            "pdf_url": _safe_url(item.get("pdf_url")),
            "comment": _text_or_missing(item.get("comment")),
            "abstract_html": render_markdown(abstract),
            "direction": analysis.get("direction") if analysis else item.get("matched_direction"),
            "level": "核心" if pid in core else "外围",
            "analysis": {
                "one_line": _text_or_missing(analysis.get("one_line_summary")),
                "detail_html": render_markdown(_text_or_missing(analysis.get("detailed_summary"))),
                "terms": analysis.get("key_terms") or [],
                "related": _text_or_missing(analysis.get("related_work_context")),
                "affiliations": analysis.get("affiliation_names") or [],
                "has_code": analysis.get("has_code"),
                "code_url": _safe_url(analysis.get("code_url")),
                "scores": [
                    (label, analysis.get(key))
                    for key, label in (
                        ("novelty_score", "新颖性"),
                        ("impact_score", "影响力"),
                        ("reproducibility_score", "可复现性"),
                        ("focus_relevance_score", "与关注方向相关度"),
                        ("weighted_score", "综合评分"),
                    )
                ],
            } if analysis else None,
            "weighted": analysis.get("weighted_score") if analysis else None,
            "report_html": render_markdown(report) if report else None,
            "editorial_note": notes.get(pid),
        })
    # 核心在前, 其次按综合评分从高到低, 没有评分的排在最后, 编号作为稳定的最终排序.
    papers.sort(key=lambda p: (p["level"] != "核心", p["weighted"] is None,
                               -(p["weighted"] or 0), p["id"]))
    return papers


def _embed_json(data: dict) -> str:
    """序列化为可以放进 script 标签的 JSON, 转义 < 避免出现 </script 与注释起始."""
    text = json.dumps(data, ensure_ascii=False)
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _input_modes(summary: dict) -> str:
    """只列出摘要实际给出的输入模式计数, 不推算缺失项."""
    parts = [f"{label} {summary[key]} 篇" for key, label in INPUT_MODE_FIELDS if summary.get(key) is not None]
    if not parts:
        return ""
    text_note = "PDF 文本输入没有直接包含图像像素。" if summary.get("pdf_text_count") else ""
    return "，".join(parts) + "。" + text_note + "模型输出仍须对照原文阅读。"


def _budget_rows(summary: dict) -> list[tuple[str, str]]:
    budget = summary.get("budget") or summary.get("aggregate_budget") or {}
    if all(budget.get(key) is None for key, _ in BUDGET_FIELDS):
        return []
    return [(label, MISSING if budget.get(key) is None else f"${budget[key]}") for key, label in BUDGET_FIELDS]


def generate_html_report(state: dict, summary: dict) -> str:
    """把检查点结构与运行摘要渲染为可离线打开的单页 HTML 文本.

    state 含 date, candidates, core_ids, peripheral_ids, analyses,
    deep_research_reports, 可选 editorial_notes. 函数不读写文件也不访问网络.
    """
    date = _text_or_missing(state.get("date"))
    papers = _build_papers(state, state.get("editorial_notes") or {})
    fields = SUMMARY_FIELDS + [f for f in OPTIONAL_SUMMARY_FIELDS if _first_present(summary, f[0]) is not None]
    stats = [(label, _display_number(_first_present(summary, keys))) for keys, label in fields]
    data = {"papers": papers, "directions": DIRECTIONS, "missing": MISSING}
    # data_json 已转义 < 与 U+2028, U+2029, 只有它可以用 safe 过滤器插入模板.
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    return env.get_template("daily_report.html.j2").render(
        date=date,
        favicon_url=os.environ.get("PAPERS_FAVICON_URL", ""),
        query_mode=summary.get("query_mode", "submitted-date-utc"),
        status_text=STATUS_TEXT.get(summary.get("status"), "运行状态未记录。"),
        input_modes=_input_modes(summary),
        stats=stats,
        budget=_budget_rows(summary),
        has_papers=bool(papers),
        data_json=_embed_json(data),
    )
