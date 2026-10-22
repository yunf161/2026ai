"""
报告生成/导出工具

将 ExperimentReport 导出为 PDF 或 Markdown 文件。
支持中文渲染。
"""
import os
import re
import glob
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool


# 默认中文字体路径
_DEFAULT_CJK_FONTS = [
    "C:/Windows/Fonts/simhei.ttf",    # Windows 黑体（TTF，兼容性更好）
    "C:/Windows/Fonts/msyh.ttf",      # Windows 微软雅黑（TTF）
    "C:/Windows/Fonts/simsun.ttc",    # Windows 宋体
    "C:/Windows/Fonts/msyh.ttc",      # Windows 微软雅黑
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux 文泉驿
    "/System/Library/Fonts/PingFang.ttc",  # macOS 苹方
]


def _normalize_docx_styles(docx_path: str) -> bool:
    """统一 DOCX 字体、字号、粗细和正文段落样式。"""
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.oxml.ns import qn
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.style import WD_STYLE_TYPE
    except Exception:
        return False

    try:
        doc = Document(docx_path)
    except Exception:
        return False

    def _set_run_font(run, name: str, size_pt: int, bold: bool = False):
        run.font.name = name
        run.font.size = Pt(size_pt)
        run.font.bold = bold
        try:
            run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
        except Exception:
            pass

    def _style_paragraph_body(p):
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        fmt = p.paragraph_format
        fmt.first_line_indent = Pt(24)  # 约等于中文首行两个字符
        fmt.line_spacing = 1.5
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(0)
        for run in p.runs:
            _set_run_font(run, "SimSun", 12, bold=False)

    def _style_paragraph_heading(p, level: int):
        size_map = {1: 16, 2: 14, 3: 13}
        size = size_map.get(level, 12)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt = p.paragraph_format
        fmt.first_line_indent = Pt(0)
        fmt.line_spacing = 1.3
        fmt.space_before = Pt(6)
        fmt.space_after = Pt(3)
        for run in p.runs:
            _set_run_font(run, "SimSun", size, bold=True)

    def _style_paragraph_code(p):
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt = p.paragraph_format
        fmt.first_line_indent = Pt(0)
        fmt.left_indent = Pt(18)
        fmt.right_indent = Pt(6)
        fmt.line_spacing = 1.2
        fmt.space_before = Pt(2)
        fmt.space_after = Pt(2)
        for run in p.runs:
            _set_run_font(run, "Consolas", 10, bold=False)

    def _style_paragraph_formula(p):
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt = p.paragraph_format
        fmt.first_line_indent = Pt(0)
        fmt.left_indent = Pt(12)
        fmt.right_indent = Pt(0)
        fmt.line_spacing = 1.3
        fmt.space_before = Pt(2)
        fmt.space_after = Pt(2)
        for run in p.runs:
            _set_run_font(run, "Cambria Math", 12, bold=False)

    def _style_paragraph_list(p):
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        fmt = p.paragraph_format
        fmt.first_line_indent = Pt(0)
        fmt.left_indent = Pt(18)
        fmt.line_spacing = 1.4
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(0)
        for run in p.runs:
            _set_run_font(run, "SimSun", 12, bold=False)

    # 全局样式兜底
    try:
        normal = doc.styles["Normal"]
        normal.font.name = "SimSun"
        normal.font.size = Pt(12)
        normal.font.bold = False
        normal._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    except Exception:
        pass

    # 标题样式兜底
    for h_name, h_level in (("Heading 1", 1), ("Heading 2", 2), ("Heading 3", 3)):
        try:
            h = doc.styles[h_name]
            h.font.name = "SimSun"
            h.font.size = Pt({1: 16, 2: 14, 3: 13}[h_level])
            h.font.bold = True
            h._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
        except Exception:
            pass

    # 段落逐个统一
    for p in doc.paragraphs:
        style_name = (p.style.name or "") if p.style is not None else ""
        if style_name == "CodeBlock":
            _style_paragraph_code(p)
        elif style_name == "FormulaBlock":
            _style_paragraph_formula(p)
        elif style_name == "List Paragraph":
            _style_paragraph_list(p)
        elif style_name.startswith("Heading"):
            level = 1
            m = re.search(r"Heading\s*(\d+)", style_name)
            if m:
                level = int(m.group(1))
            _style_paragraph_heading(p, level)
        else:
            _style_paragraph_body(p)

    # 表格单元格内段落统一正文样式
    for table in doc.tables:
        for r_idx, row in enumerate(table.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    if r_idx == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for run in p.runs:
                            _set_run_font(run, "SimSun", 13, bold=True)
                    else:
                        _style_paragraph_body(p)

    try:
        doc.save(docx_path)
        return True
    except Exception:
        return False


def _build_basic_docx(report_dict: dict, output_path: str) -> bool:
    """使用 python-docx 构建高可读 DOCX（清洗 Markdown、样式化代码、嵌入图片、规范化公式）。"""
    try:
        from docx import Document
        from docx.shared import Inches
        from docx.enum.style import WD_STYLE_TYPE
        from docx.oxml import parse_xml
        from docx.oxml.ns import nsdecls
    except Exception:
        return False

    meta = report_dict.get("meta", {})
    sections = report_dict.get("sections", [])
    figures = report_dict.get("figures", [])
    attachments = report_dict.get("attachments", [])
    image_analysis_raw = report_dict.get("image_analysis", "")

    def _canonical_path(path: str) -> str:
        p = str(path or "").strip().strip('"').strip("'")
        p = p.rstrip("，。；;：:！？!?、）)]】}>\n\r\t ")
        p = p.replace("\\", "/")
        return os.path.normcase(os.path.abspath(p)) if p else ""

    search_roots = []
    for x in (attachments or []) + (figures or []):
        nx = str(x or "").strip().replace("\\", "/")
        if nx:
            search_roots.append(os.path.dirname(nx))
    search_roots = [r for r in dict.fromkeys(search_roots) if r and os.path.isdir(r)]

    def _recover_image_path(path: str) -> str:
        p = _normalize_path(path)
        if p and os.path.exists(p):
            return p
        base = os.path.basename(p)
        if not base:
            return p
        for root in search_roots:
            try:
                matches = glob.glob(os.path.join(root, "**", base), recursive=True)
                for m in matches:
                    if os.path.exists(m):
                        return m.replace("\\", "/")
            except Exception:
                continue
        return p

    def _parse_json_payload(raw: str) -> dict:
        import json
        if isinstance(raw, dict):
            return raw
        s = str(raw or "").strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            start = s.find("{")
            end = s.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(s[start:end + 1])
                except Exception:
                    return {}
        return {}

    def _is_invalid_image_summary(text: str) -> bool:
        s = str(text or "")
        bad_tokens = ["文件未找到", "无法读取图片", "无法解析内容", "路径有误", "被移动", "被删除"]
        return any(t in s for t in bad_tokens)

    image_suggestions = {}
    allowed_figure_keys = set()
    if image_analysis_raw:
        try:
            obj = _parse_json_payload(image_analysis_raw)
            for item in obj.get("images", []) if isinstance(obj, dict) else []:
                p = str(item.get("path", "")).replace("\\", "/").strip()
                if p:
                    key = _canonical_path(p)
                    image_suggestions[key] = item
                    if item.get("should_embed", True) is not False:
                        allowed_figure_keys.add(key)
        except Exception:
            image_suggestions = {}

    def _figure_allowed(path: str) -> bool:
        """只有 image_analysis 允许或未提供分析时才允许插图。"""
        p = _recover_image_path(path)
        key = _canonical_path(p)
        sug = image_suggestions.get(key, {})
        if isinstance(sug, dict) and sug.get("should_embed") is False:
            return False
        if image_suggestions and key not in allowed_figure_keys:
            return False
        return True

    doc = Document()

    # 自定义段落样式（若不存在则创建）
    style_names = [s.name for s in doc.styles]
    if "CodeBlock" not in style_names:
        doc.styles.add_style("CodeBlock", WD_STYLE_TYPE.PARAGRAPH)
    if "FormulaBlock" not in style_names:
        doc.styles.add_style("FormulaBlock", WD_STYLE_TYPE.PARAGRAPH)

    def _add_shading(paragraph, fill: str = "F3F3F3"):
        try:
            p_pr = paragraph._p.get_or_add_pPr()
            shd = parse_xml(rf'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill}"/>')
            p_pr.append(shd)
        except Exception:
            pass

    def _clean_md_line(text: str) -> str:
        s = _clean_inline_markdown(str(text or "").rstrip())
        s = re.sub(r"^>\s*", "", s)
        s = s.replace("`", "")
        return _normalize_noisy_text(s)

    def _extract_md_image(line: str) -> Optional[str]:
        m = re.search(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
        return None

    def _normalize_path(path: str) -> str:
        p = str(path or "").strip().strip('"').strip("'")
        p = p.rstrip("，。；;：:！？!?、）)]】}>\n\r\t ")
        return p.replace("\\", "/")

    image_analysis_cache = {}

    def _summarize_image_text(raw: str) -> str:
        s = str(raw or "").strip()
        if not s:
            return "已检测到图像，等待视觉模型分析。"
        lines = []
        for ln in s.splitlines():
            t = ln.strip()
            if not t:
                continue
            if t.startswith("[Error") or t.startswith("[图片") or t.startswith("[DOCX"):
                continue
            lines.append(t)
            if len(lines) >= 3:
                break
        text = "；".join(lines) if lines else "已检测到图像，等待视觉模型分析。"
        if len(text) > 120:
            text = text[:120] + "..."
        return text

    def _analyze_image(path: str) -> str:
        p = _normalize_path(path)
        if not p:
            return "已检测到图像，等待视觉模型分析。"
        if p in image_analysis_cache:
            return image_analysis_cache[p]
        try:
            from .file_parser import parse_file
            raw = parse_file.func(p)
            summary = _summarize_image_text(raw)
        except Exception:
            summary = "已检测到图像，等待视觉模型分析。"
        image_analysis_cache[p] = summary
        return summary

    def _insert_picture_if_exists(path: str) -> bool:
        p = _recover_image_path(path)
        if not p or not os.path.exists(p):
            return False
        if not _figure_allowed(p):
            return False
        try:
            doc.add_picture(p, width=Inches(5.8))
            cap = doc.add_paragraph(f"图: {os.path.basename(p)}")
            cap.style = "List Paragraph"
            suggestion = image_suggestions.get(_canonical_path(p), {})
            summary = suggestion.get("summary") if isinstance(suggestion, dict) else ""
            placement = suggestion.get("placement_hint") if isinstance(suggestion, dict) else ""
            trend = suggestion.get("trend_analysis") if isinstance(suggestion, dict) else ""
            key_points = suggestion.get("key_points") if isinstance(suggestion, dict) else []
            if (not summary) or _is_invalid_image_summary(summary):
                summary = _analyze_image(p)
            desc_text = f"图像解读: {summary}"
            if trend:
                desc_text += f"；趋势分析: {trend}"
            if key_points:
                desc_text += f"；关键节点: {key_points}"
            if placement and ("无法读取" not in str(placement)):
                desc_text += f"；插图位置说明: {placement}"
            desc = doc.add_paragraph(desc_text)
            desc.style = "List Paragraph"
            return True
        except Exception:
            return False

    used_figures = set()
    pending_figures = []
    for fig in figures:
        nf = _normalize_path(fig)
        if not nf:
            continue
        rf = _recover_image_path(nf)
        sug = image_suggestions.get(_canonical_path(rf), {}) if rf else {}
        should_embed = sug.get("should_embed", True) if isinstance(sug, dict) else True
        if rf and os.path.exists(rf) and should_embed is not False and _figure_allowed(rf):
            pending_figures.append(rf)

    def _try_insert_context_figure(anchor_text: str, current_heading: str = "", lines_since_heading: int = 0) -> bool:
        if not pending_figures:
            return False
        anchor = str(anchor_text or "").lower()
        heading = str(current_heading or "").lower()
        trigger_words = ["结果", "曲线", "奖励", "训练", "loss", "reward", "回报", "收敛", "评测", "实验现象"]
        matched = any(w in anchor for w in trigger_words)

        # 优先使用 image_analysis 的推荐锚词
        if not matched and image_suggestions:
            for p in pending_figures:
                sug = image_suggestions.get(_canonical_path(p), {})
                kws = sug.get("keywords", []) if isinstance(sug, dict) else []
                if any(str(k).lower() in anchor for k in kws):
                    matched = True
                    break

        if not matched:
            return False

        # 即使在结果章节，也要求至少出现少量正文后再插图
        if lines_since_heading < 2:
            return False

        for p in list(pending_figures):
            if p in used_figures:
                pending_figures.remove(p)
                continue

            # 若建议了推荐章节，则优先在推荐章节命中后再插入
            sug = image_suggestions.get(_canonical_path(p), {})
            rec = str(sug.get("recommended_section", "")).strip().lower() if isinstance(sug, dict) else ""
            if rec and (rec not in heading) and (rec not in anchor):
                continue

            if _insert_picture_if_exists(p):
                used_figures.add(p)
                pending_figures.remove(p)
                return True
        return False
    doc.add_heading(meta.get("title", "实验报告"), level=0)
    doc.add_paragraph(f"课程: {meta.get('course_name', '人工智能')}")
    doc.add_paragraph(f"姓名: {meta.get('student_name', '')}    学号: {meta.get('student_id', '')}")
    doc.add_paragraph(f"日期: {meta.get('date', datetime.now().strftime('%Y-%m-%d'))}")

    for section in sections:
        doc.add_heading(section.get("title", "无标题"), level=1)
        content = _fix_dqn_formula_fragments(_stabilize_math_text(section.get("content", "")))
        in_code = False
        current_md_heading = ""
        lines_since_heading = 0
        for raw_line in content.split("\n"):
            line = str(raw_line or "")
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code = not in_code
                continue

            if in_code:
                p = doc.add_paragraph(line.replace("\t", "    "), style="CodeBlock")
                _add_shading(p, "F5F5F5")
                continue

            md_img = _extract_md_image(line)
            if md_img and _insert_picture_if_exists(md_img):
                used_figures.add(_normalize_path(md_img))
                continue

            if _insert_picture_if_exists(line):
                used_figures.add(_normalize_path(line))
                continue

            if not stripped:
                doc.add_paragraph("")
                continue

            heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading:
                level = min(len(heading.group(1)), 3)
                title = _clean_md_line(heading.group(2))
                doc.add_heading(title, level=level)
                current_md_heading = title
                lines_since_heading = 0
                _try_insert_context_figure(title, current_md_heading, lines_since_heading)
                continue

            if _looks_like_formula(line) or stripped.startswith("$$") or stripped.endswith("$$"):
                formula_text = stripped.strip("$").strip()
                formula_text = _latex_to_readable(formula_text)
                p = doc.add_paragraph(formula_text, style="FormulaBlock")
                _add_shading(p, "FAFAFA")
                continue

            m_ul = re.match(r"^[-*+]\s+(.*)$", stripped)
            if m_ul:
                text = _clean_md_line(m_ul.group(1))
                doc.add_paragraph(text, style="List Paragraph")
                lines_since_heading += 1
                _try_insert_context_figure(text, current_md_heading, lines_since_heading)
                continue

            m_ol = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            if m_ol:
                text = _clean_md_line(m_ol.group(2))
                doc.add_paragraph(f"{m_ol.group(1)}. {text}", style="List Paragraph")
                lines_since_heading += 1
                _try_insert_context_figure(text, current_md_heading, lines_since_heading)
                continue

            body_text = _clean_md_line(line)
            # 行内数学表达式统一转可读文本，降低 DOCX/PDF 导出中的乱码风险
            body_text = re.sub(r"\$\$([^$]+)\$\$", lambda m: _latex_to_readable(m.group(1)), body_text)
            body_text = re.sub(r"\$([^$]+)\$", lambda m: _latex_to_readable(m.group(1)), body_text)
            body_text = re.sub(r"\\\(([^)]+)\\\)", lambda m: _latex_to_readable(m.group(1)), body_text)
            # 兜底：未加数学定界符但包含明显 LaTeX 片段的整行公式文本
            if (re.search(r"\\[A-Za-z]+", body_text)
                    and re.search(r"[=^_]", body_text)
                    and not re.search(r"[\u4e00-\u9fff]{4,}", body_text)):
                body_text = _latex_to_readable(body_text)
            doc.add_paragraph(body_text)
            lines_since_heading += 1
            _try_insert_context_figure(body_text, current_md_heading, lines_since_heading)

    if figures:
        added_figure_heading = False
        for fig in figures:
            norm_fig = _normalize_path(fig)
            if norm_fig in used_figures:
                continue
            if not _figure_allowed(norm_fig):
                continue
            if not added_figure_heading:
                doc.add_heading("实验结果图片", level=1)
                added_figure_heading = True
            if not _insert_picture_if_exists(norm_fig):
                doc.add_paragraph(str(fig), style="List Paragraph")

    if attachments:
        doc.add_heading("附件清单", level=1)
        for att in attachments:
            doc.add_paragraph(str(att))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.save(output_path)
    return True


def _convert_docx_to_pdf(docx_path: str, pdf_path: str) -> bool:
    """将 DOCX 转 PDF（优先 docx2pdf，其次 Windows COM）。"""
    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)

    # 方案1: docx2pdf
    try:
        from docx2pdf import convert as docx2pdf_convert
        docx2pdf_convert(abs_docx, abs_pdf)
        if os.path.exists(abs_pdf):
            return True
    except Exception:
        pass

    # 方案2: Windows Word COM
    if os.name == "nt":
        try:
            import win32com.client as win32
            word = win32.Dispatch("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            doc = word.Documents.Open(abs_docx)
            # 17 = wdFormatPDF
            doc.SaveAs(abs_pdf, FileFormat=17)
            doc.Close(False)
            word.Quit()
            if os.path.exists(abs_pdf):
                return True
        except Exception:
            try:
                word.Quit()
            except Exception:
                pass

    return False


def _try_generate_pdf_via_docx(report_dict: dict, output_path: str) -> Optional[str]:
    """优先走 DOCX 管线生成 PDF，成功返回结果字符串，失败返回 None。"""
    base = output_path.rsplit(".", 1)[0]
    docx_path = base + ".docx"

    # 直接构建可控 DOCX，保证代码块/图片/公式/样式一致
    docx_ok = _build_basic_docx(report_dict, docx_path)

    if not docx_ok:
        return None

    # 4) 无论来源，统一规范化 DOCX 样式（字体/字号/段落）
    _normalize_docx_styles(docx_path)

    # 5) docx -> pdf
    pdf_ok = _convert_docx_to_pdf(docx_path, output_path)

    if pdf_ok:
        return f"[PDF] 报告已生成(DOCX管线): {output_path}"
    return None


def _find_cjk_font() -> Optional[str]:
    """查找可用的中文字体"""
    try:
        from config import config
        configured = (config.pdf_font_path or "").strip() if getattr(config, "pdf_font_path", None) is not None else ""
        if configured and os.path.exists(configured):
            return configured
    except Exception:
        pass

    for font_path in _DEFAULT_CJK_FONTS:
        if os.path.exists(font_path):
            return font_path
    return None


def _safe_multicell(pdf, text: str, line_h: int = 7):
    """安全写入 PDF 文本，避免超长不可断行字符串导致渲染异常。"""
    # 保证从左边距开始写，避免在极窄剩余宽度下触发渲染异常
    pdf.set_x(pdf.l_margin)

    s = str(text or "").replace("\r\n", "\n")
    # 优先保留原始换行，再对无空格长串做软切分
    max_chunk = 100
    out_lines = []
    for raw_line in s.split("\n"):
        line = raw_line
        if len(line) > max_chunk and (" " not in line):
            line = "\n".join(line[i:i + max_chunk] for i in range(0, len(line), max_chunk))
        out_lines.extend(line.split("\n"))

    for line in out_lines:
        # 空行用小间距，避免累计到异常位置
        if not line:
            pdf.ln(max(1, int(line_h / 2)))
            continue
        pdf.set_x(pdf.l_margin)
        # 先使用默认换行；若失败再退回字符级换行
        try:
            pdf.multi_cell(0, line_h, line, align="L")
        except Exception:
            pdf.multi_cell(0, line_h, line, wrapmode="CHAR", align="L")


def _safe_center_line(pdf, text: str, line_h: int = 10):
    """安全写入居中单行文本。"""
    s = str(text or "")
    try:
        pdf.multi_cell(0, line_h, s, align="C")
    except Exception:
        # 避免某些 fpdf2 + 字体组合下的 wrapmode/align 异常
        _safe_multicell(pdf, s, line_h=line_h)


def _embed_image_or_fallback(pdf, image_path: str, line_h: int = 7):
    """优先嵌入图片，失败时回退为路径文本。"""
    path = str(image_path or "").strip()
    if not path:
        return

    norm = path.replace("\\", "/")
    if not os.path.exists(norm):
        _safe_multicell(pdf, f"[图片不存在] {norm}", line_h=line_h)
        return

    try:
        max_w = pdf.w - pdf.l_margin - pdf.r_margin
        # 让 fpdf2 自动按宽度等比缩放，避免图片超出页面
        pdf.image(norm, x=pdf.l_margin, w=max_w)
        pdf.ln(2)
    except Exception as e:
        _safe_multicell(pdf, f"[图片嵌入失败] {norm}; 错误: {e}", line_h=line_h)


def _render_formula_text_block(pdf, formula_text: str, line_h: int = 7):
    """将公式渲染为稳定的文本块样式（不依赖 LaTeX/PNG）。"""
    raw = str(formula_text or "").strip()
    if not raw:
        return

    normalized = _latex_to_readable(raw)
    pdf.set_fill_color(248, 248, 248)
    pdf.set_text_color(20, 20, 20)
    pdf.set_x(pdf.l_margin + 1)
    pdf.multi_cell(0, line_h, f"  {normalized}", fill=True, align="L")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def _normalize_markdown_line(line: str) -> str:
    """轻量清洗：仅去除少量视觉噪声，不破坏代码/公式符号。"""
    text = str(line or "").rstrip("\n")
    # 仅清除末尾多余空白，保留前导缩进（代码块需要）
    return text.rstrip()


def _clean_inline_markdown(text: str) -> str:
    """清洗行内 Markdown 展示符号，避免正文出现多余 * _ ~。"""
    s = str(text or "")
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"~~([^~]+)~~", r"\1", s)
    return s


def _normalize_noisy_text(text: str) -> str:
    """清洗常见噪声文本：参数名断裂、科学计数法空格、连字符异常。"""
    s = str(text or "")
    # 参数名噪声: hidden_(size) -> hidden_size
    s = re.sub(r"\b([A-Za-z]+)_\(([A-Za-z0-9]+)\)", r"\1_\2", s)
    # 科学计数法噪声: 1e - 3 -> 1e-3
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*[eE]\s*([+-])\s*(\d+)\b", r"\1e\2\3", s)
    # 常见连字符噪声: CartPole - v1 -> CartPole-v1
    s = re.sub(r"\b([A-Za-z0-9]+)\s*-\s*([A-Za-z]\d+)\b", r"\1-\2", s)
    # 压缩多余空白
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _format_paragraph_for_pdf(text: str) -> str:
    """格式化普通段落：清理样式并添加首行缩进。"""
    s = _clean_inline_markdown(str(text or "").strip())
    if not s:
        return s
    # 普通正文统一首行缩进（两个中文全角空格）
    if not s.startswith("  ") and not s.startswith("\u3000\u3000"):
        s = "\u3000\u3000" + s
    return s


def _stabilize_math_text(text: str) -> str:
    """稳定化公式文本：修复未闭合定界符并清理易碎 LaTeX 残片。"""
    raw = str(text or "")
    if not raw:
        return raw

    lines_in = raw.split("\n")
    out_lines = []
    i = 0
    while i < len(lines_in):
        line = lines_in[i]
        s = str(line)
        stripped = s.strip()

        # 孤立公式定界符直接移除，避免导出后出现碎片
        if stripped in {"$", "$$"}:
            out_lines.append("")
            i += 1
            continue

        # 未闭合 $：去定界符后按可读公式归一化
        dollar_count = len(re.findall(r"(?<!\\)\$", s))
        if dollar_count % 2 == 1:
            core = re.sub(r"(?<!\\)\$", "", s)
            if re.search(r"\\[A-Za-z]+|[=_^]", core):
                s = _latex_to_readable(core)
            else:
                s = core

        # 残留 LaTeX 碎片兜底
        if re.search(r"\\(left|right|big|Big|theta|alpha|beta|gamma|epsilon)", s) and re.search(r"[=_^\\]", s):
            s = _latex_to_readable(s)

        # 统一处理 \(...\) 与 \[...\]
        s = re.sub(r"\\\(([^)]+)\\\)", lambda m: _latex_to_readable(m.group(1)), s)
        s = re.sub(r"\\\[([^\]]+)\\\]", lambda m: _latex_to_readable(m.group(1)), s)

        # 跨行公式拼接：若当前行以运算符或未闭合括号结尾，且下一行看起来是公式续行，则拼接
        if i + 1 < len(lines_in):
            nxt = str(lines_in[i + 1] or "").strip()
            dangling = bool(re.search(r"[=+\-*/^_(]$", s.strip()))
            unbalanced = s.count("(") > s.count(")")
            next_formula_like = bool(re.search(r"\\[A-Za-z]+|[=^_]|\b(max|min|argmax|sum|sqrt|log|exp|Q\(|L\(|y_t)\b", nxt, re.IGNORECASE))
            if (dangling or unbalanced) and nxt and next_formula_like:
                s = f"{s} {nxt}"
                i += 1

        out_lines.append(s)
        i += 1

    # 再做一轮未闭合定界符收敛，避免拼接后引入新残片
    normalized = []
    for s in out_lines:
        c = len(re.findall(r"(?<!\\)\$", s))
        if c % 2 == 1:
            core = re.sub(r"(?<!\\)\$", "", s)
            s = _latex_to_readable(core) if re.search(r"\\[A-Za-z]+|[=_^]", core) else core
        normalized.append(s)

    return "\n".join(normalized)


def _fix_dqn_formula_fragments(text: str) -> str:
    """将 DQN 常见残缺公式替换为固定可读模板。"""
    s = str(text or "")
    q_update = "Q更新: Q(s,a) <- Q(s,a)+α*(r+γ*max_a'Q(s',a')-Q(s,a))"
    dqn_loss = "DQN损失: L(θ)=E[(y-Q(s,a;θ))^2]"
    ddqn_target = "Double DQN目标: y=r+γ*Q_target(s',argmax_a Q_online(s',a;θ);θ_target)"

    # 截断的 Q 更新式：例如以 Q( 结尾或括号明显不平衡
    s = re.sub(
        r"Q\(s,\s*a\)\s*<-\s*Q\(s,\s*a\).*?(?:Q\(|$)",
        q_update,
        s,
        flags=re.IGNORECASE
    )
    # 残缺 max_{a'} / max_a 片段
    s = re.sub(r"max_\{?a'?\}?\s*Q\(s'?[,，].*?(?:$|。|；)", q_update, s, flags=re.IGNORECASE)
    # 残缺损失函数
    s = re.sub(r"L\(θ\)\s*=\s*E\[?.*?(?:$|。|；)", dqn_loss, s, flags=re.IGNORECASE)
    # 残缺 Double DQN target
    s = re.sub(r"y\s*=\s*r\s*\+\s*γ.*?Q_?target.*?(?:$|。|；)", ddqn_target, s, flags=re.IGNORECASE)

    # 删除常见数学域残片，如 γ ∈ 后被截断
    s = re.sub(r"γ\s*[∈in]+\s*$", "γ 为折扣因子，取值通常在 0 到 1 之间。", s)
    return s


def _latex_to_readable(expr: str) -> str:
    """将常见 LaTeX 公式转为可读文本（非严格数学排版）。"""
    s = str(expr or "")

    # 统一部分易触发字体告警的 Unicode 字符
    s = s.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")

    # 去掉首尾 $ 包裹，兼容行内数学片段
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1]

    # 常见转义字符先还原
    s = s.replace(r"\'", "'")
    s = s.replace(r"\(", "(").replace(r"\)", ")")
    s = s.replace(r"\[", "[").replace(r"\]", "]")

    # 先处理文本/字体命令，避免后续花括号清理丢失语义
    for _ in range(4):
        s = re.sub(r"\\(text|mathrm|mathbf|mathit|mathsf|mathtt|operatorname|mathcal)\{([^{}]+)\}", r"\2", s)

    # 常见结构命令
    for _ in range(4):
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    for _ in range(4):
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\[([^\]]+)\]\{([^{}]+)\}", r"root_\1(\2)", s)

    # 变换符号
    s = re.sub(r"\\hat\{([^{}]+)\}", r"\1_hat", s)
    s = re.sub(r"\\bar\{([^{}]+)\}", r"\1_bar", s)
    s = re.sub(r"\\tilde\{([^{}]+)\}", r"\1_tilde", s)
    s = re.sub(r"\\vec\{([^{}]+)\}", r"\1_vec", s)

    # mathbb 特殊集合符号
    s = re.sub(r"\\mathbb\{R\}", "R", s)
    s = re.sub(r"\\mathbb\{N\}", "N", s)
    s = re.sub(r"\\mathbb\{Z\}", "Z", s)
    s = re.sub(r"\\mathbb\{Q\}", "Q", s)
    s = re.sub(r"\\mathbb\{C\}", "C", s)
    s = re.sub(r"\\mathbb\{E\}", "E", s)
    s = re.sub(r"\\mathbb\{([A-Za-z])\}", r"\1", s)

    # 细空格/间距命令
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = s.replace(r"\quad", " ").replace(r"\qquad", " ")

    # 转义下划线与大括号
    s = s.replace(r"\_", "_")
    s = s.replace(r"\{", "{").replace(r"\}", "}")

    replace_map = {
        r"\\left": "",
        r"\\right": "",
        r"\\big": "",
        r"\\Big": "",
        r"\\bigl": "",
        r"\\bigr": "",
        r"\\Bigl": "",
        r"\\Bigr": "",
        r"\\cdot": " · ",
        r"\\times": " × ",
        r"\\leq": " <= ",
        r"\\geq": " >= ",
        r"\\neq": " != ",
        r"\\to": " -> ",
        r"\\mapsto": " |-> ",
        r"\\Rightarrow": " => ",
        r"\\Longrightarrow": " => ",
        r"\\Leftrightarrow": " <=> ",
        r"\\infty": "∞",
        r"\\pm": " ± ",
        r"\\mp": " ∓ ",
        r"\\epsilon": "ε",
        r"\\varepsilon": "ε",
        r"\\eta": "η",
        r"\\rho": "ρ",
        r"\\tau": "τ",
        r"\\omega": "ω",
        r"\\Omega": "Ω",
        r"\\alpha": "α",
        r"\\beta": "β",
        r"\\gamma": "γ",
        r"\\delta": "δ",
        r"\\theta": "θ",
        r"\\lambda": "λ",
        r"\\mu": "μ",
        r"\\sigma": "σ",
        r"\\pi": "π",
        r"\\sum": "sum",
        r"\\prod": "prod",
        r"\\int": "int",
        r"\\lim": "lim",
        r"\\max": "max",
        r"\\min": "min",
        r"\\sup": "sup",
        r"\\inf": "inf",
        r"\\log": "log",
        r"\\ln": "ln",
        r"\\exp": "exp",
        r"\\sin": "sin",
        r"\\cos": "cos",
        r"\\tan": "tan",
        r"\\argmax": "argmax",
        r"\\argmin": "argmin",
        r"\\forall": "forall",
        r"\\exists": "exists",
        r"\\in": " in ",
        r"\\notin": " not in ",
        r"\\subseteq": " subseteq ",
        r"\\subset": " subset ",
        r"\\supseteq": " supseteq ",
        r"\\cup": " union ",
        r"\\cap": " inter ",
        r"\\land": " and ",
        r"\\lor": " or ",
        r"\\neg": " not ",
        r"\\mathbb{E}": "E",
        r"\\mathbbE": "E",
        r"\\partial": "partial",
        r"\\nabla": "nabla",
        r"\\approx": " ~= ",
        r"\\sim": " ~ ",
        r"\\propto": " propto ",
    }
    for k, v in replace_map.items():
        s = s.replace(k, v)

    # 清理非 LaTeX 风格残留（常见于 LLM 生成的“类公式文本”）
    s = re.sub(r"\bbig\s*\(", "(", s, flags=re.IGNORECASE)
    s = re.sub(r"\)\s*big\b", ")", s, flags=re.IGNORECASE)
    s = re.sub(r"\bbigl\s*\(", "(", s, flags=re.IGNORECASE)
    s = re.sub(r"\)\s*bigr\b", ")", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBig\s*\(", "(", s)
    s = re.sub(r"\)\s*Big\b", ")", s)
    s = re.sub(r"\bsim\b", "~", s)

    # 纯文本 token 规范化（不依赖反斜杠）
    # 例如: leftarrow/rightarrow/left/right/gamma/theta
    s = re.sub(r"\bleftarrow\b", " ← ", s, flags=re.IGNORECASE)
    s = re.sub(r"\brightarrow\b", " → ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bleftrightarrow\b", " ↔ ", s, flags=re.IGNORECASE)

    greek_plain = {
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "theta": "θ",
        "lambda": "λ", "mu": "μ", "sigma": "σ", "pi": "π", "omega": "ω",
        "epsilon": "ε", "varepsilon": "ε",
    }
    for k, v in greek_plain.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)

    # 处理 left/right 包装残留: Q_hatleft(...right) -> Q_hat(...)
    s = re.sub(r"_hat\s*left\s*\(", "_hat(", s, flags=re.IGNORECASE)
    s = re.sub(r"\bleft\s*\(", "(", s, flags=re.IGNORECASE)
    s = re.sub(r"\)\s*right\b", ")", s, flags=re.IGNORECASE)

    # 撇号与 prime
    s = s.replace(r"\prime", "'")
    s = re.sub(r"\^\{\\prime\}", "'", s)
    s = re.sub(r"\^\\prime", "'", s)
    s = re.sub(r"\^\{\\\"\}", "''", s)

    # 处理上下标：x_{t+1} -> x_(t+1), x^2 -> x^(2)
    s = re.sub(r"_\{([^{}]+)\}", r"_(\1)", s)
    s = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", s)
    s = re.sub(r"_([A-Za-z0-9α-ωΑ-Ω]+)", r"_(\1)", s)
    s = re.sub(r"\^([A-Za-z0-9α-ωΑ-Ω]+)", r"^(\1)", s)

    # 兼容空格分隔的幂写法：theta^ - -> theta^(-)
    s = re.sub(r"\^\s*([-+])\s*", r"^(\1)", s)
    s = re.sub(r"\^\(\s*([-+])\s*\)", r"^(\1)", s)

    # 规范单字母帽子表达: Q_(hat) -> Q_hat
    s = re.sub(r"([A-Za-z0-9]+)_\(hat\)", r"\1_hat", s)

    # 处理残留形式: hatQ / hat(x) -> Q_hat / x_hat
    s = re.sub(r"hat\(([A-Za-z0-9_+\-*/]+)\)", r"\1_hat", s)
    s = re.sub(r"hat\s*([A-Za-z])", r"\1_hat", s)

    # 清理残留反斜杠命令: \theta -> theta
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)

    # 去掉其余花括号与多余空白
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\(", "(").replace("\\)", ")")
    s = s.replace("[", "(").replace("]", ")")
    # 清理重复括号与异常分号
    s = re.sub(r"\(\s*\(", "(", s)
    s = re.sub(r"\)\s*\)", ")", s)
    s = s.replace(";", " , ")
    s = re.sub(r"([α-ωΑ-Ω])([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])([α-ωΑ-Ω])", r"\1 \2", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    s = re.sub(r"\s*([,;:+\-*/=<>])\s*", r" \1 ", s)
    s = re.sub(r"\\s+", " ", s).strip()

    # 二次清理：避免格式化后出现 theta^( - ) 这类空格残留
    s = re.sub(r"\^\(\s*\-\s*\)", "^(-)", s)
    s = re.sub(r"\^\(\s*\+\s*\)", "^(+)", s)
    s = re.sub(r"~\s*D\)", "~ D", s)

    # 兜底：去除多余右括号，避免生成类似 "... D)" 的残留
    while s.count(")") > s.count("("):
        s = s.replace(")", "", 1)

    return s


def _looks_like_formula_continuation(line: str) -> bool:
    """识别公式续行（上一行已判定为公式时）。"""
    s = str(line or "").strip()
    if not s:
        return False
    if re.search(r"[\u4e00-\u9fff]", s):
        return False
    return bool(
        re.search(r"[=_^]|\b(max|min|argmax|argmin|sum|prod|sqrt|log|exp|E)\b", s, re.IGNORECASE)
        or re.match(r"^[\)\]\}\+\-\*/,\.:]", s)
        or re.search(r"[\(\)\[\]]", s)
    )


def _looks_like_formula(line: str) -> bool:
    """识别未包裹 $ 的公式行（LaTeX/数学表达式风格）。"""
    s = str(line or "").strip()
    if not s:
        return False
    # 显式公式标记
    if s.startswith("[公式]") or s.lower().startswith("formula"):
        return True

    # 含中文时从严判定，避免把正文误识别为公式
    has_cjk = re.search(r"[\u4e00-\u9fff]", s) is not None
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", s))

    has_equals = "=" in s
    has_subsup = ("_" in s or "^" in s)
    latex_cmd = re.search(r"\\[A-Za-z]+", s)
    token_hits = len(re.findall(r"\b(argmax|argmin|max|min|sum|prod|sqrt|sin|cos|tan|log|exp)\b", s, re.IGNORECASE))

    # 数学符号密度判定，避免误伤普通中文段落
    special_count = len(re.findall(r"[=+\-_^\\]", s))
    dense_symbol = special_count >= 4 and len(s) >= 12

    if has_cjk:
        # 中文占比较高且没有等式时，视为正文说明
        if cjk_count >= 4 and not has_equals:
            return False
        # 中文行仅在明确“等式型公式”时判定为公式
        return bool(has_equals and (has_subsup or latex_cmd or token_hits > 0))

    return bool(latex_cmd or (has_equals and (has_subsup or token_hits > 0)) or dense_symbol)


def _render_markdown_to_pdf(pdf, content: str):
    """将 Markdown 文本按可读性渲染到 PDF，重点保障代码与公式可见。"""
    content = _fix_dqn_formula_fragments(_stabilize_math_text(content))
    lines = str(content or "").replace("\r\n", "\n").split("\n")
    in_code_block = False
    in_math_block = False
    prev_formula_line = False
    paragraph_buf = []

    def flush_paragraph():
        if not paragraph_buf:
            return
        text = " ".join(part.strip() for part in paragraph_buf if part.strip())
        paragraph_buf.clear()
        if text:
            _safe_multicell(pdf, _format_paragraph_for_pdf(text), line_h=7)

    for raw in lines:
        line = _normalize_markdown_line(raw)
        stripped = line.strip()

        # 代码块围栏
        if stripped.startswith("```"):
            flush_paragraph()
            in_code_block = not in_code_block
            prev_formula_line = False
            if in_code_block:
                pdf.ln(1)
                pdf.set_font("CJK", "B", 10)
                _safe_multicell(pdf, "[代码块]", line_h=6)
                pdf.set_font("CJK", "", 10)
            else:
                pdf.ln(1)
                pdf.set_font("CJK", "", 11)
            continue

        # 公式块围栏
        if stripped == "$$":
            flush_paragraph()
            in_math_block = not in_math_block
            prev_formula_line = False
            if in_math_block:
                pdf.ln(1)
                pdf.set_font("CJK", "B", 10)
                _safe_multicell(pdf, "[公式]", line_h=6)
                pdf.set_font("CJK", "", 11)
            else:
                pdf.ln(1)
            continue

        # 代码块内容：保留缩进和符号
        if in_code_block:
            code_line = line.replace("\t", "    ")
            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(30, 30, 30)
            pdf.set_x(pdf.l_margin + 1)
            # 使用 0 宽度时会从当前位置写到右边距
            pdf.multi_cell(0, 6, f"  {code_line}" if code_line else " ", fill=True, align="L")
            pdf.set_text_color(0, 0, 0)
            prev_formula_line = False
            continue

        # 非代码路径下清洗行内 Markdown 强调符号
        line = _clean_inline_markdown(line)
        stripped = line.strip()

        # 公式块内容：原样保留表达式字符
        if in_math_block:
            # 若未闭合 $$ 导致进入正文，自动退出公式块
            if not _looks_like_formula(line):
                in_math_block = False
                paragraph_buf.append(line)
                prev_formula_line = False
                continue

            _render_formula_text_block(pdf, line if line else " ", line_h=7)
            prev_formula_line = True
            continue

        # 单行块公式 $$...$$
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            flush_paragraph()
            expr = stripped[2:-2].strip()
            pdf.set_font("CJK", "B", 10)
            _safe_multicell(pdf, "[公式]", line_h=6)
            pdf.set_font("CJK", "", 11)
            _render_formula_text_block(pdf, expr, line_h=7)
            prev_formula_line = True
            continue

        # 空行
        if not stripped:
            flush_paragraph()
            pdf.ln(3)
            prev_formula_line = False
            continue

        # Markdown 标题
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            size = {1: 15, 2: 14, 3: 13, 4: 12, 5: 11, 6: 11}.get(level, 12)
            pdf.set_font("CJK", "B", size)
            _safe_multicell(pdf, title, line_h=8)
            pdf.set_font("CJK", "", 11)
            prev_formula_line = False
            continue

        # 常见无序列表优化
        if re.match(r"^[-*+]\s+", stripped):
            flush_paragraph()
            text = re.sub(r"^[-*+]\s+", "- ", stripped)
            _safe_multicell(pdf, text, line_h=7)
            prev_formula_line = False
            continue

        # 常见有序列表保持
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            _safe_multicell(pdf, stripped, line_h=7)
            prev_formula_line = False
            continue

        # 未包裹 $ 的公式行
        if _looks_like_formula(line) or (prev_formula_line and _looks_like_formula_continuation(line)):
            flush_paragraph()
            formula_text = line.replace("[公式]", "").strip()
            if formula_text:
                _render_formula_text_block(pdf, formula_text, line_h=7)
            else:
                _safe_multicell(pdf, "[公式]", line_h=6)
            prev_formula_line = True
            continue

        # 行内公式 $...$：转成可读文本
        if "$" in line:
            line = re.sub(r"\$(.+?)\$", lambda m: _latex_to_readable(m.group(1)), line)

        # 普通文本：先缓存，跨行合并成段
        paragraph_buf.append(line)
        prev_formula_line = False

    flush_paragraph()


@tool
def generate_pdf(report_dict: dict, output_path: str) -> str:
    """将实验报告导出为 PDF 文件。

    基于 fpdf2 生成 PDF，自动查找系统中文字体。
    若未找到中文字体，降级为输出 Markdown。

    Args:
        report_dict: 报告数据字典，包含 meta 和 sections
            {
                "meta": {"title": "...", "student_name": "...", ...},
                "sections": [
                    {"section_id": "1", "title": "一. 实验题目", "content": "..."},
                    ...
                ]
            }
        output_path: PDF 输出路径，如 "./output/report.pdf"

    Returns:
        生成的 PDF 文件路径，或降级的 Markdown 路径
    """
    # 优先 DOCX->PDF 管线（数学公式可读性更好）
    docx_pipeline_result = _try_generate_pdf_via_docx(report_dict, output_path)
    if docx_pipeline_result:
        return docx_pipeline_result

    try:
        from fpdf import FPDF
    except ImportError:
        return _generate_markdown_fallback(report_dict, output_path)

    # 查找中文字体
    cjk_font = _find_cjk_font()
    if cjk_font is None:
        # 无中文字体，降级为 Markdown
        return _generate_markdown_fallback(report_dict, output_path)

    try:
        class ReportPDF(FPDF):
            def footer(self):
                self.set_y(-10)
                self.set_font("CJK", "", 9)
                self.cell(0, 6, f"第 {self.page_no()} 页", align="C")

        pdf = ReportPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # 注册中文字体
        pdf.add_font("CJK", "", cjk_font)
        pdf.add_font("CJK", "B", cjk_font)

        meta = report_dict.get("meta", {})
        sections = report_dict.get("sections", [])
        figures = report_dict.get("figures", [])
        attachments = report_dict.get("attachments", [])

        # === 封面 ===
        pdf.add_page()
        pdf.set_font("CJK", "B", 22)
        pdf.ln(40)
        _safe_center_line(pdf, meta.get("title", "实验报告"), line_h=15)
        pdf.ln(10)

        pdf.set_font("CJK", "", 14)
        _safe_center_line(pdf, f"课程: {meta.get('course_name', '人工智能')}", line_h=10)
        _safe_center_line(pdf, f"姓名: {meta.get('student_name', '')}    学号: {meta.get('student_id', '')}", line_h=10)
        _safe_center_line(pdf, f"日期: {meta.get('date', datetime.now().strftime('%Y-%m-%d'))}", line_h=10)

        # === 章节内容 ===
        for section in sections:
            pdf.add_page()

            # 章节标题
            pdf.set_font("CJK", "B", 16)
            _safe_multicell(pdf, section.get("title", "无标题"), line_h=10)
            pdf.ln(5)

            # 章节内容
            pdf.set_font("CJK", "", 11)
            content = _fix_dqn_formula_fragments(_stabilize_math_text(section.get("content", "")))
            _render_markdown_to_pdf(pdf, content)

        # === 图片与附件列表（PDF中以路径清单方式给出） ===
        if figures:
            pdf.add_page()
            pdf.set_font("CJK", "B", 16)
            _safe_multicell(pdf, "实验结果图片", line_h=10)
            pdf.ln(3)
            pdf.set_font("CJK", "", 11)
            for i, fig in enumerate(figures, 1):
                _safe_multicell(pdf, f"{i}. {fig}", line_h=7)
                _embed_image_or_fallback(pdf, fig, line_h=7)
                pdf.ln(2)

        if attachments:
            pdf.add_page()
            pdf.set_font("CJK", "B", 16)
            _safe_multicell(pdf, "附件清单", line_h=10)
            pdf.ln(3)
            pdf.set_font("CJK", "", 11)
            for i, item in enumerate(attachments, 1):
                _safe_multicell(pdf, f"{i}. {item}", line_h=7)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pdf.output(output_path)

        return f"[PDF] 报告已生成: {output_path}"

    except Exception as e:
        reason = f"PDF生成异常: {type(e).__name__}: {e!r}"
        return _generate_markdown_fallback(report_dict, output_path, reason=reason)


@tool
def generate_markdown(report_dict: dict, output_path: str) -> str:
    """将实验报告导出为 Markdown 文件。

    Args:
        report_dict: 报告数据字典（同 generate_pdf）
        output_path: Markdown 输出路径

    Returns:
        生成的文件路径
    """
    meta = report_dict.get("meta", {})
    sections = report_dict.get("sections", [])
    figures = report_dict.get("figures", [])
    attachments = report_dict.get("attachments", [])

    lines = [
        f"# {meta.get('course_name', '人工智能')} 实验报告",
        "",
        f"姓名: {meta.get('student_name', '')}  学号: {meta.get('student_id', '')}",
        f"日期: {meta.get('date', datetime.now().strftime('%Y-%m-%d'))}",
        "",
        "---",
        "",
    ]

    for section in sections:
        lines.append(f"### {section.get('title', '无标题')}")
        lines.append("")
        lines.append(_fix_dqn_formula_fragments(_stabilize_math_text(section.get("content", ""))))
        lines.append("")

    if figures:
        lines.append("### 实验结果图片")
        lines.append("")
        for i, fig in enumerate(figures, 1):
            lines.append(f"{i}. {fig}")
            lines.append(f"![figure_{i}]({fig})")
            lines.append("")

    if attachments:
        lines.append("### 附件清单")
        lines.append("")
        for i, item in enumerate(attachments, 1):
            lines.append(f"{i}. {item}")
        lines.append("")

    content = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"[Markdown] 报告已生成: {output_path}"


def _generate_markdown_fallback(report_dict: dict, output_path: str, reason: Optional[str] = None) -> str:
    """PDF 生成失败时的降级方案：生成 Markdown"""
    md_path = output_path.rsplit(".", 1)[0] + ".md"
    generate_markdown.func(report_dict, md_path)
    extra = f"; 原因: {reason}" if reason else ""
    return f"[降级] PDF生成失败（可能缺少中文字体），已生成Markdown替代: {md_path}{extra}"
