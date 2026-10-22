"""
多文件解析工具

统一入口 parse_file()，根据扩展名自动选择解析器。
支持: PDF / 图片(OCR) / Word / Markdown / TXT / CSV预览
"""
import os
import csv
import base64
import json
import mimetypes
from langchain_core.tools import tool


# ============================================================
# 各文件类型解析器
# ============================================================

def _image_stub_text(label: str = "图片", detail: str = "") -> str:
    """不做OCR，仅返回图片占位说明，后续交给视觉模型分析。"""
    detail = f" ({detail})" if detail else ""
    return f"[图片占位] {label}{detail}"


def _parse_pdf(filepath: str) -> str:
    """解析 PDF 文件，按阅读顺序提取文本 + 图片占位信息（不做OCR）。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "[Error] 请安装 PyMuPDF: pip install PyMuPDF"

    try:
        doc = fitz.open(filepath)
        pages = []

        for page_num, page in enumerate(doc, start=1):
            # ============================================================
            # 收集页面中所有"区块"（文本块 + 图片块），按 y 坐标排序
            # 保证输出顺序 = 阅读顺序
            # ============================================================
            blocks = []  # [(y0, type, content)]

            # --- 1. 提取文本块（带坐标）---
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block["type"] == 0:  # 文本块
                    block_text = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            block_text.append(line_text.strip())
                    if block_text:
                        y0 = block["bbox"][1]  # 文本块顶部 y 坐标
                        blocks.append((y0, "text", " ".join(block_text)))
                elif block["type"] == 1:  # 图片块（PyMuPDF 已经识别出的图片）
                    y0 = block["bbox"][1]
                    # 从 block 中提取图片 xref
                    image_xref = None
                    for img_info in block.get("image", []):
                        image_xref = img_info  # 直接是 xref 数字
                    if image_xref is not None:
                        blocks.append((y0, "image_block", image_xref))

            # --- 2. 补充 get_image_info() 中可能遗漏的图片 ---
            #      有些图片可能没出现在 text_dict 的 block 中
            image_infos = page.get_image_info()
            existing_xrefs = {
                b[2] for b in blocks if b[1] == "image_block"
            }
            for img_info in image_infos:
                xref = img_info.get("xref", 0)
                if xref and xref not in existing_xrefs:
                    y0 = img_info["bbox"][1]
                    blocks.append((y0, "image_info", xref))

            # --- 3. 处理 get_images() 中可能遗漏的内嵌图片 ---
            #      （有些图片 get_image_info 也检测不到，只在 get_images 中出现）
            all_xrefs_handled = {
                b[2] for b in blocks if b[1] in ("image_block", "image_info")
            }
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref not in all_xrefs_handled:
                    # 没有位置信息，追加到末尾
                    y0 = blocks[-1][0] + 1000 if blocks else 0
                    blocks.append((y0, "image_xref", xref))

            # --- 4. 按 y 坐标排序（从上到下 = 阅读顺序）---
            blocks.sort(key=lambda b: b[0])

            # --- 5. 遍历有序区块，依次处理 ---
            page_parts = []
            img_counter = 0

            for _, block_type, data in blocks:
                if block_type == "text":
                    if data.strip():
                        page_parts.append(data.strip())

                elif block_type in ("image_block", "image_info", "image_xref"):
                    try:
                        base_image = doc.extract_image(data)
                        image_bytes = base_image["image"]
                        if len(image_bytes) > 5120:  # 过滤 < 5KB 的小图标
                            img_counter += 1
                            label = f"PDF第{page_num}页-图{img_counter}"
                            page_parts.append(_image_stub_text(label, "已检测到图片，等待视觉模型分析"))
                    except Exception:
                        continue

            # --- 6. 如果该页没有任何内容 → 整页截图占位 ---
            if not page_parts:
                label = f"PDF第{page_num}页(整页扫描)"
                page_parts.append(_image_stub_text(label, "检测到整页图像，未执行OCR"))

            if page_parts:
                pages.append("\n\n".join(page_parts))

        doc.close()

        if not pages:
            return f"[PDF] 文件中未提取到任何内容: {os.path.basename(filepath)}"

        header = f"[PDF] {os.path.basename(filepath)} (共{len(pages)}页)\n"
        return header + "\n\n---\n\n".join(pages)
    except Exception as e:
        return f"[Error] PDF解析失败 {filepath}: {e}"


def _parse_image(filepath: str) -> str:
    """解析图片文件，返回非OCR占位信息，供视觉模型分析。"""
    try:
        with open(filepath, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        return f"[Error] 读取图片文件失败 {filepath}: {e}"

    file_size = len(image_bytes)
    header = f"[图片] {os.path.basename(filepath)}\n"
    return header + _image_stub_text(os.path.basename(filepath), f"size={file_size} bytes")


def _image_to_data_url(filepath: str) -> str:
    mime, _ = mimetypes.guess_type(filepath)
    if not mime:
        mime = "image/png"
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@tool
def analyze_image_vlm(filepath: str, context: str = "") -> str:
    """使用 Qwen2.5-VL 分析图片/图表，返回严格 JSON 字符串。需要 DASHSCOPE_API_KEY。"""
    try:
        from openai import OpenAI
        from config import config
    except Exception as e:
        return f"[Error] 视觉模型依赖不可用: {e}"

    path = str(filepath or "").strip().strip('"').strip("'").replace("\\", "/")
    if not path or not os.path.exists(path):
        return f"[Error] 图片文件不存在: {path}"
    if not config.vision_api_key:
        return "[Error] 未设置 DASHSCOPE_API_KEY，无法调用 Qwen2.5-VL"

    prompt = f"""请分析这张实验结果图片。若它是图表/曲线图，必须结合关键节点值分析大趋势。

必须输出严格 JSON，不要 Markdown，不要解释文字。JSON 结构如下：
{{
  "images": [
    {{
      "path": "{path}",
      "summary": "图片整体内容摘要",
      "chart_type": "曲线图/柱状图/截图/非图表/未知",
      "trend_analysis": "如果是图表，描述起点、上升/下降阶段、峰值/谷值、拐点、平台期、末尾表现、波动区间、是否收敛",
    "report_integration": "结合实验背景与方法，说明这张图在报告里应该怎么解读",
    "should_embed": true,
    "embed_reason": "如果用户明确要求嵌入图片则必须为true；如果用户未要求，则根据图片是否能支撑实验结果分析自行判断",
      "key_points": [
        {{"name": "起点/初始阶段", "value": "能读到则写约略值，否则写无法精确读取", "meaning": "含义"}},
        {{"name": "峰值/谷值/拐点/末尾", "value": "能读到则写约略值，否则写无法精确读取", "meaning": "含义"}}
      ],
      "evidence": ["直接从图片观察到的证据"],
      "recommended_section": "实验结果与分析",
      "placement_hint": "放在讨论训练收敛趋势或结果分析的段落后",
      "keywords": ["结果", "曲线", "奖励", "收敛", "loss", "reward"]
    }}
  ]
}}

规则：
- 如果上下文显示用户明确要求“嵌入图片/插入图片/把图片放进报告/分析并嵌入报告”，should_embed 必须为 true。
- 如果用户没有明确要求嵌入图片，请根据图片是否与实验结果、曲线趋势、关键指标分析相关，自行判断 should_embed。
- 如果 should_embed 为 true，placement_hint 必须具体说明应放在哪一段之后。
- 绝对不要复述本提示词、JSON格式说明、实验背景原文、文件路径列表或代码证据原文。
- summary、trend_analysis、report_integration 只能写“可直接放进实验报告”的最终分析结论。
- 不要编造精确数值；看不清时写“约略”或“无法精确读取”。
- 不能只写 OCR 失败；如果能看到曲线形状，必须给出趋势判断。
- 结合上下文：{str(context or '')[:1200]}
"""

    try:
        client = OpenAI(api_key=config.vision_api_key, base_url=config.vision_base_url)
        resp = client.chat.completions.create(
            model=config.vision_model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(path)}},
                ],
            }],
            temperature=0.1,
            max_tokens=config.vision_max_tokens,
        )
        content = resp.choices[0].message.content or ""
        # 简单校验，仍返回原文本供上层 JSON 抽取兜底
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(content[start:end + 1])
            return json.dumps(obj, ensure_ascii=False)
        return content
    except Exception as e:
        return f"[Error] Qwen2.5-VL 图片分析失败: {e}"


def _parse_docx(filepath: str) -> str:
    """解析 Word (.docx) 文档，按阅读顺序提取文本 + 图片占位信息（不做OCR）。"""
    try:
        from docx import Document
    except ImportError:
        return "[Error] 请安装 python-docx: pip install python-docx"

    import zipfile
    from xml.etree import ElementTree

    try:
        doc = Document(filepath)

        # XML 命名空间常量
        NS_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        NS_DRAWING = 'http://schemas.openxmlformats.org/drawingml/2006/main'
        NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

        # ============================================================
        # 1. 从 ZIP 中预提取所有内嵌图片
        # ============================================================
        image_counter = 0
        embedded_images = {}  # {rId: image_bytes}

        try:
            with zipfile.ZipFile(filepath, 'r') as z:
                rels_files = [f for f in z.namelist() if f.endswith('.rels')]
                rid_to_file = {}
                for rels_path in rels_files:
                    try:
                        tree = ElementTree.parse(z.open(rels_path))
                        ns = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}
                        for rel in tree.findall('.//r:Relationship', ns):
                            rid = rel.get('Id')
                            target = rel.get('Target')
                            if target and any(target.lower().endswith(ext) for ext in
                                              ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                                media_name = target.split('/')[-1]
                                rid_to_file[rid] = f"word/media/{media_name}"
                    except Exception:
                        continue

                for file_name in z.namelist():
                    if file_name.startswith('word/media/') and any(
                        file_name.lower().endswith(ext)
                        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
                    ):
                        match_name = file_name.split('/')[-1]
                        matched_rids = [rid for rid, path in rid_to_file.items()
                                       if path.split('/')[-1] == match_name]
                        for rid in matched_rids:
                            embedded_images[rid] = z.read(file_name)
        except Exception:
            pass

        # ============================================================
        # 2. 按文档顺序遍历 body 中的每个元素（段落 + 表格交替）
        #    doc.element.body 是 XML 元素，子元素按文档原始顺序排列
        # ============================================================
        body = doc.element.body
        paragraphs = []

        for child in body:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

            if tag == 'p':
                # ==================== 段落 ====================
                # 按 run 顺序提取文本和图片，保持阅读顺序
                para_text_parts = []
                for run_elem in child.findall(f'{{{NS_W}}}r'):
                    # --- run 中的文本 ---
                    text_elems = run_elem.findall(f'{{{NS_W}}}t')
                    for t in text_elems:
                        if t.text:
                            para_text_parts.append(t.text)

                    # --- run 中的内嵌图片 ---
                    drawing_elems = run_elem.findall(f'.//{{{NS_DRAWING}}}blip')
                    for blip in drawing_elems:
                        embed_rid = blip.get(f'{{{NS_R}}}embed')
                        if embed_rid and embed_rid in embedded_images:
                            img_bytes = embedded_images[embed_rid]
                            if len(img_bytes) > 5120:
                                image_counter += 1
                                para_text_parts.append(_image_stub_text(f"DOCX图{image_counter}", "已检测到内嵌图片，等待视觉模型分析"))

                # 合并段落内所有文本
                full_text = "".join(para_text_parts).strip()
                if full_text:
                    paragraphs.append(full_text)

            elif tag == 'tbl':
                # ==================== 表格 ====================
                rows = child.findall(f'.//{{{NS_W}}}tr')
                for row_elem in rows:
                    cells = row_elem.findall(f'.//{{{NS_W}}}tc')
                    row_texts = []
                    for cell in cells:
                        # 提取单元格内所有文本
                        cell_text = "".join(
                            t.text or ""
                            for t in cell.findall(f'.//{{{NS_W}}}t')
                        ).strip()
                        if cell_text:
                            row_texts.append(cell_text)
                    if row_texts:
                        paragraphs.append(f"[表格行] {' | '.join(row_texts)}")

        if not paragraphs:
            return f"[DOCX] 文档为空: {os.path.basename(filepath)}"

        header = f"[DOCX] {os.path.basename(filepath)}\n"
        if image_counter > 0:
            header += f"(识别到 {image_counter} 张内嵌图片)\n"
        return header + "\n\n".join(paragraphs)
    except Exception as e:
        return f"[Error] DOCX解析失败 {filepath}: {e}"


def _parse_markdown(filepath: str) -> str:
    """读取 Markdown 文件"""
    for encoding in ["utf-8", "gbk"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            return f"[Markdown] {os.path.basename(filepath)}\n{content}"
        except UnicodeDecodeError:
            continue
    return f"[Error] 无法解码文件: {filepath}"


def _parse_csv_preview(filepath: str, nrows: int = 10) -> str:
    """读取 CSV 前 N 行预览（含列名），供决策节点判断是否需 DataExecutor"""
    for encoding in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return f"[CSV] 文件为空: {os.path.basename(filepath)}"

                rows = [", ".join(header)]
                for i, row in enumerate(reader):
                    if i >= nrows:
                        break
                    rows.append(", ".join(row))

            # 统计总行数
            total_lines = sum(1 for _ in open(filepath, "r", encoding=encoding)) - 1
            return (
                f"[CSV预览] {os.path.basename(filepath)}\n"
                f"列名: {', '.join(header)} (共{len(header)}列)\n"
                f"总数据行数: {total_lines} (预览前{nrows}行)\n"
                + "\n".join(rows)
            )
        except (UnicodeDecodeError, UnicodeError):
            continue
    return f"[Error] 无法解码CSV文件: {filepath}"


def _parse_txt(filepath: str) -> str:
    """读取纯文本文件"""
    for encoding in ["utf-8", "gbk", "latin-1"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            return f"[TXT] {os.path.basename(filepath)}\n{content}"
        except (UnicodeDecodeError, UnicodeError):
            continue
    return f"[Error] 无法解码文件: {filepath}"


# ============================================================
# 解析器注册表
# ============================================================

PARSER_REGISTRY = {
    ".pdf": _parse_pdf,
    ".png": _parse_image,
    ".jpg": _parse_image,
    ".jpeg": _parse_image,
    ".bmp": _parse_image,
    ".tiff": _parse_image,
    ".docx": _parse_docx,
    ".doc": _parse_docx,
    ".md": _parse_markdown,
    ".markdown": _parse_markdown,
    ".csv": _parse_csv_preview,
    ".txt": _parse_txt,
    ".py": _parse_txt,
    ".json": _parse_txt,
}


# ============================================================
# 工具定义
# ============================================================

@tool
def parse_file(filepath: str) -> str:
    """解析用户附带的文件，根据扩展名自动选择解析器。

    支持格式: PDF / 图片(OCR) / Word / Markdown / TXT / CSV预览 / Python / JSON
    此工具被 FileParserAgent 调用，用于将用户附带的各类文件转换为文本。

    Args:
        filepath: 文件绝对路径

    Returns:
        文件解析后的文本内容（带格式标记头）
    """
    if not os.path.exists(filepath):
        return f"[Error] 文件不存在: {filepath}"

    ext = os.path.splitext(filepath)[1].lower()
    parser = PARSER_REGISTRY.get(ext)

    if parser is None:
        return (
            f"[Error] 不支持的文件格式: {ext}。"
            f"支持: {sorted(set(PARSER_REGISTRY.keys()))}"
        )

    try:
        return parser(filepath)
    except Exception as e:
        return f"[Error] 解析文件 {filepath} 失败: {e}"


@tool
def parse_multiple_files(filepaths: str) -> str:
    """批量解析多个文件，用逗号分隔路径。

    用于一次性解析用户提供的所有附件（PDF、图片、Word等），
    自动识别每种文件类型并调用对应解析器。

    Args:
        filepaths: 逗号分隔的文件路径列表

    Returns:
        所有文件解析后的文本（用分隔符隔开）
    """
    paths = [p.strip() for p in filepaths.split(",") if p.strip()]
    if not paths:
        return "[Error] 未提供有效的文件路径"

    results = []
    for path in paths:
        result = parse_file.invoke(path)
        results.append(result)

    return "\n\n" + "=" * 60 + "\n\n".join(results)
