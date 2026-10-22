"""
多智能体实验报告生成系统 —— 主入口

用法:
    python main.py --prompt "用DT_data.csv做决策树分类，我叫张三，学号20240001" --data-files "data/DT_data.csv"
    python main.py --prompt "实现一阶逻辑归结推理，需要写ResolutionFOL函数"
"""
import sys
import os
import argparse
import re
import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from state import RawUserInput
from memory import ShortTermMemory, LongTermMemory
from graph import build_report_graph
from tools import generate_pdf, generate_markdown, set_memory_instance


def _clean_path_token(path: str) -> str:
    s = str(path or "").strip().strip('"').strip("'")
    s = s.rstrip("，。；;：:！？!?、）)]】}>\n\r\t ")
    return s.replace("\\", "/")


def _extract_abs_paths_from_prompt(prompt: str):
    text = str(prompt or "")
    # 提取 Windows 绝对路径片段，去掉尾随标点
    hits = re.findall(r"[A-Za-z]:\\[^\s\",，。；;：:！？!?）)\]】>]+", text)
    return [_clean_path_token(p) for p in hits]


def _path_in_nested_command(prompt: str, path: str) -> bool:
    """判断路径是否位于“测试命令/示例命令”等嵌套命令片段中。"""
    text = str(prompt or "")
    p = str(path or "")
    if not text or not p:
        return False

    # 用户明确要求把测试命令中的图片作为当前报告材料时，允许嵌入。
    explicit_allow = [
        "把测试命令中的图片", "测试命令中的图片作为当前", "将测试命令中的图片",
        "测试截图作为当前", "测试结果图作为当前",
    ]
    if any(x in text for x in explicit_allow):
        return False

    variants = {
        p,
        p.replace("/", "\\"),
        p.replace("\\", "/"),
        os.path.basename(p),
    }
    indices = [text.find(v) for v in variants if v and text.find(v) >= 0]
    if not indices:
        return False
    path_idx = min(indices)

    markers = ["测试命令如下", "测试命令", "示例命令", "历史命令", "复现命令"]
    marker_indices = [text.rfind(m, 0, path_idx) for m in markers]
    marker_idx = max(marker_indices) if marker_indices else -1
    if marker_idx < 0:
        return False

    # 若 marker 与路径之间没有明显重新开始当前材料描述，则视为嵌套命令材料。
    between = text[marker_idx:path_idx]
    reset_markers = ["当前项目材料", "当前实验结果图", "本项目结果图"]
    return not any(x in between for x in reset_markers)


def _mark_path_context(prompt: str, paths) -> dict:
    """给路径打来源标记；不删除路径，只标注是否来自嵌套命令。"""
    out = {}
    items = paths or []
    if isinstance(items, str):
        items = [items]
    for p in items:
        np = _clean_path_token(p)
        if np:
            out[np] = "nested_command" if _path_in_nested_command(prompt, np) else "current_material"
    return out


def _is_current_material_path(raw_input: dict, path: str) -> bool:
    """路径是否可作为当前报告材料使用。"""
    ctx = raw_input.get("path_context") or {}
    p = _clean_path_token(path)
    if not p:
        return True
    variants = {p, p.replace("/", "\\"), p.replace("\\", "/")}
    for k, v in ctx.items():
        nk = _clean_path_token(k)
        if nk in variants:
            return v != "nested_command"
    return not _path_in_nested_command(str(raw_input.get("prompt", "")), p)


def _resolve_existing_path(path: str, roots: list[str]) -> str:
    norm = _clean_path_token(path)
    if not norm:
        return ""
    if os.path.exists(norm):
        return norm

    base = os.path.basename(norm)
    if not base:
        return ""

    for root in roots:
        r = _clean_path_token(root)
        if not r or not os.path.isdir(r):
            continue
        try:
            matches = glob.glob(os.path.join(r, "**", base), recursive=True)
            for m in matches:
                if os.path.exists(m):
                    return m.replace("\\", "/")
        except Exception:
            continue
    return ""


def _collect_image_paths(final_state, raw_input):
    """从输入与解析结果中提取可用图片路径。"""
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    candidates = []
    prompt = raw_input.get("prompt", "")

    refs = raw_input.get("reference_materials") or []
    if isinstance(refs, str):
        refs = [refs]
    for ref in refs:
        candidates.append((ref, "explicit_reference"))

    parsed = final_state.get("parsed_prompt_fields", {}) or {}
    for key in ("file_paths", "normalized_paths"):
        vals = parsed.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        for val in vals:
            candidates.append((val, "prompt_extracted"))

    seen = set()
    images = []
    roots = []
    code_file = _clean_path_token(raw_input.get("code_file") or "")
    if code_file:
        roots.append(os.path.dirname(code_file))
    roots.extend([os.path.dirname(_clean_path_token(x)) for x in refs if _clean_path_token(x)])
    roots.extend([os.path.dirname(p) for p in _extract_abs_paths_from_prompt(raw_input.get("prompt", ""))])
    roots = [r for r in roots if r]

    for p, source in candidates:
        path = _clean_path_token(p)
        if not path:
            continue
        norm = path
        ext = os.path.splitext(norm)[1].lower()
        if ext not in image_exts:
            continue
        if not _is_current_material_path(raw_input, norm):
            continue
        if not os.path.exists(norm):
            recovered = _resolve_existing_path(norm, roots)
            if recovered:
                norm = recovered
        if not _is_current_material_path(raw_input, norm):
            continue
        key = norm.lower()
        if key in seen:
            continue
        if os.path.exists(norm):
            images.append(norm)
            seen.add(key)
    return images


def _collect_attachments(raw_input):
    """收集附件路径（参考资料/数据文件/代码文件）。"""
    candidates = []
    prompt = raw_input.get("prompt", "")
    refs = raw_input.get("reference_materials") or []
    data_files = raw_input.get("data_files") or []
    code_file = raw_input.get("code_file")

    if isinstance(refs, str):
        refs = [refs]
    if isinstance(data_files, str):
        data_files = [data_files]

    candidates.extend(refs)
    candidates.extend(data_files)
    if code_file:
        candidates.append(code_file)

    out = []
    seen = set()
    roots = []
    roots.extend([os.path.dirname(p) for p in _extract_abs_paths_from_prompt(raw_input.get("prompt", ""))])
    code_file_norm = _clean_path_token(code_file) if code_file else ""
    if code_file_norm:
        roots.append(os.path.dirname(code_file_norm))
    for p in candidates:
        path = _clean_path_token(p)
        if not path:
            continue
        norm = path
        if not _is_current_material_path(raw_input, norm):
            continue
        if not os.path.exists(norm):
            recovered = _resolve_existing_path(norm, roots)
            if recovered:
                norm = recovered
        if not _is_current_material_path(raw_input, norm):
            continue
        key = norm.lower()
        if key in seen:
            continue
        if os.path.exists(norm):
            out.append(norm)
            seen.add(key)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="多智能体实验报告生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python main.py --prompt \"用DT_data.csv做决策树分类，我叫张三\"\n"
               "  python main.py --prompt \"实现归结推理\" --references \"实验要求.pdf\""
    )
    parser.add_argument("--prompt", type=str, required=True, help="用户提示词（必填）")
    parser.add_argument("--data-files", type=str, nargs="*", help="数据文件路径")
    parser.add_argument("--references", type=str, nargs="*", help="参考资料路径")
    parser.add_argument("--code-file", type=str, help="参考代码文件")
    parser.add_argument("--output-format", type=str, default="pdf",
                        choices=["pdf", "markdown"])
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    if not config.api_key:
        print("❌ 未设置 DEEPSEEK_API_KEY 环境变量")
        print("  请运行: $env:DEEPSEEK_API_KEY = 'sk-xxx'")
        sys.exit(1)

    raw_input: RawUserInput = {
        "prompt": args.prompt,
        "data_files": args.data_files,
        "reference_materials": args.references,
        "code_file": args.code_file,
        "template": None,
        "output_format": args.output_format,
    }
    raw_input["path_context"] = _mark_path_context(
        args.prompt,
        _extract_abs_paths_from_prompt(args.prompt) + (args.references or []) + (args.data_files or []) + ([args.code_file] if args.code_file else [])
    )

    print("=" * 60)
    print("  多智能体实验报告生成系统")
    print("=" * 60)
    print(f"  提示词: {args.prompt[:80]}...")
    if args.data_files:
        print(f"  数据文件: {args.data_files}")
    if args.references:
        print(f"  参考资料: {args.references}")
    print()

    # 记忆系统
    short_mem = ShortTermMemory()
    long_mem = LongTermMemory()
    set_memory_instance(long_mem)  # 注入长期记忆到 memory_search 工具
    prefs = long_mem.get_preferences()
    if prefs:
        print(f"📝 已加载用户偏好: {prefs.get('student_name', '?')}")

    # 初始 State
    initial_short_term = [{
        "role": "user",
        "agent_name": "User",
        "content": args.prompt,
        "timestamp": datetime.now().isoformat(),
        "metadata": {"source": "cli_prompt"},
    }]

    initial_state = {
        "raw_input": raw_input,
        "resolved_input": {},
        "plan": [],
        "current_task_index": 0,
        "current_phase": "input_resolve",
        "parsed_prompt_fields": {},
        "parsed_file_contents": [],
        "memory_retrieval_results": [],
        "rag_input_results": [],
        "theory_content": "",
        "method_content": "",
        "draft_sections": {},
        "review_comments": [],
        "revision_count": 0,
        "max_revision_count": config.max_reflection_rounds,
        "memory": {"short_term": initial_short_term, "long_term_keys": []},
        "next_agent": "",
        "final_report": None,
        "status": "planning",
        "execution_log": [],
    }

    # 编译图
    print("🔧 编译 LangGraph 图...")
    graph = build_report_graph()
    print("✅ 图编译完成 (16 节点)\n")

    print("🚀 开始生成实验报告...\n")
    print("-" * 60)

    try:
        # 使用 invoke 直接获取最终状态（stream 会导致 dict 复制问题）
        config_override = {"recursion_limit": 200}
        final_state = graph.invoke(initial_state, config=config_override)

        print("-" * 60)
        print(f"\n✅ 执行完成")

        # 用 final_state 构建报告
        resolved = final_state.get("resolved_input", {})

        # 优先使用决策节点的 merge 结果
        writer_output = final_state.get("WriterDecision_output", "")
        draft = final_state.get("draft_sections", {})
        theory = final_state.get("theory_content", "")
        method = final_state.get("method_content", "")

        if writer_output:
            report_content = writer_output
        else:
            report_content = theory + "\n\n" + method if theory or method else ""
            if draft:
                report_content = "\n\n".join(draft.values())

        final_report = {
            "meta": {
                "title": resolved.get("topic", args.prompt[:50]),
                "student_name": resolved.get("student_name", ""),
                "student_id": resolved.get("student_id", ""),
                "course_name": resolved.get("course_name", "人工智能"),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "version": final_state.get("revision_count", 0) + 1,
            },
            "sections": [{
                "section_id": "main",
                "title": "实验报告",
                "content": report_content,
                "source_agent": "SectionWriter",
                "status": "final",
            }],
            "figures": _collect_image_paths(final_state, raw_input),
            "attachments": _collect_attachments(raw_input),
            "image_analysis": final_state.get("image_analysis", ""),
        }

        # 导出
        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_basename = os.path.join(args.output_dir, f"report_{ts}")

        if args.output_format == "pdf":
            output_path = output_basename + ".pdf"
            result = generate_pdf.invoke({
                "report_dict": final_report, "output_path": output_path
            })
        else:
            output_path = output_basename + ".md"
            result = generate_markdown.invoke({
                "report_dict": final_report, "output_path": output_path
            })

        print(f"\n📄 {result}")

        # 若工具发生降级（PDF -> Markdown），从返回信息中提取真实输出路径
        real_output_path = output_path
        m = re.search(r":\s*(.+)$", str(result))
        if m:
            cand = m.group(1).strip()
            if cand:
                real_output_path = cand

        # 保存长期记忆
        try:
            summary = final_report["sections"][0].get("content", "")[:500]
            long_mem.store_report(
                topic=final_report["meta"].get("title", "未知实验"),
                report_summary=summary,
                metadata={
                    "output_format": args.output_format,
                    "generated_at": datetime.now().isoformat(),
                }
            )
            print(f"💾 已保存到长期记忆")
        except Exception as e:
            print(f"⚠️ 长期记忆保存: {e}")

        print(f"\n{'=' * 60}")
        print(f"  报告: {real_output_path}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
