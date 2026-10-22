"""
LangGraph 实验报告生成图（v3 - 决策节点只初始化一次）

核心规则:
- 每个决策节点首次访问时调用 Agent.decompose() 生成计划
- 后续再进入时**直接跳过**，不重新分解
- 执行节点完成任务后标记 done，路由回决策节点
- 路由函数是纯 State 检查
"""
from typing import Dict, Any
from datetime import datetime
import os
import re
import glob

from langgraph.graph import StateGraph, END
from config import config


def _task(task_id, desc, task_type, agent):
    return {"task_id": task_id, "description": desc, "task_type": task_type,
            "assigned_agent": agent, "dependencies": [], "context": {}, "status": "pending"}

def _all_done(plan_key, state):
    return not any(t.get("status") == "pending" for t in state.get(plan_key, []))

def _done_one(plan_key, state):
    for t in state.get(plan_key, []):
        if t.get("status") == "pending":
            t["status"] = "done"; break

def _done_type(plan_key, task_type, state):
    """按 task_type 精确标记 done"""
    for t in state.get(plan_key, []):
        if t.get("task_type") == task_type and t.get("status") == "pending":
            t["status"] = "done"; return

def _next_type(plan_key, state):
    for t in state.get(plan_key, []):
        if t.get("status") == "pending":
            return t.get("task_type", ""), t.get("description", "")
    return "", ""


# ================================================================
# 阶段一节点
# ================================================================

def n_root(state):
    """Root: 首次进入时分解为两阶段plan；后续进入跳过分解，由 r_root 根据 pending 路由"""
    if state.get("_root_plan"):
        return state

    from agents import RootPlanner
    raw = state.get("raw_input", {})
    prompt = raw.get("prompt", "")

    # 让千问分析用户输入
    task_desc = f"分析以下用户需求并生成两阶段计划: {prompt[:300]}"
    sub = RootPlanner.decompose(task_desc, state)
    if not sub:
        sub = [
            _task("P1", "解析用户输入", "input_resolve", "InputResolveDecision"),
            _task("P2", "生成实验报告", "report_generation", "ReportGenDecision"),
        ]
    state["_root_plan"] = sub
    state.setdefault("execution_log", []).append(f"[Root] {len(sub)}阶段计划")
    return state


def _normalize_paths(items):
    """规范化路径列表：去空、去引号、统一分隔符、去重并保序。"""
    if not items:
        return []
    if isinstance(items, str):
        items = [items]

    out = []
    seen = set()
    for x in items:
        p = str(x or "").strip().strip('"').strip("'")
        # 清理提示词中路径后常见的中文/英文尾随标点
        p = p.rstrip("，。；;：:！？!?、）)]】}>\n\r\t ")
        if not p:
            continue
        p = p.replace("\\", "/")
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _extract_abs_paths_from_prompt(prompt: str):
    text = str(prompt or "")
    hits = re.findall(r"[A-Za-z]:\\[^\s\",，。；;：:！？!?）)\]】>]+", text)
    return _normalize_paths(hits)


def _path_in_nested_command(prompt: str, path: str) -> bool:
    """判断路径是否位于“测试命令/示例命令”等嵌套命令片段中。"""
    text = str(prompt or "")
    p = str(path or "")
    if not text or not p:
        return False

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

    between = text[marker_idx:path_idx]
    reset_markers = ["当前项目材料", "当前实验结果图", "本项目结果图"]
    return not any(x in between for x in reset_markers)


def _mark_path_context(prompt: str, paths) -> dict:
    """给路径打来源标记；不删除路径，只标注是否来自嵌套命令。"""
    out = {}
    for p in _normalize_paths(paths or []):
        out[p] = "nested_command" if _path_in_nested_command(prompt, p) else "current_material"
    return out


def _is_current_material_path(raw_input: dict, path: str) -> bool:
    """路径是否可作为当前报告材料使用。"""
    ctx = raw_input.get("path_context") or {}
    p = _normalize_paths([path])
    if not p:
        return True
    p = p[0]
    variants = {p, p.replace("/", "\\"), p.replace("\\", "/")}
    for k, v in ctx.items():
        nk = _normalize_paths([k])
        if nk and nk[0] in variants:
            return v != "nested_command"
    return not _path_in_nested_command(str(raw_input.get("prompt", "")), p)


def _recover_path(path: str, roots):
    p = _normalize_paths([path])
    if not p:
        return ""
    p = p[0]
    if os.path.exists(p):
        return p
    base = os.path.basename(p)
    if not base:
        return ""
    for r in roots or []:
        rr = _normalize_paths([r])
        if not rr:
            continue
        root = rr[0]
        if not os.path.isdir(root):
            continue
        try:
            matches = glob.glob(os.path.join(root, "**", base), recursive=True)
            for m in matches:
                if os.path.exists(m):
                    return m.replace("\\", "/")
        except Exception:
            continue
    return ""


CODE_EXTS = {
    ".py", ".ipynb", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
    ".h", ".hpp", ".go", ".rs", ".sh", ".ps1", ".sql", ".yaml", ".yml",
    ".json", ".toml", ".ini", ".cfg", ".md",
}


def _path_has_code(path: str) -> bool:
    """判断路径是否是代码文件或包含代码文件的目录。"""
    p = _normalize_paths([path])
    if not p:
        return False
    p = p[0]
    if os.path.isfile(p):
        return os.path.splitext(p)[1].lower() in CODE_EXTS
    if not os.path.isdir(p):
        return False
    for cur_root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        if any(os.path.splitext(f)[1].lower() in CODE_EXTS for f in files):
            return True
    return False


def _collect_valid_code_paths(raw_input: dict, parsed: dict) -> list[str]:
    """从 CLI 与提示词解析结果中收集存在且相关的代码路径。"""
    prompt = str(raw_input.get("prompt", ""))
    candidates = []
    if raw_input.get("code_file"):
        candidates.append(str(raw_input.get("code_file")))
    candidates.extend(_normalize_paths((parsed or {}).get("code_paths") or []))

    prompt_paths = _extract_abs_paths_from_prompt(prompt)
    roots = [os.path.dirname(p) for p in prompt_paths if p]
    roots = [r for r in roots if r]

    out = []
    seen = set()
    for item in candidates:
        p = _normalize_paths([item])
        if not p:
            continue
        p = p[0]
        if _path_in_nested_command(prompt, p):
            continue
        if not os.path.exists(p):
            recovered = _recover_path(p, roots)
            if recovered:
                p = recovered
        if _path_in_nested_command(prompt, p):
            continue
        if not _path_has_code(p):
            continue
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        seen.add(key)
        out.append(p.replace("\\", "/"))
    return out


PROMPT_REQUIRED_KEYS = [
    "topic", "requirements", "student_name", "student_id", "experiment_purpose",
    "theory_background", "output_format", "file_paths", "data_files", "code_paths",
    "normalized_paths", "missing_fields", "path_evidence",
]


def _normalize_text_list(items):
    if not items:
        return []
    if isinstance(items, str):
        return [items.strip()] if items.strip() else []
    out = []
    for x in items:
        s = str(x or "").strip()
        if s:
            out.append(s)
    return out


def _is_prompt_schema_valid(parsed):
    return isinstance(parsed, dict) and all(k in parsed for k in PROMPT_REQUIRED_KEYS)


def _sanitize_prompt_fields(parsed):
    data = parsed if isinstance(parsed, dict) else {}
    clean = {
        "topic": str(data.get("topic") or "").strip(),
        "requirements": str(data.get("requirements") or "").strip(),
        "student_name": str(data.get("student_name") or "").strip(),
        "student_id": str(data.get("student_id") or "").strip(),
        "experiment_purpose": str(data.get("experiment_purpose") or "").strip(),
        "theory_background": str(data.get("theory_background") or "").strip(),
        "output_format": str(data.get("output_format") or "").strip(),
        "file_paths": _normalize_paths(data.get("file_paths") or []),
        "data_files": _normalize_paths(data.get("data_files") or []),
        "code_paths": _normalize_paths(data.get("code_paths") or []),
        "normalized_paths": _normalize_paths(data.get("normalized_paths") or []),
        "missing_fields": _normalize_text_list(data.get("missing_fields") or []),
        "path_evidence": _normalize_text_list(data.get("path_evidence") or []),
    }

    if not clean["normalized_paths"]:
        all_paths = clean["file_paths"] + clean["data_files"] + clean["code_paths"]
        clean["normalized_paths"] = _normalize_paths(all_paths)

    for key in PROMPT_REQUIRED_KEYS:
        clean.setdefault(key, [] if key.endswith("_paths") or key in {"missing_fields", "path_evidence"} else "")
    return clean


def _search_short_term_memory(state, query: str, top_k: int = 3):
    """在 state.memory.short_term 中做轻量关键词检索。"""
    records = ((state.get("memory") or {}).get("short_term") or [])
    if not records:
        return []

    q = str(query or "").strip().lower()
    if not q:
        return []

    import re
    tokens = [t for t in re.split(r"\s+|[,，。；;:：/\\]+", q) if len(t) >= 2]
    if not tokens:
        tokens = [q]

    hits = []
    for r in records:
        content = str(r.get("content", ""))
        content_lower = content.lower()
        agent_name = str(r.get("agent_name", ""))
        role = str(r.get("role", ""))

        score = 0
        if q in content_lower:
            score += 2
        score += sum(1 for t in tokens if t in content_lower)

        if score > 0:
            hits.append({
                "score": score,
                "text": f"[{agent_name}|{role}] {content[:300]}"
            })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return [h["text"] for h in hits[:top_k]]


def _sync_short_term_memory(state, agent_name: str, content: str,
                            role: str = "agent", metadata: Dict[str, Any] = None):
    """把节点产出同步到 state.memory.short_term（滑动窗口）。"""
    if not content:
        return

    memory = state.setdefault("memory", {"short_term": [], "long_term_keys": []})
    short_term = memory.setdefault("short_term", [])

    short_term.append({
        "role": role,
        "agent_name": agent_name,
        "content": str(content),
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
    })

    max_size = getattr(config, "short_term_window", 10)
    if len(short_term) > max_size:
        memory["short_term"] = short_term[-max_size:]


def n_input_resolve(state):
    """InputResolve: 首次分解plan，后续跳过分解，由 r_input 路由"""
    if state.get("_input_plan"):
        return state

    from agents import InputResolveDecision
    raw = state.get("raw_input", {})
    prompt = raw.get("prompt", "")

    refs = _normalize_paths(raw.get("reference_materials") or [])
    code_file = str(raw.get("code_file") or "").strip()
    task_desc = (
        f"解析用户输入。已给附件: {refs[:5]}，已给代码路径: {code_file or '无'}。"
        f"请优先从提示词中抽取 file_paths/code_paths 并决定是否触发 file_parse/code_read。"
        f"提示词: {prompt[:240]}"
    )
    sub = InputResolveDecision.decompose(task_desc, state)

    if not sub:
        sub = [
            _task("PP", "提取提示词中的实验题目、姓名等", "prompt_parse", "PromptParserAgent"),
            _task("MR", "检索历史报告和用户偏好", "memory_retrieve", "MemoryRetrieverAgent"),
            _task("RI", "检索课件理论知识", "rag_retrieve", "RAGExecutorAgent"),
        ]

    # 不在图层强制注入 file_parse/code_read，由 PromptParserAgent 输出驱动
    raw["reference_materials"] = refs

    state["_input_plan"] = sub
    state.setdefault("execution_log", []).append(
        f"[InResolve] {len(sub)}子任务（智能体驱动路径抽取）"
    )
    return state


def n_prompt_parser(state):
    from agents import PromptParserAgent
    prompt = state.get("raw_input", {}).get("prompt", "")

    # 使用智能 prompt 让千问提取所有字段，包括文件路径
    plan = [_task("PP", f"""从以下提示词中提取结构化字段，并严格输出JSON:
提示词: {prompt}

要求:
1) 只提取提示词中明确出现的路径，禁止臆造路径。
2) 同时提取 file_paths / data_files / code_paths。
3) 每个路径在 path_evidence 中写明来源片段。
4) 输出必须包含以下全部键（不存在时给空字符串或空数组，不要省略）:
   topic, requirements, student_name, student_id, experiment_purpose,
   theory_background, output_format, file_paths, data_files, code_paths,
   normalized_paths, missing_fields, path_evidence""", "prompt_parse", "PromptParserAgent")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = PromptParserAgent.process(state)
    raw_output = result.get("_PromptParserAgent_result", str(result))
    # 解析 JSON 字符串为 dict
    parsed = _parse_json_from_agent(raw_output)

    # 如果 schema 不完整，自动重试一次并要求仅输出严格 JSON
    if not _is_prompt_schema_valid(parsed):
        retry_plan = [_task("PP_RETRY", f"""上一次输出格式不完整。请仅输出严格JSON，必须包含全部键:
{', '.join(PROMPT_REQUIRED_KEYS)}

提示词: {prompt}
上一次输出(截断): {str(raw_output)[:500]}""", "prompt_parse", "PromptParserAgent")]
        state["plan"] = retry_plan
        state["current_task_index"] = 0
        retry_result = PromptParserAgent.process(state)
        retry_raw = retry_result.get("_PromptParserAgent_result", str(retry_result))
        retry_parsed = _parse_json_from_agent(retry_raw)
        if _is_prompt_schema_valid(retry_parsed):
            parsed = retry_parsed
            state.setdefault("execution_log", []).append("[PromptParser] 首次格式异常，重试后成功")
        else:
            state.setdefault("execution_log", []).append("[PromptParser] 首次和重试均格式异常，执行本地字段补齐")

    parsed = _sanitize_prompt_fields(parsed)

    # 由 PromptParserAgent 主导提取路径，并回写到 raw_input 供后续节点使用
    raw_input = state.get("raw_input", {})
    prompt_text = str(raw_input.get("prompt", ""))
    cli_refs = _normalize_paths(raw_input.get("reference_materials") or [])
    parsed_refs = _normalize_paths(parsed.get("file_paths") or [])
    merged_refs = cli_refs + [p for p in parsed_refs if p.lower() not in {x.lower() for x in cli_refs}]
    raw_input["reference_materials"] = merged_refs
    raw_input["path_context"] = _mark_path_context(prompt_text, merged_refs + _normalize_paths(parsed.get("code_paths") or []))

    code_candidates = _collect_valid_code_paths(raw_input, parsed)
    if code_candidates:
        raw_input["code_file"] = code_candidates[0]
        raw_input["code_paths"] = code_candidates

    # 智能体抽取结果驱动 _input_plan：按需激活 file_parse/code_read
    input_plan = state.get("_input_plan", [])
    has_file_task = any(t.get("task_type") == "file_parse" for t in input_plan)
    has_code_task = any(t.get("task_type") == "code_read" for t in input_plan)
    if merged_refs and not has_file_task:
        input_plan.append(_task("FP", f"解析{len(merged_refs)}个附件", "file_parse", "FileParserAgent"))
    if raw_input.get("code_file") and not has_code_task:
        input_plan.append(_task("CR", f"读取项目代码: {raw_input['code_file']}", "code_read", "CodeReaderAgent"))

    state["raw_input"] = raw_input
    state["_input_plan"] = input_plan
    state["parsed_prompt_fields"] = parsed
    _sync_short_term_memory(
        state,
        "PromptParserAgent",
        str(parsed),
        role="execution",
        metadata={"task_type": "prompt_parse"}
    )
    state.setdefault("execution_log", []).append(
        f"[PromptParser] 提取路径 file={len(merged_refs)} code={'1' if raw_input.get('code_file') else '0'}"
    )
    _done_one("_input_plan", state)
    return state


def _parse_json_from_agent(raw: str) -> dict:
    """从 Agent 返回中提取 JSON dict（兼容各种格式）"""
    if isinstance(raw, dict):
        return raw
    import json
    s = str(raw)
    # 提取 {...} 部分
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def n_file_parser(state):
    from agents import FileParserAgent
    # 直接从 raw_input 读取（已在 n_input_resolve 中完成拼接）
    refs = state.get("raw_input", {}).get("reference_materials") or []
    refs = [p for p in refs if _is_current_material_path(state.get("raw_input", {}), p)]
    if not refs:
        _done_one("_input_plan", state); return state

    plan = [_task("FP", f"解析文件: {', '.join(refs[:5])}", "file_parse", "FileParserAgent")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = FileParserAgent.process(state)
    content = result.get("_FileParserAgent_result", str(result))
    state["parsed_file_contents"] = state.get("parsed_file_contents", []) + [content]
    _sync_short_term_memory(
        state,
        "FileParserAgent",
        content,
        role="execution",
        metadata={"task_type": "file_parse"}
    )
    _done_one("_input_plan", state)
    return state


def n_memory_retriever(state):
    # 先查短时记忆；命中则跳过长时检索
    query = state.get("resolved_input", {}).get("topic") or state.get("raw_input", {}).get("prompt", "")
    short_hits = _search_short_term_memory(state, query, top_k=3)
    if short_hits:
        short_text = "[短期记忆命中]\n" + "\n".join(f"- {x}" for x in short_hits)
        state["_MemoryRetrieverAgent_result"] = short_text
        state["memory_retrieval_results"] = state.get("memory_retrieval_results", []) + [short_text]
        _sync_short_term_memory(
            state,
            "MemoryRetrieverAgent",
            short_text,
            role="execution",
            metadata={"task_type": "memory_retrieve", "source": "short_term"}
        )
        state.setdefault("execution_log", []).append(
            f"[MemoryRetriever] 命中短时记忆 {len(short_hits)} 条，跳过长时检索"
        )
        _done_one("_input_plan", state)
        return state

    from agents import MemoryRetrieverAgent
    plan = [_task("MR", "检索历史报告和用户偏好", "memory_retrieve", "MemoryRetrieverAgent")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = MemoryRetrieverAgent.process(state)
    mem_text = result.get("_MemoryRetrieverAgent_result", str(result))
    state["memory_retrieval_results"] = state.get("memory_retrieval_results", []) + [mem_text]
    _sync_short_term_memory(
        state,
        "MemoryRetrieverAgent",
        mem_text,
        role="execution",
        metadata={"task_type": "memory_retrieve", "source": "long_term"}
    )
    state.setdefault("execution_log", []).append("[MemoryRetriever] 短时未命中，已执行长时检索")
    _done_one("_input_plan", state)
    return state


def n_rag_input(state):
    from agents import RAGExecutorAgent
    resolved = state.get("resolved_input", {})
    prompt = state.get("raw_input", {}).get("prompt", "")[:80]
    topic = resolved.get("topic") or prompt
    plan = [_task("RI", f"检索课件中关于 {topic} 的理论知识", "rag_retrieve", "RAGExecutorAgent")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = RAGExecutorAgent.process(state)
    content = result.get("_RAGExecutorAgent_result", str(result))
    state["rag_input_results"] = state.get("rag_input_results", []) + [content]
    _sync_short_term_memory(
        state,
        "RAGExecutorAgent",
        content,
        role="execution",
        metadata={"task_type": "rag_retrieve"}
    )
    _done_one("_input_plan", state)
    return state


def n_code_reader(state):
    """读取用户代码文件（条件激活）"""
    from agents import CodeReaderAgent
    from tools import read_code
    raw = state.get("raw_input", {})
    code_paths = raw.get("code_paths") or []
    if isinstance(code_paths, str):
        code_paths = [code_paths]
    if raw.get("code_file") and raw.get("code_file") not in code_paths:
        code_paths.insert(0, raw.get("code_file"))
    code_paths = _collect_valid_code_paths(raw, {"code_paths": code_paths})
    if not code_paths:
        _done_one("_input_plan", state); return state

    # 优先走确定性工具调用，避免 LLM 在未读代码时“补写模板代码”。
    contents = []
    errors = []
    for code_path in code_paths:
        try:
            part = str(read_code.invoke({"path": code_path}))
            if part.startswith("[Error]"):
                errors.append(part)
            else:
                contents.append(part)
        except Exception as e:
            errors.append(f"[Error] 读取代码失败 {code_path}: {e}")

    if contents:
        content = "\n\n".join(contents)
        if errors:
            state.setdefault("execution_log", []).append(
                f"[CodeReader] 忽略无效代码路径 {len(errors)} 个"
            )
        state["_CodeReaderAgent_result"] = content
        state.setdefault("execution_log", []).append(
            f"[CodeReader] 直接调用 read_code 成功，读取 {len(contents)} 个路径"
        )
    else:
        state.setdefault("execution_log", []).append("[CodeReader] 直接调用失败，回退Agent")
        plan = [_task("CR", f"读取项目代码: {', '.join(code_paths[:5])}", "code_read", "CodeReaderAgent")]
        state["plan"] = plan; state["current_task_index"] = 0
        result = CodeReaderAgent.process(state)
        content = result.get("_CodeReaderAgent_result", "\n".join(errors) or str(result))

    state["parsed_file_contents"] = state.get("parsed_file_contents", []) + [content]
    _sync_short_term_memory(
        state,
        "CodeReaderAgent",
        content,
        role="execution",
        metadata={"task_type": "code_read", "paths": code_paths}
    )
    _done_one("_input_plan", state)
    return state


def _clip_text(text: str, max_chars: int) -> str:
    s = str(text or "")
    return s if len(s) <= max_chars else (s[:max_chars] + "\n...[已截断]")


def _build_stage1_context(state: Dict[str, Any], file_chars: int = 5000) -> str:
    """汇总阶段一全部关键产出，供阶段二统一引用。"""
    parsed = state.get("parsed_prompt_fields", {})
    files = state.get("parsed_file_contents", [])
    mems = state.get("memory_retrieval_results", [])
    rag_inputs = state.get("rag_input_results", [])

    files_text = _clip_text("\n\n".join(str(x) for x in files if x), file_chars)
    mem_text = _clip_text("\n\n".join(str(x) for x in mems if x), 2000)
    rag_text = _clip_text("\n\n".join(str(x) for x in rag_inputs if x), 2000)

    return (
        "[阶段一产出汇总]\n"
        f"1) 提示词结构化字段(parsed_prompt_fields):\n{_clip_text(parsed, 1500)}\n\n"
        f"2) 文件/代码解析结果(parsed_file_contents):\n{files_text or '无'}\n\n"
        f"3) 记忆检索(memory_retrieval_results):\n{mem_text or '无'}\n\n"
        f"4) 输入阶段RAG结果(rag_input_results):\n{rag_text or '无'}"
    )


# ================================================================
# 阶段二节点
# ================================================================

def n_report_gen(state):
    if state.get("_report_plan"):
        return state

    from agents import ReportGenDecision
    resolved = state.get("resolved_input", {})
    stage1_ctx = _build_stage1_context(state, file_chars=3000)
    task_desc = (
        f"生成 {resolved.get('topic', '实验')} 报告。\n"
        f"请严格基于以下阶段一产出进行阶段二分解：\n{stage1_ctx}"
    )
    sub = ReportGenDecision.decompose(task_desc, state)
    if not sub:
        sub = [
            _task("TH", "检索理论知识", "theory_decomposition", "TheoryDecision"),
            _task("WR", "撰写报告", "writing_decomposition", "WriterDecision"),
        ]
        if state.get("raw_input", {}).get("data_files") or state.get("raw_input", {}).get("code_file"):
            sub.insert(1, _task("ME", "分析方法/代码", "method_decomposition", "MethodDecision"))
    state["_report_plan"] = sub
    state["status"] = "executing"
    return state


def n_theory(state):
    if state.get("_theory_plan"):
        return state
    from agents import TheoryDecision
    stage1_ctx = _build_stage1_context(state, file_chars=2000)
    sub = TheoryDecision.decompose(f"检索理论知识。阶段一依据如下：\n{stage1_ctx}", state)
    if not sub:
        sub = [_task("TH1", "RAG检索理论知识", "theory_retrieval", "RAGExecutor")]
    state["_theory_plan"] = sub
    return state


def n_method(state):
    if state.get("_method_plan"):
        return state
    from agents import MethodDecision
    stage1_ctx = _build_stage1_context(state, file_chars=4000)
    sub = MethodDecision.decompose(f"分析实验方法。阶段一依据如下：\n{stage1_ctx}", state)
    if not sub:
        sub = []
        raw = state.get("raw_input", {})
        if raw.get("data_files"):
            sub.append(_task("ME1", "分析数据", "data_analysis", "DataExecutor"))
        if raw.get("code_file"):
            sub.append(_task("ME2", "执行代码", "code_execution", "CodeExecutor"))
        refs = _normalize_paths(raw.get("reference_materials") or [])
        image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff")
        if any(str(x).lower().endswith(image_exts) for x in refs):
            sub.append(_task("ME3", "分析实验结果图片并给出插图位置建议", "image_analysis", "ImageInterpreter"))
    state["_method_plan"] = sub
    return state


def n_writer(state):
    if state.get("_writer_plan"):
        # plan 已存在，跳过分解（Reflection 重入也不重新分解）
        return state
    from agents import WriterDecision
    stage1_ctx = _build_stage1_context(state, file_chars=3000)
    sub = WriterDecision.decompose(f"撰写和排版实验报告。阶段一依据如下：\n{stage1_ctx}", state)
    if not sub:
        sub = [_task("WR1", "撰写报告", "section_writing", "SectionWriter"),
               _task("WR2", "检查格式", "format_check", "FormatChecker")]
    state["_writer_plan"] = sub
    return state


# ================================================================
# 阶段二执行节点
# ================================================================

def n_rag_exec(state):
    from agents import RAGExecutor
    resolved = state.get("resolved_input", {})
    topic = resolved.get("topic", "")
    bg = resolved.get("theory_background", "")
    stage1_rag = "\n".join(state.get("rag_input_results", []))
    query_hint = f"{topic} {bg} {stage1_rag[:300]}".strip() or "相关知识"
    plan = [_task("RE", f"检索课件中关于 {query_hint} 的理论知识（请搜索 {query_hint[:100]} 相关内容）", "theory_retrieval", "RAGExecutor")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = RAGExecutor.process(state)
    content = result.get("_RAGExecutor_result", str(result))
    state["theory_content"] = (state.get("theory_content", "") + "\n" + content).strip()
    _done_one("_theory_plan", state)
    return state


def n_search_exec(state):
    from agents import SearchExecutor
    plan = [_task("SE", "搜索最新资料", "theory_retrieval", "SearchExecutor")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = SearchExecutor.process(state)
    content = result.get("_SearchExecutor_result", str(result))
    state["theory_content"] = (state.get("theory_content", "") + "\n" + content).strip()
    _done_one("_theory_plan", state)
    return state


def n_data_exec(state):
    from agents import DataExecutor
    stage1_ctx = _build_stage1_context(state, file_chars=2000)
    plan = [_task("DE", f"分析实验数据。请基于阶段一产出执行：\n{stage1_ctx}", "data_analysis", "DataExecutor")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = DataExecutor.process(state)
    content = result.get("_DataExecutor_result", str(result))
    state["method_content"] = (state.get("method_content", "") + "\n" + content).strip()
    _done_one("_method_plan", state)
    return state


def n_code_exec(state):
    from agents import CodeExecutor
    code = state.get("raw_input", {}).get("code_file", "")
    stage1_ctx = _build_stage1_context(state, file_chars=6000)
    desc = f"执行代码: {code}" if code else "执行代码"
    plan = [_task("CE", f"{desc}\n请结合阶段一产出：\n{stage1_ctx}", "code_execution", "CodeExecutor")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = CodeExecutor.process(state)
    content = result.get("_CodeExecutor_result", str(result))
    state["method_content"] = (state.get("method_content", "") + "\n" + content).strip()
    _done_one("_method_plan", state)
    return state


def n_image_exec(state):
    from agents import ImageInterpreter
    from tools import parse_file, analyze_image_vlm
    import json
    raw = state.get("raw_input", {})
    refs = _normalize_paths(raw.get("reference_materials") or [])
    image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff")
    prompt_text = str(raw.get("prompt", ""))
    prompt_paths = _extract_abs_paths_from_prompt(prompt_text)
    roots = [os.path.dirname(p) for p in (prompt_paths + refs) if p]
    images = []

    image_candidates = [(x, "explicit_reference") for x in refs]
    image_candidates.extend((x, "prompt_extracted") for x in prompt_paths)

    for x, source in image_candidates:
        sx = str(x).replace("\\", "/")
        if not sx.lower().endswith(image_exts):
            continue
        if not _is_current_material_path(raw, sx):
            continue
        if not os.path.exists(sx):
            rec = _recover_path(sx, roots)
            if rec:
                sx = rec
        if not _is_current_material_path(raw, sx):
            continue
        if os.path.exists(sx) and sx not in images:
            images.append(sx)
    if not images:
        _done_one("_method_plan", state)
        return state

    def _summarize_image_text(raw_text: str) -> str:
        s = str(raw_text or "").strip()
        if not s:
            return "检测到实验结果图，等待视觉模型分析。"
        lines = []
        for ln in s.splitlines():
            t = ln.strip()
            if not t:
                continue
            if t.startswith("[图片") or t.startswith("[DOCX"):
                continue
            lines.append(t)
            if len(lines) >= 3:
                break
        if not lines:
            return "检测到实验结果图，等待视觉模型分析。"
        out = "；".join(lines)
        if len(out) > 140:
            out = out[:140] + "..."
        return out

    # 先做确定性本地解析，作为稳定兜底
    fallback = {"images": []}
    stage1_ctx = _build_stage1_context(state, file_chars=2000)
    resolved = state.get("resolved_input", {}) or {}
    parsed = state.get("parsed_prompt_fields", {}) or {}
    prompt_text = str(raw.get("prompt", ""))
    explicit_embed = any(x in prompt_text for x in ["嵌入图片", "插入图片", "把图片", "放进报告", "嵌入报告", "分析并嵌入"])
    background_ctx = (
        "实验背景与方法摘要:\n"
        f"- 实验题目: {resolved.get('topic') or parsed.get('topic') or raw.get('prompt', '')[:80]}\n"
        f"- 实验要求: {resolved.get('requirements') or parsed.get('requirements') or raw.get('prompt', '')[:200]}\n"
        f"- 实验目的: {resolved.get('experiment_purpose') or parsed.get('experiment_purpose') or ''}\n"
        f"- 理论背景: {resolved.get('theory_background') or parsed.get('theory_background') or ''}\n"
        f"- 方法线索: {state.get('method_content', '')[:1200]}\n"
        f"- 用户是否明确要求嵌入图片: {explicit_embed}\n"
        f"- 代码/文件证据: {stage1_ctx[:1200]}\n"
    )

    def _clean_vision_text(text: str) -> str:
        """清理视觉模型输出，避免把提示词/上下文原文写进报告。"""
        s = str(text or "").strip()
        if not s:
            return ""
        banned_prefixes = [
            "实验背景与方法摘要", "图片列表", "请结合上下文", "必须输出严格 JSON",
            "嵌图决策规则", "如果图片是图表", "规则：", "上下文："
        ]
        lines = []
        for line in s.splitlines():
            t = line.strip()
            if not t:
                continue
            if any(t.startswith(p) for p in banned_prefixes):
                continue
            # 不允许把整段文件证据/代码证据复制进报告
            if "阶段一产出汇总" in t or "parsed_prompt_fields" in t or "代码/文件证据" in t:
                continue
            lines.append(t)
        out = "；".join(lines) if lines else s
        return out[:500] + "..." if len(out) > 500 else out

    def _sanitize_vision_obj(obj: dict) -> dict:
        if not isinstance(obj, dict):
            return obj
        for item in obj.get("images", []) if isinstance(obj.get("images", []), list) else []:
            if not isinstance(item, dict):
                continue
            for key in ("summary", "trend_analysis", "report_integration", "embed_reason", "placement_hint"):
                item[key] = _clean_vision_text(item.get(key, ""))
            if isinstance(item.get("evidence"), list):
                item["evidence"] = [_clean_vision_text(x) for x in item["evidence"] if _clean_vision_text(x)]
            if isinstance(item.get("key_points"), list):
                for kp in item["key_points"]:
                    if isinstance(kp, dict):
                        kp["name"] = _clean_vision_text(kp.get("name", ""))
                        kp["value"] = _clean_vision_text(kp.get("value", ""))
                        kp["meaning"] = _clean_vision_text(kp.get("meaning", ""))
        return obj

    for p in images:
        try:
            raw_txt = parse_file.func(p)
            summary = _summarize_image_text(raw_txt)
            fallback["images"].append({
                "path": p,
                "summary": summary,
                "trend_analysis": "检测到实验结果图。若为训练曲线，请结合曲线的初始阶段、中段波动、后段平台和末尾表现分析整体收敛趋势；具体数值以图中可读刻度为准。",
                "report_integration": "这张图应作为实验结果与分析的核心证据，结合训练方法说明曲线变化与 Double DQN / 经验回放 / 目标网络的作用关系。",
                "should_embed": True if explicit_embed else True,
                "embed_reason": "用户明确要求嵌入图片" if explicit_embed else "图片可支撑实验结果与趋势分析",
                "key_points": [
                    {"name": "初始阶段", "value": "无法精确读取", "meaning": "用于判断模型起始表现"},
                    {"name": "中后期/末尾", "value": "无法精确读取", "meaning": "用于判断是否收敛和是否稳定"},
                ],
                "evidence": ["来源: 本地 parse_file 解析"],
                "recommended_section": "实验结果与分析",
                "placement_hint": "放在展示训练曲线与指标分析的段落后",
                "keywords": ["结果", "曲线", "奖励", "收敛", "loss", "reward"],
            })
        except Exception:
            fallback["images"].append({
                "path": p,
                "summary": "检测到实验结果图，文字识别有限。",
                "trend_analysis": "检测到实验结果图，文字识别有限；若图中包含曲线，应描述整体走势、波动区间、是否趋于稳定以及末尾表现。",
                "report_integration": "这张图应放入实验结果部分，作为训练效果的可视化证据，并与方法章节中的训练策略对应解释。",
                "should_embed": True if explicit_embed else True,
                "embed_reason": "用户明确要求嵌入图片" if explicit_embed else "图片可支撑实验结果与趋势分析",
                "key_points": [
                    {"name": "起点", "value": "无法精确读取", "meaning": "初始性能"},
                    {"name": "末尾", "value": "无法精确读取", "meaning": "最终性能和稳定性"},
                ],
                "evidence": ["来源: 本地解析异常，使用保守摘要"],
                "recommended_section": "实验结果与分析",
                "placement_hint": "放在结果图讨论段落后",
                "keywords": ["结果", "曲线", "分析"],
            })

    # 优先直接调用 Qwen2.5-VL，避免只依赖 OCR 导致曲线图理解失败
    vlm_obj = None
    try:
        vlm_context = background_ctx + "\n" + stage1_ctx[:1600]
        vlm_raw = analyze_image_vlm.invoke({"filepath": images[0], "context": vlm_context[:2500]})
        vlm_parsed = _parse_json_from_agent(vlm_raw)
        if isinstance(vlm_parsed, dict) and vlm_parsed.get("images"):
            vlm_obj = _sanitize_vision_obj(vlm_parsed)
    except Exception:
        vlm_obj = None

    desc = (
        "读取并理解实验结果图片，输出图片摘要、关键节点值、趋势分析与推荐插入位置。"
        f"\n图片列表: {images}\n"
        f"\n{background_ctx}\n"
        "嵌图决策规则：如果用户提示词明确要求嵌入图片，则 should_embed 必须为 true；如果用户未要求，则自行判断图片是否支撑实验结果分析。\n"
        "如果图片是图表/曲线图，必须分析起点、峰值/谷值、拐点、平台期、末尾值、波动区间和是否收敛；"
        "数值看不清时不要编造精确值，用约略或无法精确读取表述，但必须保留趋势判断。\n"
        f"请结合上下文: \n{stage1_ctx[:3000]}"
    )
    plan = [_task("IE", desc, "image_analysis", "ImageInterpreter")]
    state["plan"] = plan
    state["current_task_index"] = 0
    if vlm_obj:
        parsed = vlm_obj
        content = json.dumps(vlm_obj, ensure_ascii=False)
    else:
        result = ImageInterpreter.process(state)
        content = result.get("_ImageInterpreter_result", str(result))
        parsed = _parse_json_from_agent(content)

    # 过滤明显不稳定/误报内容，优先采用本地兜底结构化结果
    bad_text = str(content or "")
    unreliable = any(x in bad_text for x in ["文件未找到", "无法读取图片", "无法解析内容", "路径有误", "被移动", "被删除"])
    if isinstance(parsed, dict) and parsed.get("images") and not unreliable:
        final_obj = _sanitize_vision_obj(parsed)
    else:
        final_obj = _sanitize_vision_obj(fallback)

    state["image_analysis"] = json.dumps(final_obj, ensure_ascii=False)
    vision_summary_lines = []
    for item in final_obj.get("images", []):
        vision_summary_lines.append(
            f"【视觉分析】{item.get('path', '')}\n"
            f"- 摘要: {item.get('summary', '')}\n"
            f"- 图表类型: {item.get('chart_type', '未知')}\n"
            f"- 趋势分析: {item.get('trend_analysis', '')}\n"
            f"- 报告解读: {item.get('report_integration', '')}\n"
            f"- 是否嵌入: {item.get('should_embed', True)}，原因: {item.get('embed_reason', '')}\n"
            f"- 关键节点: {item.get('key_points', [])}\n"
            f"- 建议插入位置: {item.get('recommended_section', '')} / {item.get('placement_hint', '')}"
        )
    state.setdefault("vision_image_analysis", []).append(final_obj)
    state["method_content"] = (state.get("method_content", "") + "\n" + "\n".join(vision_summary_lines)).strip()
    _done_one("_method_plan", state)
    return state


def n_writer_exec(state):
    from agents import SectionWriter
    theory = state.get("theory_content", "")
    method = state.get("method_content", "")
    image_analysis = state.get("image_analysis", "")
    code_evidence_list = state.get("parsed_file_contents", [])
    code_evidence = "\n\n".join(str(x) for x in code_evidence_list if x).strip()
    stage1_ctx = _build_stage1_context(state, file_chars=8000)
    resolved = state.get("resolved_input", {})
    comments = state.get("review_comments", [])
    task_desc = (
        f"撰写 {resolved.get('topic', '实验')} 报告。\n"
        f"阶段一完整依据:\n{stage1_ctx}\n\n"
        f"理论依据:\n{theory[:2000]}\n\n"
        f"方法与实验记录:\n{method[:2000]}\n\n"
        f"图片分析结构化结果（若非空，图片相关描述必须以此为准）:\n{image_analysis[:2000]}\n\n"
        f"代码证据(来自read_code逐行读取):\n{code_evidence[:12000]}\n\n"
        "硬性要求:\n"
        "1) 代码实现章节必须仅基于上面“代码证据”撰写；\n"
        "2) 至少给出3段真实代码片段，并标注文件路径与行号；\n"
        "3) 若证据不足，不得编造，明确写“代码证据不足”；\n"
        "4) 禁止输出与证据无关的通用模板代码；\n"
        "5) 数学公式必须优先使用纯文本可读形式，不要输出LaTeX控制词；\n"
        "6) 若有图片分析，请在实验结果章节结合关键节点值说明大趋势（起点、波动、平台期、末尾表现、是否收敛），数值读不清时写“约略/无法精确读取”，但仍要给出趋势判断；\n"
        "7) 如果“图片分析结构化结果”非空且 should_embed=true，严禁写“图片文件未找到/无法获取图片/无法直接展示”等矛盾表述；\n"
        "8) 若阶段一文件解析中出现图片占位、图片不存在、OCR失败等文字，但图片分析结构化结果已给出，应以图片分析结构化结果为准。"
    )
    if comments:
        task_desc += "\n\n【Reflection修改意见】\n" + "\n".join(f"- {c}" for c in comments)
    plan = [_task("SW", task_desc, "section_writing", "SectionWriter")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = SectionWriter.process(state)
    content = result.get("_SectionWriter_result", str(result))
    state["draft_sections"] = {"main": content}
    _done_one("_writer_plan", state)
    return state


def n_format_check(state):
    from agents import FormatChecker
    draft = state.get("draft_sections", {})
    text = "\n".join(draft.values()) if draft else "无内容"
    plan = [_task("FC", f"检查报告:\n{text[:500]}", "format_check", "FormatChecker")]
    state["plan"] = plan; state["current_task_index"] = 0
    result = FormatChecker.process(state)
    state["_fc_result"] = result.get("_FormatChecker_result", str(result))
    _done_one("_writer_plan", state)
    return state


# ================================================================
# 合并辅助：收集子节点结果 → 调用决策节点的 merge() → 存入 state
# ================================================================

def _collect_and_merge(decision_agent, plan_key, state, result_key):
    """收集子节点结果，调用决策节点的 merge() 合并"""
    plan = state.get(plan_key, [])
    sub_results = {}
    for t in plan:
        agent_name = t.get("assigned_agent", "")
        key = f"_{agent_name}_result"
        if key in state:
            sub_results[agent_name] = str(state[key])
    if sub_results:
        merged = decision_agent.merge("合并子结果", sub_results)
        state[result_key] = merged
        state.setdefault("execution_log", []).append(
            f"[{decision_agent.name}] Merged {len(sub_results)} child results"
        )


def _build_memory_tool_summary(state: Dict[str, Any]) -> str:
    """生成运行时小结（仅用于日志输出，不参与任何节点输入）。"""
    short_mem = ((state.get("memory") or {}).get("short_term") or [])
    mem_logs = state.get("memory_retrieval_results") or []
    exec_logs = state.get("execution_log") or []

    # 记忆命中判定
    short_hit = any("[短期记忆命中]" in str(x) for x in mem_logs)
    long_used = any("长时检索" in str(x) or "长期记忆" in str(x) for x in mem_logs)

    # 基于各执行节点结果粗略统计工具调用成功（成功=结果存在且非Error）
    tool_agents = [
        "FileParserAgent", "MemoryRetrieverAgent", "RAGExecutorAgent",
        "CodeReaderAgent", "RAGExecutor", "SearchExecutor",
        "DataExecutor", "CodeExecutor",
    ]
    attempted = 0
    success = 0
    fail_agents = []
    for agent in tool_agents:
        key = f"_{agent}_result"
        if key not in state:
            continue
        attempted += 1
        text = str(state.get(key, ""))
        ok = ("[Error]" not in text) and ("检索失败" not in text) and ("执行失败" not in text)
        if ok:
            success += 1
        else:
            fail_agents.append(agent)

    # 直接工具调用的可见日志
    direct_code_read = any("[CodeReader] 直接调用 read_code 成功" in str(x) for x in exec_logs)

    lines = [
        "[运行小结] 记忆与工具使用",
        f"- 短时记忆条数: {len(short_mem)}",
        f"- 记忆检索: short_hit={short_hit}, long_term_used={long_used}",
        f"- 工具节点执行: success={success}/{attempted}",
        f"- CodeReader直连工具: {direct_code_read}",
    ]
    if fail_agents:
        lines.append(f"- 工具失败节点: {', '.join(fail_agents)}")
    return "\n".join(lines)


# ================================================================
# 路由（纯 State 检查）
# ================================================================

def r_root(state):
    if _all_done("_root_plan", state):
        from agents import RootPlanner
        _collect_and_merge(RootPlanner, "_root_plan", state, "RootPlanner_output")
        summary = _build_memory_tool_summary(state)
        state.setdefault("execution_log", []).append(summary)
        print(summary)
        state["status"] = "done"
        return END
    ttype, _ = _next_type("_root_plan", state)
    if ttype == "input_resolve":
        return "input_resolve"
    return "report_gen"


def r_input(state):
    """阶段一: 路由到执行节点或回到 root"""
    if not state.get("_input_plan"):
        return "input_resolve"

    if _all_done("_input_plan", state):
        from agents import InputResolveDecision
        _collect_and_merge(InputResolveDecision, "_input_plan", state, "InputResolveDecision_output")
        _done_type("_root_plan", "input_resolve", state)

        resolved = state.get("resolved_input", {})
        raw = state.get("raw_input", {})
        parsed = state.get("parsed_prompt_fields", {})
        if isinstance(parsed, dict):
            resolved["topic"] = parsed.get("topic", raw.get("prompt", "")[:50])
            resolved["requirements"] = parsed.get("requirements", raw.get("prompt", ""))
            resolved["student_name"] = parsed.get("student_name", "")
            resolved["student_id"] = parsed.get("student_id", "")
            resolved["theory_background"] = parsed.get("theory_background", "")
        if state.get("rag_input_results"):
            resolved["theory_background"] = (resolved.get("theory_background", "") +
                "\n" + "\n".join(state["rag_input_results"]))
        state["resolved_input"] = resolved
        state["current_phase"] = "report_generation"
        return "root_planner"

    ttype, _ = _next_type("_input_plan", state)
    m = {"prompt_parse": "prompt_parser", "file_parse": "file_parser",
         "memory_retrieve": "memory_retriever", "rag_retrieve": "rag_input",
         "code_read": "code_reader"}
    return m.get(ttype, "root_planner")


def r_input_child(state):
    return "input_resolve"


def r_report(state):
    if not state.get("_report_plan"):
        return "report_gen"

    if _all_done("_report_plan", state):
        from agents import ReportGenDecision
        _collect_and_merge(ReportGenDecision, "_report_plan", state, "ReportGenDecision_output")
        _done_type("_root_plan", "report_generation", state)
        return "root_planner"

    ttype, _ = _next_type("_report_plan", state)
    m = {"theory_decomposition": "theory", "method_decomposition": "method",
         "writing_decomposition": "writer"}
    return m.get(ttype, "root_planner")


def r_theory(state):
    if not state.get("_theory_plan"):
        return "theory"
    if _all_done("_theory_plan", state):
        from agents import TheoryDecision
        _collect_and_merge(TheoryDecision, "_theory_plan", state, "TheoryDecision_output")
        _done_type("_report_plan", "theory_decomposition", state)
        return "report_gen"
    return "rag"


def r_method(state):
    if not state.get("_method_plan"):
        return "method"
    if _all_done("_method_plan", state):
        from agents import MethodDecision
        _collect_and_merge(MethodDecision, "_method_plan", state, "MethodDecision_output")
        _done_type("_report_plan", "method_decomposition", state)
        return "report_gen"
    ttype, _ = _next_type("_method_plan", state)
    if "data" in ttype:
        return "data"
    if "image" in ttype:
        return "image"
    return "code"


def r_writer(state):
    if not state.get("_writer_plan"):
        return "writer"
    if _all_done("_writer_plan", state):
        from agents import WriterDecision
        _collect_and_merge(WriterDecision, "_writer_plan", state, "WriterDecision_output")
        _done_type("_report_plan", "writing_decomposition", state)
        return "report_gen"
    ttype, _ = _next_type("_writer_plan", state)
    return "writer_exec" if "section_writing" in ttype else "format"


def r_theory_child(state):
    return "theory"

def r_method_child(state):
    return "method"

def r_format_after(state):
    result = str(state.get("_fc_result", ""))
    rev = state.get("revision_count", 0)
    max_rev = state.get("max_revision_count", 3)

    if "PASS" in result or "通过" in result:
        return "report_gen"

    if rev < max_rev:
        state["revision_count"] = rev + 1
        if "问题" in result:
            lines = result.replace("FinalAnswer: FAIL", "").split("\n")
            comments = [l.strip("- ").strip() for l in lines if l.strip().startswith("-")]
            state["review_comments"] = comments or [result[:200]]
        for t in state.get("_writer_plan", []):
            if t.get("task_type") == "section_writing":
                t["status"] = "pending"
        return "writer"

    return "report_gen"


# ================================================================
# 构建图
# ================================================================

def build_report_graph():
    w = StateGraph(Dict)

    # 显式注册所有节点（函数名 ≠ 节点名的情况）
    node_map = {
        "root_planner": n_root,
        "input_resolve": n_input_resolve,
        "prompt_parser": n_prompt_parser,
        "file_parser": n_file_parser,
        "memory_retriever": n_memory_retriever,
        "rag_input": n_rag_input,
        "code_reader": n_code_reader,
        "report_gen": n_report_gen,
        "theory": n_theory,
        "method": n_method,
        "writer": n_writer,
        "rag": n_rag_exec,
        "search": n_search_exec,
        "data": n_data_exec,
        "code": n_code_exec,
        "image": n_image_exec,
        "writer_exec": n_writer_exec,
        "format": n_format_check,
    }
    for name, func in node_map.items():
        w.add_node(name, func)

    w.set_entry_point("root_planner")

    w.add_conditional_edges("root_planner", r_root,
        {"input_resolve": "input_resolve", "report_gen": "report_gen", END: END})

    w.add_conditional_edges("input_resolve", r_input,
        {"prompt_parser": "prompt_parser", "file_parser": "file_parser",
         "memory_retriever": "memory_retriever", "rag_input": "rag_input",
        "code_reader": "code_reader",
        "root_planner": "root_planner"})
    for n in ["prompt_parser", "file_parser", "memory_retriever", "rag_input", "code_reader"]:
        w.add_edge(n, "input_resolve")

    w.add_conditional_edges("report_gen", r_report,
        {"theory": "theory", "method": "method", "writer": "writer",
         "root_planner": "root_planner"})

    w.add_conditional_edges("theory", r_theory,
        {"rag": "rag", "search": "search", "report_gen": "report_gen"})
    w.add_edge("rag", "theory")
    w.add_edge("search", "theory")

    w.add_conditional_edges("method", r_method,
        {"data": "data", "code": "code", "image": "image", "report_gen": "report_gen"})
    w.add_edge("data", "method")
    w.add_edge("code", "method")
    w.add_edge("image", "method")

    w.add_conditional_edges("writer", r_writer,
        {"writer_exec": "writer_exec", "format": "format", "report_gen": "report_gen"})
    w.add_edge("writer_exec", "writer")
    w.add_conditional_edges("format", r_format_after,
        {"report_gen": "report_gen", "writer": "writer"})

    return w.compile()
