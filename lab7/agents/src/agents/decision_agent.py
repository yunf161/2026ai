"""
决策智能体

继承 BaseAgent，实现 ReAct 推理模式的决策节点。
职责：任务分解（decompose）+ 结果合并（merge）。
约束：不能直接调用工具，只能将子任务交给子节点。

decompose 流程:
  Think → 分析任务和当前状态 → 决定拆成几个子任务
  → 用 JSON 输出分解方案（子任务列表）
"""
from typing import List, Dict, Any, Optional

from .base_agent import BaseAgent
from state import SubTask


# ================================================================
# Agent 路由表：task_type → assigned_agent
# ================================================================

AGENT_ROUTING = {
    # 阶段一：输入解析
    "prompt_parse": "PromptParserAgent",
    "file_parse": "FileParserAgent",
    "memory_retrieve": "MemoryRetrieverAgent",
    "rag_retrieve": "RAGExecutorAgent",
    "code_read": "CodeReaderAgent",
    "input_resolve": "InputResolveDecision",
    # 阶段二：报告生成
    "theory_decomposition": "TheoryDecision",
    "method_decomposition": "MethodDecision",
    "writing_decomposition": "WriterDecision",
    "theory_retrieval": "RAGExecutor",
    "data_analysis": "DataExecutor",
    "code_execution": "CodeExecutor",
    "image_analysis": "ImageInterpreter",
    "section_writing": "SectionWriter",
    "format_check": "FormatChecker",
    "report_generation": "ReportGenDecision",
}

# 决策节点的子节点路由（每个决策节点允许的下层任务类型）
DECISION_CHILDREN = {
    "RootPlanner": ["input_resolve", "report_generation"],
    "InputResolveDecision": ["prompt_parse", "file_parse", "memory_retrieve", "rag_retrieve", "code_read"],
    "ReportGenDecision": ["theory_decomposition", "method_decomposition", "writing_decomposition"],
    "TheoryDecision": ["theory_retrieval"],
    "MethodDecision": ["data_analysis", "code_execution", "image_analysis"],
    "WriterDecision": ["section_writing", "format_check"],
}


class DecisionAgent(BaseAgent):
    """决策智能体"""

    # 本决策节点允许分配的子任务类型
    allowed_child_types: List[str] = []

    def __init__(self, name: str, system_prompt: str,
                 allowed_child_types: Optional[List[str]] = None):
        super().__init__(name, "decision", system_prompt)
        self.allowed_child_types = allowed_child_types or DECISION_CHILDREN.get(name, [])

    # ================================================================
    # decompose: 调用千问拆解任务 → 生成 SubTask 列表
    # ================================================================

    def decompose(self, task_description: str,
                  state: Dict[str, Any]) -> List[SubTask]:
        """ReAct: Think → Decompose → Assign

        用千问分析任务描述，拆分出子任务列表，分配 Agent。
        """
        self.log(f"开始分解任务: {task_description[:80]}...")

        # 获取已有的上下文（阶段一产出、历史记忆等）
        context_parts = []
        resolved = state.get("resolved_input", {})
        if resolved.get("topic"):
            context_parts.append(f"实验题目: {resolved['topic']}")
        if resolved.get("theory_background"):
            context_parts.append(f"理论背景: {resolved['theory_background']}")
        if state.get("parsed_prompt_fields"):
            context_parts.append(f"已提取字段: {state['parsed_prompt_fields']}")
        if state.get("memory_retrieval_results"):
            context_parts.append(f"记忆检索: {state['memory_retrieval_results'][:3]}")

        context_str = "\n".join(context_parts) if context_parts else "无"

        # 允许的任务类型
        allowed_types = ", ".join(self.allowed_child_types)

        prompt = f"""你需要将以下任务分解为子任务。

任务描述: {task_description}

已知上下文:
{context_str}

你可以分配的子任务类型（只能从以下选择）: {allowed_types}

请用 JSON 格式输出分解方案:
{{
    "reasoning": "你的分析思路（一句话）",
    "sub_tasks": [
        {{
            "task_id": "给子任务一个唯一ID",
            "description": "子任务描述",
            "task_type": "从允许列表中选择",
        }}
    ]
}}

注意:
- 只输出 JSON，不要输出其他内容
- 子任务数量 1-5 个
- 如果不需要某类子任务（如没有附件就不分 file_parse），就不列入
"""

        result = self.call_llm_json(prompt)

        if "error" in result:
            self.log(f"分解失败: {result['error']}")
            return []

        sub_tasks_data = result.get("sub_tasks", [])
        self.log(f"推理: {result.get('reasoning', '')}")

        # 补全字段 → SubTask
        sub_tasks = []
        for i, st in enumerate(sub_tasks_data):
            task_type = st.get("task_type", "")
            if task_type not in self.allowed_child_types:
                self.log(f"跳过非法类型: {task_type}")
                continue

            assigned = AGENT_ROUTING.get(task_type, "UnknownAgent")

            sub_tasks.append(SubTask(
                task_id=st.get("task_id", f"ST_{i + 1}"),
                description=st.get("description", task_description),
                task_type=task_type,
                assigned_agent=assigned,
                dependencies=st.get("dependencies", []),
                context=st.get("context", {}),
                status="pending",
            ))

        self.log(f"分解完成: {len(sub_tasks)} 个子任务")
        self.remember_now("decision",
                          f"Decomposed '{task_description[:60]}' into {len(sub_tasks)} sub-tasks")
        return sub_tasks

    # ================================================================
    # merge: 调用千问合并子节点结果
    # ================================================================

    def merge(self, task_description: str,
              sub_results: Dict[str, str]) -> str:
        """合并子节点的返回结果"""
        if not sub_results:
            return "[merge] 无子结果"

        results_text = "\n\n".join(
            f"### {name}\n{text}" for name, text in sub_results.items()
        )

        prompt = f"""请合并以下子节点返回的结果。

原始任务: {task_description}

子节点结果:
{results_text}

请整合为一段连贯的文本输出。保持信息完整，去除重复内容。"""

        merged = self.call_llm(prompt)
        self.remember_now("decision", f"Merged {len(sub_results)} sub-results")
        return merged

    # ================================================================
    # process: 决策节点主逻辑（LangGraph 调用入口）
    # ================================================================

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """决策节点处理流程"""
        plan = state.get("plan", [])
        current_idx = state.get("current_task_index", 0)

        if current_idx >= len(plan):
            return state

        current_task = plan[current_idx]
        task_desc = current_task.get("description", str(current_task))

        # 1. 分解任务
        sub_tasks = self.decompose(task_desc, state)

        # 2. 写入 state（LangGraph 条件边将根据 plan 路由到子节点）
        state["plan"] = sub_tasks
        state["current_task_index"] = 0
        state["next_agent"] = sub_tasks[0]["assigned_agent"] if sub_tasks else ""

        state.setdefault("execution_log", []).append(
            f"[{self.name}] Decomposed into {len(sub_tasks)} sub-tasks"
        )

        # 3. 如果子任务已被执行（由父节点收集结果后再次进入），则合并
        if state.get(f"_{self.name}_sub_results"):
            sub_results = state.pop(f"_{self.name}_sub_results")
            merged = self.merge(task_desc, sub_results)
            state[f"{self.name}_output"] = merged
            state.setdefault("execution_log", []).append(
                f"[{self.name}] Merged {len(sub_results)} results"
            )

        return state
