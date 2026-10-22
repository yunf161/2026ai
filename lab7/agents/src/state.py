"""
全局 State 类型定义

定义 LangGraph 中流转的 ReportState 及所有数据结构。
严格遵循 input_output_spec.md 中的定义。
"""
from typing import List, Dict, Any, Literal, Optional, TypedDict


# ============================================================
# 用户输入相关
# ============================================================

class RawUserInput(TypedDict):
    """用户原始输入 —— 灵活开放，最少只需一个提示词"""
    # ===== 必填 =====
    prompt: str                         # 用户提示词，自然语言描述实验需求
                                        # 如："用DT_data.csv做决策树分类实验，我叫张三"

    # ===== 选填（用户可按需提供任意补充材料）=====
    data_files: Optional[List[str]]     # 数据文件（CSV、JSON、图片等）
    reference_materials: Optional[List[str]]  # 参考资料（PDF实验要求、课件、论文等）
    code_file: Optional[str]            # 参考代码文件
    template: Optional[str]             # 自定义报告模板（Markdown）
    output_format: Optional[str]        # 输出格式偏好 "pdf" | "markdown"


class ExperimentInput(TypedDict):
    """系统补全后的完整输入（经记忆检索 + 课件RAG + 文件解析后填充）"""
    # ===== 从提示词中直接提取 =====
    topic: str                          # 实验题目，如 "决策树分类实验"
    requirements: str                   # 实验要求完整描述

    # ===== 从长期记忆 / RAG / 对话中补全 =====
    experiment_purpose: str             # 实验目的（从历史报告模板补全）
    theory_background: str              # 实验原理提示/知识点大纲（从RAG检索课件补全）
    experiment_method: str              # 实验方法/步骤描述（从历史类似报告补全）
    data_files: List[str]               # 数据文件路径列表
    code_file: str                      # 参考代码文件路径
    reference_materials: List[str]      # 参考资料路径（PDF/Markdown）
    template: str                       # 报告模板（优先历史偏好，否则默认模板）
    output_format: str                  # 输出格式，默认 "pdf"，可选 "markdown"

    # ===== 元信息 =====
    student_name: str                   # 姓名（从长期记忆中的用户偏好获取）
    student_id: str                     # 学号（从长期记忆中的用户偏好获取）
    course_name: str                    # 课程名称，默认 "人工智能"

    # ===== 记忆溯源 =====
    memory_sources: List[str]           # 各字段从哪些记忆来源补全，便于追溯
                                        # 如：["history:2024决策树实验", "rag:决策树课件第3章"]


# ============================================================
# 子任务相关
# ============================================================

# 任务类型枚举
TaskType = Literal[
    # === 阶段一：输入解析 ===
    "prompt_parse",                     # 提示词解析
    "file_parse",                       # 通用文件解析（PDF/图片/Word/MD等）
    "memory_retrieve",                  # 长期记忆检索
    "rag_retrieve",                     # RAG课件检索
    "input_resolve",                    # 输入解析总任务（决策节点用）
    # === 阶段二：报告生成 ===
    "theory_retrieval",                 # 理论检索（RAG/搜索）
    "data_analysis",                    # 数据分析
    "code_execution",                   # 代码执行
    "section_writing",                  # 章节撰写
    "format_check",                     # 格式检查
    "theory_decomposition",             # 理论部分拆解（决策节点用）
    "method_decomposition",             # 方法部分拆解（决策节点用）
    "writing_decomposition",            # 撰写部分拆解（决策节点用）
    "report_generation",                # 报告生成总任务（决策节点用）
]


class SubTask(TypedDict):
    """子任务定义"""
    task_id: str                        # 任务唯一ID，如 "P1.1"
    description: str                    # 任务描述
    task_type: TaskType                 # 任务类型
    assigned_agent: str                 # 分配的Agent名称
    dependencies: List[str]             # 依赖的task_id列表
    context: Dict[str, Any]             # 上下文信息
    status: Literal["pending", "running", "done", "failed"]


# ============================================================
# 报告相关
# ============================================================

class ReportMeta(TypedDict):
    """报告元信息"""
    title: str                          # 报告标题
    student_name: str                   # 姓名
    student_id: str                     # 学号
    course_name: str                    # 课程
    date: str                           # 日期
    version: int                        # 版本号（Reflection修改后递增）


class ReportSection(TypedDict):
    """报告章节"""
    section_id: str                     # 章节ID
    title: str                          # 章节标题
    content: str                        # 章节内容（Markdown格式）
    source_agent: str                   # 由哪个Agent生成
    status: Literal["draft", "reviewed", "final"]


class ExperimentReport(TypedDict):
    """最终实验报告"""
    meta: ReportMeta
    sections: List[ReportSection]
    figures: List[str]                  # 图片路径列表
    attachments: List[str]              # 附件路径列表（代码等）


# ============================================================
# 记忆相关
# ============================================================

class MemoryRecord(TypedDict):
    """单条记忆记录"""
    role: str                           # "user" | "agent" | "system"
    agent_name: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]


class AgentMemory(TypedDict):
    """Agent 记忆"""
    short_term: List[MemoryRecord]      # 当前会话的短期记忆
    long_term_keys: List[str]           # ChromaDB 中的文档 ID 列表


# ============================================================
# Agent 间通信
# ============================================================

class ToolCallRecord(TypedDict):
    """工具调用记录"""
    tool: str                           # 工具名称
    query: str                          # 调用参数/查询
    result: str                         # 工具返回结果


class DecisionResult(TypedDict):
    """决策节点返回给父节点的结果"""
    agent_name: str                     # 当前决策Agent名称
    original_task: str                  # 接收到的原始任务
    decomposition: List[SubTask]        # 任务分解结果
    sub_results: Dict[str, str]         # 子节点返回结果 {子节点名: 结果内容}
    merged_result: str                  # 合并后的结果
    status: Literal["success", "partial", "failed"]
    execution_log: List[str]            # 执行日志


class ExecutionResult(TypedDict):
    """执行节点返回给父节点的结果"""
    agent_name: str                     # 当前执行Agent名称
    task: SubTask                       # 被分配的子任务
    tool_calls: List[ToolCallRecord]    # 工具调用记录
    result: str                         # 执行结果
    status: Literal["success", "failed"]
    execution_log: List[str]            # 执行日志


# ============================================================
# LangGraph 全局 State
# ============================================================

class ReportState(TypedDict):
    """LangGraph 全局 State —— 在所有Agent节点间流转"""

    # === 输入 ===
    raw_input: RawUserInput             # 用户原始输入
    resolved_input: ExperimentInput     # 系统补全后的完整输入（阶段一完成后填充）

    # === 规划 ===
    plan: List[SubTask]                 # 当前阶段的子任务列表
    current_task_index: int             # 当前执行到第几个子任务（用于遍历 plan）
    current_phase: Literal[
        "input_resolve",                # 阶段一：输入解析
        "report_generation",            # 阶段二：报告生成
        "done"
    ]

    # === 阶段一：输入解析产出 ===
    parsed_prompt_fields: Dict[str, str]          # PromptParserAgent 提取的字段
    parsed_file_contents: List[str]               # FileParserAgent 提取的各文件文本
    memory_retrieval_results: List[str]           # 长期记忆检索结果
    rag_input_results: List[str]                  # RAG 课件检索结果

    # === 阶段二：报告生成产出 ===
    theory_content: str                           # 理论章节内容（TheoryDecision 合并结果）
    method_content: str                           # 方法/结果章节内容（MethodDecision 合并结果）
    draft_sections: Dict[str, str]                # 草稿章节 {section_id: content}

    # === Reflection 反思 ===
    review_comments: List[str]                    # FormatChecker 的修改意见
    revision_count: int                           # 已修改次数
    max_revision_count: int                       # 最大反思轮数（默认3）

    # === 记忆 ===
    memory: AgentMemory                           # 短期 + 长期记忆

    # === 控制 ===
    next_agent: str                               # 下一个要执行的Agent名称（条件边路由用）
    final_report: Optional[ExperimentReport]      # 最终报告（阶段二完成后填充）
    status: Literal[
        "planning",                               # 规划中
        "input_resolving",                        # 阶段一执行中
        "executing",                              # 阶段二执行中
        "reviewing",                              # 反思检查中
        "done",                                   # 完成
        "error"                                   # 出错
    ]

    # === 日志 ===
    execution_log: List[str]                      # 全局执行日志


# ============================================================
# 默认报告模板
# ============================================================

DEFAULT_REPORT_TEMPLATE = """\
# {course_name} 实验报告 第X周

姓名: {student_name}  学号: {student_id}

### 一. 实验题目

{topic}

### 二. 实验内容

###### 1. 算法原理

{{algorithm_principles}}

###### 2. 关键代码展示（可选）

{{code_snippets}}

###### 3. 创新点 & 优化（如果有）

{{innovations}}

### 三. 实验结果及分析

###### 1. 实验结果展示

{{experiment_results}}

###### 2. 评测指标展示及分析

{{metrics_analysis}}

### 四. 参考资料（可选）

{{references}}
"""

