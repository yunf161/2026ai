"""
记忆检索工具

封装 LongTermMemory 为 LangChain 工具，
供 MemoryRetrieverAgent 调用。
"""
from langchain_core.tools import tool

# 全局单例，由 main.py 初始化
_memory_instance = None


def set_memory_instance(mem):
    """在 main.py 中注入 LongTermMemory 实例"""
    global _memory_instance
    _memory_instance = mem


@tool
def memory_search(query: str) -> str:
    """从长期记忆中语义搜索历史实验报告、用户偏好和模板。

    用于检索:
    - 相似历史实验报告（用 "决策树分类实验" 之类的查询）
    - 用户偏好（姓名、学号、输出格式等）
    - 报告模板

    Args:
        query: 检索查询词

    Returns:
        检索到的记忆内容
    """
    global _memory_instance
    if _memory_instance is None:
        return "[记忆] 长期记忆未初始化，请先运行系统。"

    try:
        # 1. 搜索历史报告
        reports = _memory_instance.search_reports(query, top_k=2)

        # 2. 获取用户偏好
        prefs = _memory_instance.get_preferences()

        # 3. 搜索模板
        templates = _memory_instance.search_templates(query, top_k=1)

        lines = []
        if prefs:
            lines.append("--- 用户偏好 ---")
            for k, v in prefs.items():
                if k != "updated_at":
                    lines.append(f"  {k}: {v}")

        if reports:
            lines.append("\n--- 相似历史报告 ---")
            for r in reports:
                topic = r.get("metadata", {}).get("topic", "")
                content = r.get("document", "")[:300]
                lines.append(f"  标题: {topic}")
                lines.append(f"  内容: {content}...")

        if templates:
            lines.append("\n--- 报告模板 ---")
            for t in templates:
                name = t.get("metadata", {}).get("name", "")
                content = t.get("document", "")[:200]
                lines.append(f"  模板名: {name}")
                lines.append(f"  内容: {content}...")

        if not lines:
            return "[记忆] 长期记忆中暂无相关记录。"

        return "\n".join(lines)
    except Exception as e:
        return f"[记忆] 检索失败: {e}"
