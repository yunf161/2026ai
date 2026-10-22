"""
短期记忆

存储当前会话中的对话历史、中间步骤、任务分解结果。
采用滑动窗口机制，保留最近 N 轮交互。
"""
from typing import List, Optional, Dict, Any
from datetime import datetime

from state import MemoryRecord
from config import config


class ShortTermMemory:
    """短期记忆管理器

    用途：在当前会话中，各 Agent 共享同一个 ShortTermMemory 实例，
    通过 ReportState.memory.short_term 引用。
    每次添加记忆时自动裁剪到 max_size。
    """

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size or config.short_term_window
        self.records: List[MemoryRecord] = []

    # === 增删改查 ===

    def add(self, role: str, agent_name: str, content: str,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加一条记忆（自动裁剪）

        Args:
            role: "user" | "agent" | "system" | "decision" | "execution"
            agent_name: 产生这条记忆的 Agent 名称
            content: 记忆内容
            metadata: 附加信息（如 task_id, sub_task 等）
        """
        record = MemoryRecord(
            role=role,
            agent_name=agent_name,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.records.append(record)
        self._trim()

    def add_record(self, record: MemoryRecord) -> None:
        """直接添加 MemoryRecord 对象"""
        self.records.append(record)
        self._trim()

    def get_recent(self, n: Optional[int] = None) -> List[MemoryRecord]:
        """获取最近 n 条记忆"""
        if n is None:
            n = self.max_size
        return self.records[-n:]

    def search(self, keyword: str) -> List[MemoryRecord]:
        """按关键词搜索记忆（简单子串匹配）"""
        keyword_lower = keyword.lower()
        return [
            r for r in self.records
            if keyword_lower in r["content"].lower()
            or keyword_lower in r["agent_name"].lower()
        ]

    def filter_by_agent(self, agent_name: str) -> List[MemoryRecord]:
        """按 Agent 名称过滤"""
        return [r for r in self.records if r["agent_name"] == agent_name]

    def filter_by_role(self, role: str) -> List[MemoryRecord]:
        """按角色过滤（'user', 'agent', 'system', 'decision', 'execution'）"""
        return [r for r in self.records if r["role"] == role]

    def get_last_by_agent(self, agent_name: str) -> Optional[MemoryRecord]:
        """获取指定 Agent 的最后一条记忆"""
        for r in reversed(self.records):
            if r["agent_name"] == agent_name:
                return r
        return None

    # === 上下文拼接 ===

    def get_context(self, n: Optional[int] = None,
                    include_meta: bool = False) -> str:
        """获取记忆上下文字符串（喂给千问的格式）

        Args:
            n: 取最近 n 条，默认全部
            include_meta: 是否包含时间戳和元信息
        """
        records = self.records[-n:] if n else self.records
        lines = []
        for r in records:
            if include_meta:
                lines.append(
                    f"[{r['timestamp'][:19]}][{r['agent_name']}]"
                    f"({r['role']}): {r['content']}"
                )
            else:
                lines.append(f"[{r['agent_name']}]({r['role']}): {r['content']}")
        return "\n".join(lines)

    def get_context_by_agent(self, agent_names: List[str],
                             n: Optional[int] = None) -> str:
        """只获取指定 Agent 的记忆上下文"""
        records = self.records[-n:] if n else self.records
        lines = []
        for r in records:
            if r["agent_name"] in agent_names:
                lines.append(f"[{r['agent_name']}]: {r['content']}")
        return "\n".join(lines)

    # === 快照与导出 ===

    def snapshot(self) -> List[Dict[str, Any]]:
        """导出所有记忆为字典列表（可序列化）"""
        return [dict(r) for r in self.records]

    def summarize(self) -> str:
        """生成记忆摘要"""
        if not self.records:
            return "[短期记忆] 无记录"
        agents = set(r["agent_name"] for r in self.records)
        roles = set(r["role"] for r in self.records)
        return (
            f"[短期记忆] 共 {len(self.records)} 条 | "
            f"Agent: {agents} | 角色: {roles}"
        )

    # === 管理 ===

    def clear(self) -> None:
        """清空短期记忆"""
        self.records.clear()

    def _trim(self) -> None:
        """裁剪到 max_size（滑动窗口）"""
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return self.summarize()
