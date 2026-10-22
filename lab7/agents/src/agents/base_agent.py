"""
Agent 基类

所有 Agent（决策节点 & 执行节点）的抽象基类。
定义通用接口：LLM调用、记忆读写、上下文拼接。
"""
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import config
from state import MemoryRecord


class BaseAgent(ABC):
    """Agent 抽象基类"""

    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.short_term_memory: List[MemoryRecord] = []
        self.model: Optional[ChatOpenAI] = None

    # ================================================================
    # LLM 调用
    # ================================================================

    def _init_model(self) -> None:
        """延迟初始化模型"""
        if self.model is None:
            self.model = ChatOpenAI(
                model=config.model_name,
                base_url=config.base_url,
                api_key=config.api_key,
                temperature=config.temperature,
            )

    def call_llm(self, user_message: str,
                 system_override: Optional[str] = None) -> str:
        """调用千问 API，返回文本"""
        self._init_model()
        sys_prompt = system_override or self.system_prompt
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_message),
        ]
        for attempt in range(config.max_retries):
            try:
                response = self.model.invoke(messages)
                return response.content
            except Exception as e:
                self.log(f"API 调用失败 (尝试 {attempt + 1}/{config.max_retries}): {e}")
                if attempt == config.max_retries - 1:
                    return f"[Error] API 调用失败: {e}"
        return "[Error] 未知错误"

    def call_llm_json(self, user_message: str,
                      system_override: Optional[str] = None) -> Dict[str, Any]:
        """调用千问并强制返回 JSON"""
        raw = self.call_llm(user_message, system_override)
        json_str = raw
        if "```json" in raw:
            start = raw.find("```json") + 7
            end = raw.find("```", start)
            if end > start:
                json_str = raw[start:end]
        elif "```" in raw:
            start = raw.find("```") + 3
            end = raw.find("```", start)
            if end > start:
                json_str = raw[start:end]
        else:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                json_str = raw[start:end + 1]
        try:
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            self.log(f"JSON 解析失败，原始: {raw[:200]}...")
            return {"error": "JSON解析失败", "raw": raw[:500]}

    # ================================================================
    # 短期记忆接口（per-agent）
    # ================================================================

    def remember(self, record: MemoryRecord) -> None:
        self.short_term_memory.append(record)
        if len(self.short_term_memory) > config.short_term_window:
            self.short_term_memory = self.short_term_memory[-config.short_term_window:]

    def remember_now(self, role: str, content: str,
                     metadata: Optional[Dict] = None) -> None:
        record = MemoryRecord(
            role=role, agent_name=self.name, content=content,
            timestamp=datetime.now().isoformat(), metadata=metadata or {}
        )
        self.remember(record)

    def get_memory_context(self, n: Optional[int] = None) -> str:
        if n is None:
            n = config.short_term_window
        recent = self.short_term_memory[-n:]
        return "\n".join(
            f"[{m['agent_name']}]({m['role']}): {m['content']}" for m in recent
        )

    # ================================================================
    # 上下文拼接
    # ================================================================

    def build_context(self, extra: Optional[Dict[str, str]] = None) -> str:
        parts = [self.system_prompt]
        memory = self.get_memory_context()
        if memory:
            parts.append(f"\n--- 历史记录 ---\n{memory}")
        if extra:
            for key, value in extra.items():
                parts.append(f"\n--- {key} ---\n{value}")
        return "\n\n".join(parts)

    # ================================================================
    # 抽象方法
    # ================================================================

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    # ================================================================
    # 辅助方法
    # ================================================================

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}][{self.name}] {message}")

    def make_record(self, role: str, content: str,
                    metadata: Optional[Dict] = None) -> MemoryRecord:
        return MemoryRecord(
            role=role, agent_name=self.name, content=content,
            timestamp=datetime.now().isoformat(), metadata=metadata or {}
        )
