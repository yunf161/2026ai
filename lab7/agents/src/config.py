"""
系统配置模块

管理 API Key、模型参数、路径等全局配置。
API Key 通过环境变量或 .env 文件注入，禁止硬编码。
"""
import os
from dataclasses import dataclass, field
from typing import Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SRC_DIR, "data")

# 尝试加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Config:
    """系统全局配置"""

    # === 大模型 API ===
    model_name: str = os.getenv("LLM_MODEL", "deepseek-v4-pro")
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    temperature: float = 0.7
    max_tokens: int = 4096

    # === 视觉模型 API（Qwen2.5-VL / DashScope OpenAI兼容模式） ===
    vision_model_name: str = os.getenv("VISION_MODEL_NAME", "qwen2.5-vl-72b-instruct")
    vision_base_url: str = os.getenv("VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    vision_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    vision_max_tokens: int = int(os.getenv("VISION_MAX_TOKENS", "1200"))

    # === 重试策略 ===
    max_retries: int = 3
    retry_delay: float = 1.0

    # === 记忆系统 ===
    short_term_window: int = 10          # 短期记忆滑动窗口大小
    chroma_persist_dir: str = os.path.join(DATA_DIR, "chroma_db")

    # === RAG 检索 ===
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "dashscope")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    embedding_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    # 默认使用 DashScope 云端 embedding；如需本地可设置 EMBEDDING_PROVIDER=local 和 EMBEDDING_MODEL=本地/Hub模型
    rag_top_k: int = 3

    # === Reflection ===
    max_reflection_rounds: int = 3       # 最大反思重做轮数

    # === 路径 ===
    # 发布版使用可配置路径，避免绑定原开发者目录。
    courseware_dir: str = os.getenv("COURSEWARE_DIR") or os.path.join(DATA_DIR, "courseware")
    output_dir: str = os.path.join(SRC_DIR, "output")

    # === PDF 生成 ===
    pdf_font_path: Optional[str] = os.getenv("PDF_FONT_PATH") or None


# 全局配置单例
config = Config()
