"""
搜索引擎工具

从互联网获取最新资料，作为 RAG 检索的补充。
使用 DuckDuckGo 搜索引擎（免费，无需 API Key）。
"""
import time
from typing import Any, Dict, List

from langchain_core.tools import tool


def _load_ddgs_class():
    """优先使用新包 ddgs，兼容旧包 duckduckgo_search。"""
    try:
        from ddgs import DDGS
        return DDGS, "ddgs"
    except Exception:
        pass

    try:
        from duckduckgo_search import DDGS
        return DDGS, "duckduckgo_search"
    except Exception:
        return None, ""


def _run_ddg_query(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """执行 DuckDuckGo 查询并返回统一结构。"""
    DDGS, provider = _load_ddgs_class()
    if DDGS is None:
        return []

    # 多种调用参数做兼容，降低不同版本客户端行为差异导致的失败概率
    kwargs_candidates = [
        {"max_results": max_results},
        {"max_results": max_results, "backend": "lite"},
        {"max_results": max_results, "region": "wt-wt"},
    ]

    for kwargs in kwargs_candidates:
        try:
            with DDGS() as ddgs:
                data = list(ddgs.text(query, **kwargs))
        except Exception:
            continue

        if not data:
            continue

        normalized = []
        for item in data:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": str(item.get("title") or item.get("heading") or "无标题"),
                    "href": str(item.get("href") or item.get("url") or ""),
                    "body": str(item.get("body") or item.get("snippet") or item.get("description") or ""),
                    "provider": provider,
                }
            )
        if normalized:
            return normalized

    return []


@tool
def web_search(query: str) -> str:
    """从互联网搜索最新资料。

    用于查询课件中没有的最新研究、API 文档、新闻等。
    基于 DuckDuckGo 搜索引擎（免费且无需 API Key）。

    Args:
        query: 搜索查询，如 "sklearn DecisionTreeClassifier 用法"

    Returns:
        搜索结果摘要（标题 + 链接 + 摘要），最多5条
    """
    q = str(query or "").strip()
    if not q:
        return "[搜索] 查询词为空"

    DDGS, provider = _load_ddgs_class()
    if DDGS is None:
        return "[搜索降级] 未安装 ddgs/duckduckgo_search，建议安装: pip install ddgs"

    # 重试机制：短退避，减少临时网络抖动影响
    attempts = 3
    for i in range(attempts):
        results = _run_ddg_query(q, max_results=5)
        if results:
            lines = [f"[搜索] 查询: {q} (provider={provider})\n"]
            for idx, r in enumerate(results, 1):
                title = r.get("title", "无标题")
                href = r.get("href", "")
                body = r.get("body", "")
                lines.append(f"{idx}. {title}")
                if href:
                    lines.append(f"   链接: {href}")
                lines.append(f"   摘要: {body}\n")
            return "\n".join(lines)

        # 轻微退避，避免瞬时失败直接报错
        if i < attempts - 1:
            time.sleep(0.8 * (i + 1))

    # 兜底：返回可消费的降级文本，避免上游链路把它当作 hard error
    return (
        f"[搜索降级] 外网检索暂时不可用，查询: {q}\n"
        "可能原因: 网络波动、上游搜索服务限流或地区可用性问题。\n"
        "建议: 先使用 RAG 课件检索结果继续生成；稍后重试 web_search。"
    )
