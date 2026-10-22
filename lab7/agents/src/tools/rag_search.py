"""
RAG 检索工具

从课件/教材知识库中语义检索相关内容。
离线构建知识库: build_knowledge_base()
在线检索: rag_search()
"""
import os
import time
from pathlib import Path
from typing import List
from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings

# 状态
_store = None
_last_init_failed_at = 0.0
_retry_cooldown_sec = 30
_embedding_ready = False


def _get_persist_dir():
    """按 embedding provider/model 隔离 Chroma 目录，避免维度冲突。"""
    from config import config
    provider = getattr(config, "embedding_provider", "dashscope").lower()
    model = str(getattr(config, "embedding_model", "embedding")).replace("/", "_").replace(":", "_")
    if provider == "dashscope":
        return os.path.join(config.chroma_persist_dir, f"{provider}_{model}")
    return os.path.join(config.chroma_persist_dir, f"local_{model}")


class DashScopeEmbeddings(Embeddings):
    """DashScope OpenAI-compatible embedding wrapper."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置，无法使用 DashScope embedding")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        clean = [str(t or "") for t in texts]
        resp = self.client.embeddings.create(model=self.model, input=clean)
        return [item.embedding for item in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 分批避免单次请求过大
        out = []
        # DashScope text-embedding-v3 单批 input 不能超过 10
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            out.extend(self._embed(texts[i:i + batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def _get_embeddings(local_only: bool):
    """获取 embedding 模型；默认优先 DashScope 云端，local provider 时才用本地缓存。"""
    from config import config

    if getattr(config, "embedding_provider", "dashscope").lower() == "dashscope":
        return DashScopeEmbeddings(
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
        )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    kwargs = {
        'device': 'cpu',
        'local_files_only': local_only,
    }
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        cache_folder=cache_dir,
        model_kwargs=kwargs,
        encode_kwargs={'normalize_embeddings': True}
    )


def _warmup_embedding_model(force_download: bool = False):
    """预热 embedding：DashScope 做一次轻量请求；本地模式则下载/缓存模型。"""
    global _embedding_ready
    from config import config

    if _embedding_ready and not force_download:
        return True

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if getattr(config, "embedding_provider", "dashscope").lower() == "dashscope":
        try:
            emb = _get_embeddings(local_only=False)
            emb.embed_query("测试")
            _embedding_ready = True
            print(f"[RAG] DashScope embedding 可用: {config.embedding_model}")
            return True
        except Exception as e:
            print(f"[RAG] DashScope embedding 预热失败: {e}")
            return False

    # 1) 先尝试本地缓存；若没有，再允许下载一次
    for local_only in (True, False):
        try:
            _get_embeddings(local_only=local_only)
            _embedding_ready = True
            if local_only:
                print(f"[RAG] embedding 模型已命中本地缓存: {config.embedding_model}")
            else:
                print(f"[RAG] embedding 模型已下载并缓存: {config.embedding_model}")
            return True
        except Exception as e:
            mode = "本地缓存" if local_only else "联网下载"
            print(f"[RAG] {mode}预热失败: {e}")

    return False


def _get_store():
    """获取向量存储。初始化失败后进入短暂冷却，再自动重试。"""
    global _store, _last_init_failed_at
    if _store is not None:
        return _store
    if _last_init_failed_at and (time.time() - _last_init_failed_at < _retry_cooldown_sec):
        return None

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from langchain_chroma import Chroma
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        _last_init_failed_at = time.time()
        print(f"[RAG] 依赖缺失: {e}")
        return None

    from config import config
    persist_dir = _get_persist_dir()

    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        _last_init_failed_at = time.time()
        return None

    # 先预热 embedding：DashScope云端或本地缓存
    if not _embedding_ready:
        _warmup_embedding_model(force_download=False)

    errors = []
    for local_only in (True, False):
        try:
            embeddings = _get_embeddings(local_only=local_only)
            _store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            _last_init_failed_at = 0.0
            return _store
        except Exception as e:
            mode = "本地缓存" if local_only else "联网下载"
            errors.append(f"{mode}: {e}")

    _last_init_failed_at = time.time()
    print(f"[RAG] 初始化失败: {' | '.join(errors)}")
    return None


def build_knowledge_base(courseware_dir=None):
    """离线构建知识库（运行一次即可）"""
    global _store, _last_init_failed_at
    from config import config

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    source_dir = courseware_dir or config.courseware_dir
    persist_dir = _get_persist_dir()

    if not os.path.exists(source_dir):
        return f"[Error] 课件目录不存在: {source_dir}"

    try:
        from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        return f"[Error] 缺少依赖: {e}"

    # 预热模型：DashScope云端可用或本地模型已缓存
    if not _warmup_embedding_model(force_download=False):
        return "[Error] embedding 模型预热失败，请检查网络或 HuggingFace 缓存"

    docs = []
    failed_files = []
    source_path = Path(source_dir)
    for fp in source_path.rglob("*"):
        if not fp.is_file():
            continue
        suffix = fp.suffix.lower()
        try:
            if suffix == ".md" or suffix == ".markdown":
                loader = TextLoader(str(fp), encoding="utf-8")
                try:
                    docs.extend(loader.load())
                except Exception:
                    loader = TextLoader(str(fp), encoding="gbk")
                    docs.extend(loader.load())
            elif suffix == ".pdf":
                loader = PyMuPDFLoader(str(fp))
                docs.extend(loader.load())
        except Exception as e:
            failed_files.append(f"{fp}: {e}")
            print(f"[RAG] 文件加载警告: {fp} -> {e}")

    if not docs:
        detail = "；".join(failed_files[:5]) if failed_files else "未找到可解析文件"
        return f"[Error] 未找到任何可解析的课件文件: {source_dir}。{detail}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ".", ",", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # 构建阶段：DashScope默认云端；本地模式则优先缓存
    if not _embedding_ready:
        _warmup_embedding_model(force_download=False)

    embeddings = _get_embeddings(local_only=True)
    _store = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=persist_dir
    )
    _last_init_failed_at = 0.0

    os.makedirs(persist_dir, exist_ok=True)
    msg = f"[RAG] 构建完成: {len(docs)}篇 → {len(chunks)}块 → {persist_dir}"
    if failed_files:
        msg += f"；跳过 {len(failed_files)} 个文件"
    return msg


@tool
def rag_search(query: str) -> str:
    """从课程课件和教材中检索相关知识。

    用于查询实验原理、算法定义、公式推导等学术内容。
    基于语义相似度检索，返回最相关的课件段落。

    Args:
        query: 检索查询，如 "决策树 信息增益 基尼系数"

    Returns:
        Top-K 相关课件段落，用分隔符隔开。若知识库未构建则返回提示。
    """
    from config import config

    store = _get_store()
    if store is None:
        return (
            "[RAG] 知识库未构建或不可用。\n"
            "请先运行: from tools import build_knowledge_base; "
            f"build_knowledge_base('{config.courseware_dir}')\n"
            "或跳过RAG，搜索引擎将作为替代。"
        )

    try:
        docs = store.similarity_search(query, k=config.rag_top_k)
        if not docs:
            return f"[RAG] 未找到与 '{query}' 相关的课件内容"

        results = []
        for i, doc in enumerate(docs):
            src = os.path.basename(doc.metadata.get('source', '未知'))
            results.append(f"[资料{i + 1}] 来源: {src}\n{doc.page_content}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"[RAG] 检索失败: {e}"
