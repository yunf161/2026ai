"""
长期记忆

基于 ChromaDB 向量数据库的持久化记忆存储。
存储三类内容:
  1. report_history  — 历史实验报告摘要（用于检索相似报告）
  2. user_preferences — 用户偏好（姓名、学号、输出格式等）
  3. report_templates — 常用报告模板

和 RAG 知识库的区别:
  - RAG 知识库: 存储课件内容（公开知识，一次性构建）
  - 长期记忆: 存储历史交互数据（私有经验，持续累积）
"""
import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime


class LongTermMemory:
    """长期记忆管理器（ChromaDB）

    使用独立的 ChromaDB 存储目录，与 RAG 课件知识库分离。
    通过不同的 collection 区分报告历史/用户偏好/模板。
    """

    def __init__(self, persist_dir: Optional[str] = None):
        """
        Args:
            persist_dir: ChromaDB 持久化目录，默认从 config 读取
        """
        from config import config
        self.persist_dir = persist_dir or os.path.join(
            os.path.dirname(config.chroma_persist_dir), "memory_db"
        )
        self._client = None
        self._collections = {}

    # === 延迟初始化 ===

    def _ensure_initialized(self):
        """延迟初始化 ChromaDB 连接和 collection"""
        if self._client is not None:
            return

        import chromadb
        from config import config

        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)

        # 三个 collection: 报告历史 / 用户偏好 / 模板
        collection_names = ["report_history", "user_preferences", "report_templates"]
        for name in collection_names:
            try:
                self._collections[name] = self._client.get_collection(name)
            except Exception:
                self._collections[name] = self._client.create_collection(
                    name=name,
                    metadata={"description": f"长期记忆 - {name}"}
                )

    def _get_collection(self, name: str):
        """获取指定 collection"""
        self._ensure_initialized()
        return self._collections.get(name)

    # ============================================================
    # 1. 报告历史（report_history）
    # ============================================================

    def store_report(self, topic: str, report_summary: str,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储一份实验报告摘要

        Args:
            topic: 实验题目
            report_summary: 报告摘要（实验目的 + 方法 + 关键结果）
            metadata: 附加信息（student_name, date, experiment_type 等）

        Returns:
            文档 ID
        """
        collection = self._get_collection("report_history")
        doc_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{topic[:20]}"

        meta = metadata or {}
        meta["topic"] = topic
        meta["stored_at"] = datetime.now().isoformat()

        collection.add(
            ids=[doc_id],
            documents=[report_summary],
            metadatas=[meta]
        )
        return doc_id

    def search_reports(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """语义搜索历史报告

        Args:
            query: 查询文本（如 "决策树分类"）
            top_k: 返回结果数

        Returns:
            [{id, document, metadata, distance}, ...]
        """
        collection = self._get_collection("report_history")
        if collection is None:
            return []

        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            output = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    output.append({
                        "id": doc_id,
                        "document": results["documents"][0][i] if results["documents"][0] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                        "distance": results["distances"][0][i] if results.get("distances") else None,
                    })
            return output
        except Exception:
            return []

    def get_recent_reports(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近 n 份报告（按存储时间）"""
        collection = self._get_collection("report_history")
        if collection is None:
            return []

        try:
            # ChromaDB 获取全部，然后按时间排序
            all_docs = collection.get()
            if not all_docs["ids"]:
                return []

            items = []
            for i, doc_id in enumerate(all_docs["ids"]):
                items.append({
                    "id": doc_id,
                    "document": all_docs["documents"][i] if all_docs["documents"] else "",
                    "metadata": all_docs["metadatas"][i] if all_docs["metadatas"] else {},
                })

            # 按时间降序排序
            items.sort(
                key=lambda x: x["metadata"].get("stored_at", ""),
                reverse=True
            )
            return items[:n]
        except Exception:
            return []

    # ============================================================
    # 2. 用户偏好（user_preferences）
    # ============================================================

    def save_preferences(self, prefs: Dict[str, Any]) -> str:
        """保存或更新用户偏好

        Args:
            prefs: 偏好字典 {"student_name": "张三", "output_format": "pdf", ...}

        Returns:
            文档 ID
        """
        collection = self._get_collection("user_preferences")
        doc_id = "user_prefs"

        # 删除旧偏好，存入新的（保持只有一条）
        try:
            collection.delete(ids=[doc_id])
        except Exception:
            pass

        prefs["updated_at"] = datetime.now().isoformat()
        collection.add(
            ids=[doc_id],
            documents=[json.dumps(prefs, ensure_ascii=False)],
            metadatas=[prefs]
        )
        return doc_id

    def get_preferences(self) -> Dict[str, Any]:
        """获取用户偏好"""
        collection = self._get_collection("user_preferences")
        if collection is None:
            return {}

        try:
            result = collection.get(ids=["user_prefs"])
            if result["ids"] and result["metadatas"] and result["metadatas"][0]:
                return dict(result["metadatas"][0])
        except Exception:
            pass
        return {}

    def update_preference(self, key: str, value: Any) -> None:
        """更新单个偏好项"""
        prefs = self.get_preferences()
        prefs[key] = value
        self.save_preferences(prefs)

    # ============================================================
    # 3. 报告模板（report_templates）
    # ============================================================

    def store_template(self, name: str, template_content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储一个报告模板

        Args:
            name: 模板名称
            template_content: 模板内容（Markdown）
            metadata: 附加信息（适用于什么类型的实验等）

        Returns:
            文档 ID
        """
        collection = self._get_collection("report_templates")
        doc_id = f"template_{name}"

        meta = metadata or {}
        meta["name"] = name
        meta["stored_at"] = datetime.now().isoformat()

        try:
            collection.delete(ids=[doc_id])
        except Exception:
            pass

        collection.add(
            ids=[doc_id],
            documents=[template_content],
            metadatas=[meta]
        )
        return doc_id

    def search_templates(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """搜索最匹配的模板"""
        collection = self._get_collection("report_templates")
        if collection is None:
            return []

        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            output = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    output.append({
                        "id": doc_id,
                        "document": results["documents"][0][i] if results["documents"][0] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                    })
            return output
        except Exception:
            return []

    # ============================================================
    # 4. 通用方法
    # ============================================================

    def search_by_metadata(self, collection_name: str,
                           filters: Dict[str, Any],
                           top_k: int = 3) -> List[Dict[str, Any]]:
        """按元信息过滤检索（用于精确查找）

        Args:
            collection_name: "report_history" | "user_preferences" | "report_templates"
            filters: 过滤条件（如 {"topic": "决策树"}）
            top_k: 返回结果数

        Returns:
            匹配的结果列表
        """
        collection = self._get_collection(collection_name)
        if collection is None:
            return []

        try:
            results = collection.get(where=filters)
            output = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    output.append({
                        "id": doc_id,
                        "document": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    })
            return output[:top_k]
        except Exception:
            return []

    def delete(self, collection_name: str, doc_id: str) -> bool:
        """删除一条长期记忆"""
        collection = self._get_collection(collection_name)
        if collection is None:
            return False
        try:
            collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False

    def summarize(self) -> str:
        """长期记忆状态摘要"""
        self._ensure_initialized()
        lines = ["[长期记忆]"]
        for name, col in self._collections.items():
            try:
                count = col.count()
                lines.append(f"  {name}: {count} 条")
            except Exception:
                lines.append(f"  {name}: 获取失败")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summarize()
