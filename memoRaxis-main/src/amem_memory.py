from __future__ import annotations

from typing import Any, Dict, List, Optional
import sys
from pathlib import Path
import logging

from .memory_interface import BaseMemorySystem, Evidence

logger = logging.getLogger(__name__)


def _add_amem_to_syspath() -> None:
    root = Path(__file__).resolve().parents[1]  # memoRaxis-main
    amem_dir = root / "A-mem"
    if not amem_dir.exists():
        raise FileNotFoundError(f"Cannot find A-mem directory at: {amem_dir}")
    p = str(amem_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


class AMemMemorySystem(BaseMemorySystem):
    """BaseMemorySystem adapter over A-mem/AgenticMemorySystem."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        enable_llm: bool = False,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        evo_threshold: int = 10**9,  # avoid evolution side-effects by default
    ):
        _add_amem_to_syspath()
        from memory_system import AgenticMemorySystem  # in A-mem/

        self._ms = AgenticMemorySystem(
            model_name=model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )

        # Default: disable LLM network calls for evaluation stability
        if not enable_llm:
            self._ms.analyze_content = lambda content: {"keywords": [], "context": "General", "tags": []}
            self._ms.process_memory = lambda note: (False, note)

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        metadata = metadata or {}
        time_val = metadata.get("time") or metadata.get("timestamp")

        kwargs: Dict[str, Any] = {}
        for k in [
            "id",
            "keywords",
            "links",
            "retrieval_count",
            "last_accessed",
            "context",
            "evolution_history",
            "category",
            "tags",
        ]:
            if k in metadata:
                kwargs[k] = metadata[k]

        self._ms.add_note(content=data, time=time_val, **kwargs)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        results = self._ms.search(query, k=top_k) or []
        evidences: List[Evidence] = []

        for r in results[:top_k]:
            if not isinstance(r, dict):
                evidences.append(Evidence(content=str(r), metadata={"source": "A-mem", "raw": r}))
                continue

            md = dict(r)
            md.setdefault("source", "A-mem")
            # 这里的 score 是 chroma distance（越小越相似），先原样输出给 adaptor 打日志
            md.setdefault("score", md.get("score"))
            evidences.append(Evidence(content=md.get("content", ""), metadata=md))

        return evidences

    def reset(self) -> None:
        """重置记忆系统"""
        try:
            # 清除内存中的记忆
            self._ms.memories.clear()
            # 重置 ChromaDB 客户端
            self._ms.retriever.client.reset()
            # 重新创建集合
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._ms.retriever.collection = self._ms.retriever.client.get_or_create_collection(
                name="memories",
                embedding_function=SentenceTransformerEmbeddingFunction(model_name=self._ms.model_name)
            )
            logger.info("AMemMemorySystem 重置成功")
        except Exception as e:
            logger.warning("Chroma reset failed: %s", e)
