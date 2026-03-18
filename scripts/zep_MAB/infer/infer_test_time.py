# -*- coding: utf-8 -*-
"""
Zep infer pipeline for Test_Time.

改动说明：
1. 仅对 R1 / SingleTurnAdaptor 启用 multi-query retrieval
2. R2 / R3 保持原始单 query 检索逻辑
3. 适配 Zep search query ~400 chars 的限制
"""

import argparse
import asyncio
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional
import yaml

from dotenv import load_dotenv
from zep_cloud.client import AsyncZep
from zep_cloud.core.api_error import ApiError

# Add project root to sys.path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.logger import get_logger
from src.config import get_config
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import (
    SingleTurnAdaptor,
    IterativeAdaptor,
    PlanAndActAdaptor,
    AdaptorResult,
)
from src.memory_interface import BaseMemorySystem, Evidence

logger = get_logger()


def load_yaml_config():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    config_path = project_root / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _split_turns(question: str) -> list[str]:
    lines = [line.strip() for line in question.split("\n") if line.strip()]
    turns = [x for x in lines if x.startswith("User:") or x.startswith("System:")]
    if turns and turns[-1] == "System:":
        turns = turns[:-1]
    return turns


def _clip_query(text: str, max_chars: int = 380) -> str:
    text = " ".join(text.split())
    return text[:max_chars].strip()


def _extract_entities_from_recent_text(text: str, max_entities: int = 8) -> list[str]:
    entities = re.findall(r'([A-Z][A-Za-z0-9\'\-\s:/,&]+?$\d{4}$)', text)
    entities = [e.strip() for e in entities if e.strip()]
    deduped = list(OrderedDict.fromkeys(entities))
    return deduped[:max_entities]


def build_testtime_multi_queries(
    question: str,
    max_turns: int = 6,
    max_chars: int = 380,
) -> list[str]:
    """
    将超长对话 question 改写成多个适配 Zep 的短 query。
    每条 query <= max_chars，避免触发 400-char truncate。
    """
    q = question.strip()
    if not q:
        return []

    turns = _split_turns(q)
    if not turns:
        return [_clip_query(q, max_chars=max_chars)]

    recent_turns = turns[-max_turns:]
    recent_block = "\n".join(recent_turns)
    recent_text = " ".join(recent_turns)

    user_msgs = [t[len("User:"):].strip() for t in recent_turns if t.startswith("User:")]
    system_msgs = [t[len("System:"):].strip() for t in recent_turns if t.startswith("System:")]
    entities = _extract_entities_from_recent_text(recent_text, max_entities=8)

    queries = []

    # 1) 最近对话
    recent_query = _clip_query(recent_block, max_chars=max_chars)
    if recent_query:
        queries.append(recent_query)

    # 2) 用户最近表达 / 偏好
    if user_msgs:
        q_user = "user preferences and recent user mentions: " + " | ".join(user_msgs[-3:])
        queries.append(_clip_query(q_user, max_chars=max_chars))

    # 3) assistant 最近表达 / 偏好
    if system_msgs:
        q_sys = "assistant preferences and recent assistant mentions: " + " | ".join(system_msgs[-3:])
        queries.append(_clip_query(q_sys, max_chars=max_chars))

    # 4) 实体 query
    if entities:
        q_ent = "recent mentioned entities: " + ", ".join(entities)
        queries.append(_clip_query(q_ent, max_chars=max_chars))

    deduped = []
    seen = set()
    for item in queries:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped


class ZepGraphMemorySystem(BaseMemorySystem):
    """
    基于 Zep graph 的 memory system。

    检索逻辑参考 Zep evaluation:
    - 对 query 并行检索 nodes / edges
    - 将 nodes / edges 转换为 adaptor 可消费的 Evidence 列表
    """

    def __init__(
        self,
        api_key: str,
        graph_id: str,
        node_limit: int = 8,
        edge_limit: int = 16,
        node_reranker: str = "rrf",
        edge_reranker: str = "rrf",
    ):
        self.api_key = api_key
        self.graph_id = graph_id
        self.node_limit = node_limit
        self.edge_limit = edge_limit
        self.node_reranker = node_reranker
        self.edge_reranker = edge_reranker

        self.zep = AsyncZep(api_key=api_key)

        # 因为 adaptor 是同步调用 memory.retrieve()，这里维护一个私有 event loop
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # 启动时检查 graph 存在
        self._run(self._ensure_graph_exists())

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def add_memory(self, data: str, metadata: dict) -> None:
        logger.info(
            f"ZepGraphMemorySystem.add_memory() called during infer, no-op. "
            f"graph={self.graph_id}, data_preview={data[:80]!r}"
        )

    async def _ensure_graph_exists(self):
        try:
            await self.zep.graph.get(self.graph_id)
            logger.info(f"Using existing Zep graph: {self.graph_id}")
        except Exception as e:
            raise RuntimeError(
                f"Graph not found or inaccessible: graph_id={self.graph_id}, error={e}"
            )

    async def _graph_search_with_retry(
        self,
        query: str,
        scope: str,
        reranker: str,
        limit: int,
        max_retries: int = 3,
    ):
        last_error = None
        for attempt in range(max_retries):
            try:
                return await self.zep.graph.search(
                    query=query,
                    graph_id=self.graph_id,
                    scope=scope,
                    reranker=reranker,
                    limit=limit,
                )
            except ApiError as e:
                last_error = e
                wait_time = min(10, 0.2 * (2 ** attempt))
                logger.warning(
                    f"[Zep search retry] graph={self.graph_id}, scope={scope}, "
                    f"attempt={attempt + 1}/{max_retries}, error={e}, wait={wait_time}s"
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                last_error = e
                break

        raise RuntimeError(
            f"Zep graph search failed: graph_id={self.graph_id}, scope={scope}, error={last_error}"
        )

    def _normalize_retrieval_score(self, obj) -> tuple[float, float | None, float | None, float | None]:
        """
        统一检索分数语义为：越大越相关

        返回:
            display_score: 推荐用于展示/排序的分数
            raw_relevance: Zep relevance（若有）
            raw_score: Zep 原始 reranker score
            raw_distance: Zep 原始 distance
        """
        raw_relevance = getattr(obj, "relevance", None)
        raw_score = getattr(obj, "score", None)
        raw_distance = getattr(obj, "distance", None)

        if isinstance(raw_relevance, (int, float)):
            return (
                float(raw_relevance),
                float(raw_relevance),
                float(raw_score) if isinstance(raw_score, (int, float)) else None,
                float(raw_distance) if isinstance(raw_distance, (int, float)) else None,
            )

        if isinstance(raw_score, (int, float)):
            return (
                float(raw_score),
                None,
                float(raw_score),
                float(raw_distance) if isinstance(raw_distance, (int, float)) else None,
            )

        if isinstance(raw_distance, (int, float)):
            display_score = 1.0 / (1.0 + float(raw_distance))
            return display_score, None, None, float(raw_distance)

        return 0.0, None, None, None

    def _edge_to_evidence(self, edge) -> Evidence:
        fact = getattr(edge, "fact", None) or ""
        valid_at = getattr(edge, "valid_at", None)

        display_score, raw_relevance, raw_score, raw_distance = self._normalize_retrieval_score(edge)

        content = fact
        if valid_at:
            content = f"{fact} (event_time: {valid_at})"

        return Evidence(
            content=content,
            metadata={
                "source": "zep_edge",
                "graph_id": self.graph_id,
                "scope": "edges",
                "score": display_score,
                "relevance": raw_relevance,
                "raw_score": raw_score,
                "raw_distance": raw_distance,
                "event_time": str(valid_at) if valid_at is not None else "",
            },
        )

    def _node_to_evidence(self, node) -> Evidence:
        name = getattr(node, "name", None) or ""
        summary = getattr(node, "summary", None) or ""

        display_score, raw_relevance, raw_score, raw_distance = self._normalize_retrieval_score(node)

        content = f"{name}: {summary}" if summary else name

        return Evidence(
            content=content,
            metadata={
                "source": "zep_node",
                "graph_id": self.graph_id,
                "scope": "nodes",
                "score": display_score,
                "relevance": raw_relevance,
                "raw_score": raw_score,
                "raw_distance": raw_distance,
                "entity_name": name,
            },
        )

    async def _retrieve_async(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        参考 Zep evaluation 的检索方式：
        - 并行搜 nodes / edges
        - 再转成 Evidence 列表

        top_k 在 adaptor 侧传入，这里将其映射到 node/edge 的 limit。
        """
        node_limit = min(self.node_limit, top_k) if top_k > 0 else self.node_limit
        edge_limit = min(self.edge_limit, max(top_k, 1) * 2) if top_k > 0 else self.edge_limit

        search_results = await asyncio.gather(
            self._graph_search_with_retry(
                query=query,
                scope="nodes",
                reranker=self.node_reranker,
                limit=node_limit,
            ),
            self._graph_search_with_retry(
                query=query,
                scope="edges",
                reranker=self.edge_reranker,
                limit=edge_limit,
            ),
        )

        node_result = search_results[0]
        edge_result = search_results[1]

        nodes = getattr(node_result, "nodes", []) or []
        edges = getattr(edge_result, "edges", []) or []

        logger.info(
            f"[Zep retrieve] graph={self.graph_id}, query={query[:100]!r}, "
            f"nodes={len(nodes)}, edges={len(edges)}"
        )

        evidences: List[Evidence] = []

        for edge in edges:
            try:
                evidences.append(self._edge_to_evidence(edge))
            except Exception as e:
                logger.warning(f"Failed to convert edge to evidence: {e}")

        for node in nodes:
            try:
                evidences.append(self._node_to_evidence(node))
            except Exception as e:
                logger.warning(f"Failed to convert node to evidence: {e}")

        return evidences

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        return self._run(self._retrieve_async(query, top_k=top_k))

    def reset(self):
        logger.info(
            f"ZepGraphMemorySystem.reset() called, no-op for infer. graph={self.graph_id}"
        )

    def close(self):
        try:
            self._loop.run_until_complete(self.zep.aclose())
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass


class ZepGraphMemorySystemForR1(ZepGraphMemorySystem):
    """
    仅给 R1 / SingleTurnAdaptor 使用：
    - 长 query 自动拆成多个 <= 380 chars 的短 query
    - 分别检索 nodes / edges
    - 合并、去重、按 score 排序
    """

    def __init__(
        self,
        *args,
        multi_query_threshold: int = 400,
        max_query_chars: int = 380,
        max_turns: int = 6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_query_threshold = multi_query_threshold
        self.max_query_chars = max_query_chars
        self.max_turns = max_turns

    def _dedup_and_rank_evidences(self, evidences: List[Evidence], top_k: int) -> List[Evidence]:
        merged = {}
        for e in evidences:
            key = (
                e.metadata.get("source", ""),
                e.metadata.get("scope", ""),
                e.metadata.get("entity_name", ""),
                e.content.strip(),
            )
            score = e.metadata.get("score", 0.0) or 0.0

            if key not in merged:
                merged[key] = e
            else:
                old_score = merged[key].metadata.get("score", 0.0) or 0.0
                if score > old_score:
                    merged[key] = e

        ranked = sorted(
            merged.values(),
            key=lambda x: x.metadata.get("score", 0.0) or 0.0,
            reverse=True,
        )

        final_k = max(top_k * 2, top_k)
        return ranked[:final_k]

    async def _retrieve_multi_query_async(self, query: str, top_k: int = 5) -> List[Evidence]:
        queries = build_testtime_multi_queries(
            question=query,
            max_turns=self.max_turns,
            max_chars=self.max_query_chars,
        )

        if not queries:
            return []

        logger.info(
            f"[R1 multi-query] original_len={len(query)}, subqueries={len(queries)}, "
            f"graph={self.graph_id}"
        )
        for i, q in enumerate(queries):
            logger.info(f"[R1 multi-query] subquery[{i}]={q!r}")

        all_evidences: List[Evidence] = []
        for subq in queries:
            try:
                evs = await super()._retrieve_async(subq, top_k=top_k)
                all_evidences.extend(evs)
            except Exception as e:
                logger.warning(f"[R1 multi-query] subquery failed: {e}")

        return self._dedup_and_rank_evidences(all_evidences, top_k=top_k)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if len(query) <= self.multi_query_threshold:
            return super().retrieve(query, top_k=top_k)

        return self._run(self._retrieve_multi_query_async(query, top_k=top_k))


def evaluate_adaptor(name: str, adaptor, questions: list, limit: int) -> list:
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)

    for i, q in enumerate(target_questions):
        logger.info(f"[{name}] Running Q{i+1}/{total}: {q}")
        try:
            res: AdaptorResult = adaptor.run(q)
            results.append(
                {
                    "question": q,
                    "answer": res.answer,
                    "steps": res.steps_taken,
                    "tokens": res.token_consumption,
                    "replan": res.replan_count,
                    "evidence_count": len(res.evidence_collected),
                    "evidences": [
                        {
                            "content": e.content,
                            "metadata": e.metadata,
                        }
                        for e in res.evidence_collected
                    ],
                }
            )
        except Exception as e:
            logger.error(f"[{name}] Failed on Q{i+1}: {e}")
            results.append({"question": q, "error": str(e)})
    return results


def evaluate_one_instance(
    instance_idx: int,
    adaptors_to_run: List[str],
    limit: int,
    output_suffix: str = "",
    shard: str = "00001",
    prefix: str = "bench",
    node_limit: Optional[int] = None,
    edge_limit: Optional[int] = None,
    node_reranker: Optional[str] = None,
    edge_reranker: Optional[str] = None,
):
    logger.info(f"=== Evaluating Instance {instance_idx} with Zep ===")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent

    json_data_path = (
        project_root
        / "MemoryAgentBench"
        / "data"
        / f"Test_Time_Learning-00000-of-{shard}.json"
    )
    parquet_data_path = (
        project_root
        / "MemoryAgentBench"
        / "data"
        / f"Test_Time_Learning-00000-of-{shard}.parquet"
    )

    try:
        if json_data_path.exists():
            data = load_benchmark_data(str(json_data_path), instance_idx)
            logger.info(f"Loaded data from JSON: {json_data_path}")
        else:
            data = load_benchmark_data(str(parquet_data_path), instance_idx)
            logger.info(f"Loaded data from Parquet: {parquet_data_path}")
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    questions = list(data["questions"])

    yaml_conf = load_yaml_config()
    zep_conf = yaml_conf.get("zep", {})
    graph_conf = yaml_conf.get("graph_params", {})

    api_key = zep_conf.get("api_key")
    if not api_key:
        raise ValueError("zep.api_key is not set in config.yaml")

    graph_id = f"{prefix}_test_time_graph_{instance_idx}"

    common_kwargs = dict(
        api_key=api_key,
        graph_id=graph_id,
        node_limit=node_limit if node_limit is not None else graph_conf.get("node_limit", 8),
        edge_limit=edge_limit if edge_limit is not None else graph_conf.get("edge_limit", 16),
        node_reranker=node_reranker if node_reranker is not None else graph_conf.get("node_reranker", "rrf"),
        edge_reranker=edge_reranker if edge_reranker is not None else graph_conf.get("edge_reranker", "rrf"),
    )

    memory_r1 = ZepGraphMemorySystemForR1(
        **common_kwargs,
        multi_query_threshold=400,
        max_query_chars=380,
        max_turns=6,
    )

    memory_base = ZepGraphMemorySystem(**common_kwargs)

    logger.info(f"Using ZepGraphMemorySystem for evaluation, graph_id={graph_id}")

    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"],
    )

    results = {}
    try:
        if "all" in adaptors_to_run or "R1" in adaptors_to_run:
            results["R1"] = evaluate_adaptor(
                "R1",
                SingleTurnAdaptor(llm, memory_r1),
                questions,
                limit,
            )

        if "all" in adaptors_to_run or "R2" in adaptors_to_run:
            results["R2"] = evaluate_adaptor(
                "R2",
                IterativeAdaptor(llm, memory_base),
                questions,
                limit,
            )

        if "all" in adaptors_to_run or "R3" in adaptors_to_run:
            results["R3"] = evaluate_adaptor(
                "R3",
                PlanAndActAdaptor(llm, memory_base),
                questions,
                limit,
            )
    finally:
        memory_r1.close()
        memory_base.close()

    final_report = {
        "dataset": "Test_Time_Learning",
        "instance_idx": instance_idx,
        "memory_system": "ZepGraphMemorySystem",
        "graph_id": graph_id,
        "retrieval_config": {
            "node_limit": memory_base.node_limit,
            "edge_limit": memory_base.edge_limit,
            "node_reranker": memory_base.node_reranker,
            "edge_reranker": memory_base.edge_reranker,
            "r1_multi_query": {
                "enabled": True,
                "threshold": 400,
                "max_query_chars": 380,
                "max_turns": 6,
            },
        },
        "results": results,
    }

    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    filename = f"zep_test_time_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = output_dir / filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Instance {instance_idx} Finished. Results saved to {output_file}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate Adaptors on MemoryAgentBench using ZepGraphMemorySystem"
    )
    parser.add_argument(
        "--adaptor",
        nargs="+",
        default=["all"],
        choices=["R1", "R2", "R3", "all"],
        help="Adaptors to run (e.g., R1 R2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of questions to run (-1 for all)",
    )
    parser.add_argument(
        "--instance_idx",
        type=str,
        default="0",
        help="Index range (e.g., '0-5', '1,3')",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix for output filename",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default="00001",
        help="Shard id, e.g. 00001 or 00002",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="bench",
        help="Graph prefix. Must match ingest prefix.",
    )
    parser.add_argument(
        "--node_limit",
        type=int,
        default=None,
        help="Override node retrieval limit",
    )
    parser.add_argument(
        "--edge_limit",
        type=int,
        default=None,
        help="Override edge retrieval limit",
    )
    parser.add_argument(
        "--node_reranker",
        type=str,
        default=None,
        help="Override node reranker",
    )
    parser.add_argument(
        "--edge_reranker",
        type=str,
        default=None,
        help="Override edge reranker",
    )

    args = parser.parse_args()
    indices = parse_instance_indices(args.instance_idx)

    for idx in indices:
        evaluate_one_instance(
            instance_idx=idx,
            adaptors_to_run=args.adaptor,
            limit=args.limit,
            output_suffix=args.output_suffix,
            shard=args.shard,
            prefix=args.prefix,
            node_limit=args.node_limit,
            edge_limit=args.edge_limit,
            node_reranker=args.node_reranker,
            edge_reranker=args.edge_reranker,
        )


if __name__ == "__main__":
    main()
