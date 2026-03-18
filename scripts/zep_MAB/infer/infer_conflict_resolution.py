# -*- coding: utf-8 -*-
"""
Zep infer pipeline for Conflict_resolution.

"""

import argparse
import asyncio
import json
import sys
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

        # evaluation 里 nodes / edges 都会参与 compose context
        # 这里同样保留两类证据
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
        """
        infer 阶段通常不 reset。
        保留这个接口是为了兼容 BaseMemorySystem 风格。
        """
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
        / f"Conflict_Resolution-00000-of-{shard}.json"
    )
    parquet_data_path = (
        project_root
        / "MemoryAgentBench"
        / "data"
        / f"Conflict_Resolution-00000-of-{shard}.parquet"
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

    graph_id = f"{prefix}_conflict_resolution_graph_{instance_idx}"

    memory = ZepGraphMemorySystem(
        api_key=api_key,
        graph_id=graph_id,
        node_limit=node_limit if node_limit is not None else graph_conf.get("node_limit", 8),
        edge_limit=edge_limit if edge_limit is not None else graph_conf.get("edge_limit", 16),
        node_reranker=node_reranker if node_reranker is not None else graph_conf.get("node_reranker", "rrf"),
        edge_reranker=edge_reranker if edge_reranker is not None else graph_conf.get("edge_reranker", "rrf"),
    )

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
                SingleTurnAdaptor(llm, memory),
                questions,
                limit,
            )

        if "all" in adaptors_to_run or "R2" in adaptors_to_run:
            results["R2"] = evaluate_adaptor(
                "R2",
                IterativeAdaptor(llm, memory),
                questions,
                limit,
            )

        if "all" in adaptors_to_run or "R3" in adaptors_to_run:
            results["R3"] = evaluate_adaptor(
                "R3",
                PlanAndActAdaptor(llm, memory),
                questions,
                limit,
            )
    finally:
        memory.close()

    final_report = {
        "dataset": "Conflict_Resolution",
        "instance_idx": instance_idx,
        "memory_system": "ZepGraphMemorySystem",
        "graph_id": graph_id,
        "retrieval_config": {
            "node_limit": memory.node_limit,
            "edge_limit": memory.edge_limit,
            "node_reranker": memory.node_reranker,
            "edge_reranker": memory.edge_reranker,
        },
        "results": results,
    }

    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    filename = f"zep_conflict_res_results_{instance_idx}"
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
