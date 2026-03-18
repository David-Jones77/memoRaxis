import argparse
import asyncio
import sys
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
from zep_cloud.client import AsyncZep

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices, load_benchmark_data

logger = get_logger()


def load_config():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent

    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]

    if isinstance(obj, Path):
        return str(obj)

    if hasattr(obj, "tolist"):
        try:
            return make_json_safe(obj.tolist())
        except Exception:
            pass

    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    return str(obj)


def chunk_dialogues(context: str) -> List[str]:
    """
    策略 A: 针对 Dialogue N: 格式的正则切分
    """
    parts = re.split(r'\n(Dialogue \d+:)', '\n' + context)
    chunks = []

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        full_text = f"{header}\n{body}"
        if len(full_text) > 10:
            chunks.append(full_text)

    return chunks


def chunk_accumulation(context: str, min_chars: int = 800) -> List[str]:
    """
    策略 B: 累积切分
    """
    lines = [line.strip() for line in context.split('\n') if line.strip()]
    chunks = []
    current_chunk_lines = []
    current_length = 0

    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)

        if current_length > min_chars:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_length = 0

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks


class ZepGraphIngestor:
    """
    Test-Time 用的简化版 Zep ingestor:
    1. 确保 graph 存在
    2. 将 chunk 直接写入 Zep graph
    """

    def __init__(self, api_key: str, graph_id: str):
        self.api_key = api_key
        self.graph_id = graph_id
        self.zep = AsyncZep(api_key=api_key)

    async def ensure_graph(self):
        try:
            await self.zep.graph.get(self.graph_id)
            logger.info(f"Graph exists: {self.graph_id}")
        except Exception:
            logger.info(f"Graph not found, creating: {self.graph_id}")
            await self.zep.graph.create(graph_id=self.graph_id)
            logger.info(f"Graph created: {self.graph_id}")

    async def reset_graph(self):
        try:
            logger.info(f"Trying to delete graph first: {self.graph_id}")
            await self.zep.graph.delete(self.graph_id)
            logger.info(f"Deleted graph: {self.graph_id}")
        except Exception as e:
            logger.info(f"Skip deleting graph {self.graph_id}: {e}")

        await self.ensure_graph()

    async def add_chunk(
        self,
        chunk: str,
        chunk_id: int,
        instance_idx: int,
        created_at: str,
        metadata: dict | None = None,
    ):
        metadata = metadata or {}

        logger.info(
            f"[graph={self.graph_id}] chunk={chunk_id}, chars={len(chunk)}"
        )

        # 如果你当前 zep SDK 的 graph.add 支持 metadata，可以加上 metadata=metadata
        # 若不支持则先只传 source_description / created_at / data
        await self.zep.graph.add(
            graph_id=self.graph_id,
            type="text",
            data=chunk,
            created_at=created_at,
            source_description=(
                f"test_time_chunk chunk_id={chunk_id} instance_idx={instance_idx}"
            ),
        )


async def ingest_one_instance(
    instance_idx: int,
    shard: str,
    prefix: str,
    reset: bool = True,
    min_chars: int = 800,
    max_chunks: int | None = None,
):
    logger.info(f"=== Processing Instance {instance_idx} ===")

    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent

        parquet_data_path = (
            project_root
            / "MemoryAgentBench"
            / "data"
            / f"Test_Time_Learning-00000-of-{shard}.parquet"
        )

        data = load_benchmark_data(str(parquet_data_path), instance_idx)
        logger.info(f"Loaded data from Parquet: {parquet_data_path}")

    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    context = data.get("context", "")
    if not context or not context.strip():
        logger.warning(f"Instance {instance_idx} has empty context, skipping.")
        return

    questions = data.get("questions", [])
    answers = data.get("answers", [])
    metadata = data.get("metadata", {})

    # chunk 逻辑参考 A-Mem test_time ingest
    if "Dialogue 1:" in context[:500]:
        logger.info("Strategy: Regex Split (Dialogue mode)")
        all_chunks = chunk_dialogues(context)
    else:
        logger.info(f"Strategy: Accumulation > {min_chars} chars (ShortText mode)")
        all_chunks = chunk_accumulation(context, min_chars=min_chars)

    chunks = all_chunks[:max_chunks] if max_chunks is not None else all_chunks

    logger.info(f"Total chunks before limit: {len(all_chunks)}")
    logger.info(f"Total chunks to ingest: {len(chunks)}")

    graph_id = f"{prefix}_test_time_graph_{instance_idx}"

    config = load_config()
    zep_config = config.get("zep", {})
    api_key = zep_config.get("api_key")
    if not api_key:
        raise ValueError("zep.api_key is not set in config.yaml")

    logger.info(f"Ingesting instance {instance_idx} into Zep graph: {graph_id}")

    ingestor = ZepGraphIngestor(api_key=api_key, graph_id=graph_id)

    if reset:
        await ingestor.reset_graph()
    else:
        await ingestor.ensure_graph()

    print(f"Starting ingestion of {len(chunks)} chunks into {graph_id}...")

    base_time = datetime.now(timezone.utc)

    for i, chunk in enumerate(chunks):
        created_at = (base_time + timedelta(seconds=i)).isoformat()

        await ingestor.add_chunk(
            chunk=chunk,
            chunk_id=i,
            instance_idx=instance_idx,
            created_at=created_at,
            metadata=make_json_safe({
                "task": "test_time",
                "source": metadata.get("source", "test_time_learning"),
                "num_questions": len(questions) if isinstance(questions, list) else 0,
                "num_answers": len(answers) if isinstance(answers, list) else 0,
                "chunk_strategy": (
                    "dialogue_regex"
                    if "Dialogue 1:" in context[:500]
                    else "accumulation"
                ),
                "min_chars": min_chars,
            }),
        )

        if i % 100 == 0:
            print(f"Ingested {i}/{len(chunks)} chunks...", end="\r", flush=True)

    print(f"\nIngestion complete. {len(chunks)} chunks processed into graph {graph_id}.")
    logger.info(
        f"Finished instance {instance_idx}: "
        f"graph_id={graph_id}, chunks={len(chunks)}, "
        f"questions={len(questions) if isinstance(questions, list) else 0}, "
        f"answers={len(answers) if isinstance(answers, list) else 0}"
    )


async def main_async():
    parser = argparse.ArgumentParser(
        description="Ingest Test_Time_Learning data into Zep"
    )
    parser.add_argument(
        "--instance_idx",
        type=str,
        default="0-5",
        help="Index range (e.g., '0', '0-5', '1,3')",
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
        help="Prefix for Zep graph namespace",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=800,
        help="Min chars for accumulation chunking",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to ingest",
    )
    parser.add_argument(
        "--no_reset",
        action="store_true",
        help="Do not delete existing graph before ingest",
    )

    args = parser.parse_args()
    indices = parse_instance_indices(args.instance_idx)

    logger.info(f"Target instances: {indices}")
    logger.info(f"Config: min_chars={args.min_chars}, max_chunks={args.max_chunks}")

    for idx in indices:
        await ingest_one_instance(
            instance_idx=idx,
            shard=args.shard,
            prefix=args.prefix,
            reset=not args.no_reset,
            min_chars=args.min_chars,
            max_chunks=args.max_chunks,
        )


def main():
    load_dotenv()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
