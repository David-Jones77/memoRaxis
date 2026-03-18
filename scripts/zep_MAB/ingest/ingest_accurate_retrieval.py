import argparse
import asyncio
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml

import spacy
from dotenv import load_dotenv
from zep_cloud.client import AsyncZep

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.logger import get_logger
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices

logger = get_logger()

# 初始化 NLP 模型
nlp = spacy.load("en_core_web_sm")
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


class ZepGraphEntityIngestor:


    def __init__(self, api_key: str, graph_id: str):
        self.api_key = api_key
        self.graph_id = graph_id
        self.zep = AsyncZep(api_key=api_key)
        print("graph methods:", dir(self.zep.graph))

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



    def extract_entities(self, text: str):
        """
        提取实体并去重
        返回:
        [
            {
                "text": "...",
                "label": "PERSON",
                "start": 0,
                "end": 4
            },
            ...
        ]
        """
        doc = nlp(text)
        entities = []

        for ent in doc.ents:
            ent_text = ent.text.strip()
            if not ent_text:
                continue
            entities.append(
                {
                    "text": ent_text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )

        # 去重：同一 chunk 中同名同类型只保留一个
        unique_entities = []
        seen = set()
        for ent in entities:
            key = (ent["text"], ent["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)

        return unique_entities

    def build_cooccurrence_relations(self, entities: list[dict]):
        """
        基于同一 chunk 内共现构造简单关系
        返回:
        [
            {
                "source": "Alice",
                "source_label": "PERSON",
                "target": "Google",
                "target_label": "ORG",
                "relation": "CO_OCCURS_WITH"
            },
            ...
        ]
        """
        relations = []
        n = len(entities)

        for i in range(n):
            for j in range(i + 1, n):
                e1 = entities[i]
                e2 = entities[j]

                relations.append(
                    {
                        "source": e1["text"],
                        "source_label": e1["label"],
                        "target": e2["text"],
                        "target_label": e2["label"],
                        "relation": "CO_OCCURS_WITH",
                    }
                )

        return relations

    async def add_chunk(
        self,
        chunk: str,
        chunk_id: int,
        instance_idx: int,
        created_at: str,
        metadata: dict | None = None,
    ):
        metadata = metadata or {}

        entities = self.extract_entities(chunk)
        relations = self.build_cooccurrence_relations(entities)

        logger.info(
            f"[graph={self.graph_id}] chunk={chunk_id}, entities={len(entities)}, relations={len(relations)}"
        )

        await self.zep.graph.add(
            graph_id=self.graph_id,
            type="text",
            data=chunk,
            created_at=created_at,
            source_description=f"document_chunk chunk_id={chunk_id} instance_idx={instance_idx}",
        )





    async def add_entity_summary(
        self,
        entity_stats: dict,
        instance_idx: int,
        created_at: str,
        metadata: dict | None = None,
    ):
        metadata = metadata or {}

        summary_entities = []
        for (text, label), count in entity_stats.items():
            summary_entities.append(
                {
                    "text": text,
                    "label": label,
                    "count": count,
                }
            )

        summary_entities.sort(key=lambda x: x["count"], reverse=True)

        top_entities = summary_entities[:50]

        summary_text = "Entity summary for this instance:\n" + "\n".join(
            f"- {item['text']} ({item['label']}), count={item['count']}"
            for item in top_entities
        )

        await self.zep.graph.add(
            graph_id=self.graph_id,
            type="text",
            data=summary_text,
            created_at=created_at,
            source_description=f"entity_summary instance_idx={instance_idx}",
        )





async def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    shard: str,
    prefix: str,
    reset: bool = True,
):
    logger.info(f"=== Processing Instance {instance_idx} ===")

    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent

        json_data_path = (
            project_root
            / "MemoryAgentBench"
            / "data"
            / f"Accurate_Retrieval-00000-of-{shard}.json"
        )
        parquet_data_path = (
            project_root
            / "MemoryAgentBench"
            / "data"
            / f"Accurate_Retrieval-00000-of-{shard}.parquet"
        )

        if json_data_path.exists():
            data = load_benchmark_data(str(json_data_path), instance_idx)
            logger.info(f"Loaded data from JSON: {json_data_path}")
        else:
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

    chunks = chunk_context(context, chunk_size=chunk_size)
    graph_id = f"{prefix}_accurate_retrieval_graph_{instance_idx}"

    config = load_config()
    zep_config = config.get("zep", {})

    api_key = zep_config.get("api_key")
    if not api_key:
        raise ValueError("zep.api_key is not set in config.yaml")

    logger.info(f"Ingesting instance {instance_idx} into Zep graph: {graph_id}")
    logger.info(f"Total chunks: {len(chunks)}")

    ingestor = ZepGraphEntityIngestor(api_key=api_key, graph_id=graph_id)
    import inspect
    print("graph.add signature:", inspect.signature(ingestor.zep.graph.add))
    print("graph.add doc:", ingestor.zep.graph.add.__doc__)
    print("episode methods:", dir(ingestor.zep.graph.episode))



    if reset:
        await ingestor.reset_graph()

    # 如果你的 Zep 环境支持显式 ontology，可以在这里补
    # 例如：
    # await ingestor.zep.graph.set_ontology(
    #     graph_id=graph_id,
    #     ontology={
    #         "entities": ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "EVENT"],
    #         "relations": ["CO_OCCURS_WITH", "MENTIONED_IN"]
    #     }
    # )

    print(f"Starting ingestion of {len(chunks)} chunks into {graph_id}...")

    base_time = datetime.now(timezone.utc)
    global_entity_stats = defaultdict(int)
    await ingestor.ensure_graph()

    for i, chunk in enumerate(chunks):
        # 先本地提取一次，顺便累计全局实体频次
        entities = ingestor.extract_entities(chunk)
        for ent in entities:
            global_entity_stats[(ent["text"], ent["label"])] += 1

        created_at = (base_time + timedelta(seconds=i)).isoformat()

        await ingestor.add_chunk(
            chunk=chunk,
            chunk_id=i,
            instance_idx=instance_idx,
            created_at=created_at,
            metadata=make_json_safe({
    "source": metadata.get("source", "accurate_retrieval"),
    "qa_pair_ids": metadata.get("qa_pair_ids"),
    "num_questions": len(questions),
    "num_answers": len(answers),
}),

        )

        if i % 10 == 0:
            print(f"Ingested {i}/{len(chunks)} chunks...", end="\r", flush=True)

    # 可选：再补一条 summary 节点
    summary_created_at = (base_time + timedelta(seconds=len(chunks) + 1)).isoformat()
    await ingestor.add_entity_summary(
        entity_stats=global_entity_stats,
        instance_idx=instance_idx,
        created_at=summary_created_at,
        metadata={
            "source": metadata.get("source", "accurate_retrieval"),
            "qa_pair_ids": metadata.get("qa_pair_ids"),
        },
    )

    print(f"\nIngestion complete. {len(chunks)} chunks processed into graph {graph_id}.")
    logger.info(
        f"Finished instance {instance_idx}: "
        f"graph_id={graph_id}, chunks={len(chunks)}, "
        f"questions={len(questions)}, answers={len(answers)}, "
        f"unique_entities={len(global_entity_stats)}"
    )


async def main_async():
    parser = argparse.ArgumentParser(
        description="Ingest Accurate Retrieval data into Zep with entity extraction"
    )
    parser.add_argument(
        "--instance_idx",
        type=str,
        default="0",
        help="Index range (e.g., '0', '0-5', '1,3')",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=850,
        help="Fallback chunk size",
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
        "--no_reset",
        action="store_true",
        help="Do not delete existing graph before ingest",
    )

    args = parser.parse_args()
    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        await ingest_one_instance(
            instance_idx=idx,
            chunk_size=args.chunk_size,
            shard=args.shard,
            prefix=args.prefix,
            reset=not args.no_reset,
        )


def main():
    load_dotenv()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
