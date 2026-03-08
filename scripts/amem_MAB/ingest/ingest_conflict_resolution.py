import argparse
import sys
from pathlib import Path
import spacy
from neo4j import GraphDatabase

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices,load_benchmark_data
from src.config import get_config

# 初始化NLP模型（实体提取）
nlp = spacy.load("en_core_web_sm")  # 中文：zh_core_web_sm，英文：en_core_web_sm
logger = get_logger()

# 加载配置
cfg = get_config()
neo4j_cfg = cfg._app_config.get("neo4j", {})

def chunk_facts(context: str, chunk_size: int = 800):
    """
    Conflict Resolution 专用切分策略：
    按行读取 Fact，累积直到缓冲区字符数 > chunk_size，然后作为一个 Chunk。
    """
    lines = [line.strip() for line in context.split('\n') if line.strip()]
    
    chunks = []
    current_chunk_lines = []
    current_length = 0
    
    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)
        
        if current_length > chunk_size:
            # 形成一个 chunk
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append(chunk_text)
            # 重置缓冲区
            current_chunk_lines = []
            current_length = 0
            
    # 处理剩余的缓冲区
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunks.append(chunk_text)
        
    return chunks

class Neo4jEntityIngestor:
    """专门用于提取实体+关系并写入Neo4j的类"""
    def __init__(self):
        self.uri = neo4j_cfg.get("uri")
        self.user = neo4j_cfg.get("user")
        self.password = neo4j_cfg.get("password")
        print(f"[Neo4jEntityIngestor] connect: {self.uri} user={self.user}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def close(self):
        self.driver.close()
    
    def extract_entities(self, text: str):
        """从文本中提取实体（人名、地名、机构、事件等）"""
        doc = nlp(text)
        entities = []
        # 提取实体：label=实体类型（PERSON/LOC/ORG等），text=实体名称
        for ent in doc.ents:
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        # 去重（同一实体多次出现只保留一个）
        unique_entities = []
        seen_ents = set()
        for ent in entities:
            ent_key = (ent["text"], ent["label"])
            if ent_key not in seen_ents:
                seen_ents.add(ent_key)
                unique_entities.append(ent)
        return unique_entities
    
    def write_to_neo4j(self, chunk: str, chunk_id: int, instance_idx: int):
        """写入Neo4j：Entity节点 + Memory节点 + 关联关系"""
        # 1. 提取当前chunk的实体
        entities = self.extract_entities(chunk)
        logger.info(f"Chunk {chunk_id} extracted {len(entities)} entities")

        # 2. 创建/更新 Entity 节点 + 共现关系
        with self.driver.session() as session:
            for ent in entities:
                session.run("""
                    MERGE (e:Entity {text: $ent_text, label: $ent_label})
                    SET e.count = COALESCE(e.count, 0) + 1
                """, ent_text=ent["text"], ent_label=ent["label"])

            # 共现关系（可选）
            if len(entities) >= 2:
                for i in range(len(entities)):
                    for j in range(i+1, len(entities)):
                        ent1 = entities[i]
                        ent2 = entities[j]
                        session.run("""
                            MERGE (e1:Entity {text: $ent1_text, label: $ent1_label})
                            MERGE (e2:Entity {text: $ent2_text, label: $ent2_label})
                            MERGE (e1)-[r:CO_OCCURS_IN {chunk_id: $chunk_id}]->(e2)
                            SET r.count = COALESCE(r.count, 0) + 1
                        """, ent1_text=ent1["text"], ent1_label=ent1["label"],
                            ent2_text=ent2["text"], ent2_label=ent2["label"],
                            chunk_id=chunk_id)

        # 3. 创建 Memory 节点（AMem兼容）
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "A-mem"))
        from graphdb_retriever import Neo4jRetriever

        retriever = Neo4jRetriever()
        metadata = {
            "chunk_id": chunk_id,
            "instance_idx": instance_idx,
            "type": "document",
            "task": "conflict_resolution"
        }
        doc_id = f"conflict_doc_{chunk_id}"
        retriever.add_document(chunk, metadata, doc_id)

        # 4. Memory 直接挂 Entity
        with self.driver.session() as session:
            for ent in entities:
                session.run("""
                    MERGE (m:Memory {id: $mid})
                    MERGE (e:Entity {text: $ent_text, label: $ent_label})
                    MERGE (m)-[:HAS_ENTITY]->(e)
                """, mid=doc_id,
                    ent_text=ent["text"], ent_label=ent["label"])

        

def ingest_one_instance(instance_idx: int, chunk_size: int, shard: str):

    logger.info(f"=== Processing Instance {instance_idx} ===")
    # 加载数据（保留你原有逻辑）
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        json_data_path = project_root / "MemoryAgentBench" / "data" / f"Conflict_Resolution-00000-of-{shard}.json"
        parquet_data_path = project_root / "MemoryAgentBench" / "data" / f"Conflict_Resolution-00000-of-{shard}.parquet"

        
        if json_data_path.exists():
            data = load_benchmark_data(str(json_data_path), instance_idx)
        else:
            data = load_benchmark_data(str(parquet_data_path), instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # 使用专用切分策略
    chunks = chunk_facts(data["context"], chunk_size=chunk_size)

    logger.info(f"Ingesting entities into Neo4j")
    # 初始化Neo4j实体写入器
    neo4j_ingestor = Neo4jEntityIngestor()
    
    print(f"Starting ingestion of {len(chunks)} chunks (extract entities)...")
    for i, chunk in enumerate(chunks):
        # 写入Neo4j（提取实体+创建节点/关系）
        neo4j_ingestor.write_to_neo4j(chunk, i, instance_idx)
        if i % 10 == 0:
            print(f"Ingested {i}/{len(chunks)} chunks...", end="\r", flush=True)
    print(f"\nIngestion complete. {len(chunks)} chunks processed, entities stored in Neo4j.")
    
    # 关闭Neo4j连接
    # neo4j_ingestor.close()
    logger.warning("注意：Neo4j连接未自动关闭，请手动执行close()或在Neo4j端断开连接！")

def main():
    parser = argparse.ArgumentParser(description="Ingest Conflict_Resolution data into Neo4j (Entity+Relation)")
    parser.add_argument("--instance_idx", type=str, default="0-7", help="Index range (e.g., '0-7')")
    parser.add_argument("--chunk_size", type=int, default=800, help="Minimum chars per chunk")
    parser.add_argument("--shard", type=str, default="00001",
                    help="Shard id, e.g. 00001 or 00002")

    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.chunk_size, args.shard)


if __name__ == "__main__":
    main()
