import argparse
import sys
from pathlib import Path
import spacy
from neo4j import GraphDatabase

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices, chunk_context,load_benchmark_data
from src.config import get_config

# 初始化NLP模型（实体提取）
nlp = spacy.load("en_core_web_sm")  # 中文：zh_core_web_sm，英文：en_core_web_sm
logger = get_logger()

# 加载配置
cfg = get_config()
neo4j_cfg = cfg._app_config.get("neo4j", {})

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
        """写入Neo4j：Document节点 + Entity节点 + 关联关系"""
        # 1. 提取当前chunk的实体
        entities = self.extract_entities(chunk)
        logger.info(f"Chunk {chunk_id} extracted {len(entities)} entities")
        
        with self.driver.session() as session:
            # 2. 创建Document节点（对应原chunk，记录完整文本）
            session.run("""
                MERGE (d:Document {chunk_id: $chunk_id, instance_idx: $instance_idx, task: 'long_range'})
                SET d.text = $text, d.created_at = timestamp()
            """, chunk_id=chunk_id, instance_idx=instance_idx, text=chunk)
            
            # 3. 创建Entity节点 + Document-Entity关联关系（"CONTAINS"：文档包含该实体）
            for ent in entities:
                # MERGE避免重复创建同一实体
                session.run("""
                    MERGE (e:Entity {text: $ent_text, label: $ent_label})
                    SET e.count = COALESCE(e.count, 0) + 1
                    
                    MERGE (d:Document {chunk_id: $chunk_id, instance_idx: $instance_idx, task: 'long_range'})
                    MERGE (d)-[r:CONTAINS {start: $start, end: $end}]->(e)
                """, ent_text=ent["text"], ent_label=ent["label"],
                     chunk_id=chunk_id, instance_idx=instance_idx,
                     start=ent["start"], end=ent["end"])
            
            # 4. （可选）提取实体间关系（进阶：比如“张三→参加→会议”）
            # 简单版：基于共现关系（同一chunk里的实体互相关联）
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
        
        # 5. 同时创建Memory节点，确保与AMemMemorySystem兼容
        import sys
        from pathlib import Path
        # 添加A-mem目录到搜索路径
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "A-mem"))
        from graphdb_retriever import Neo4jRetriever
        retriever = Neo4jRetriever()
        metadata = {
            "chunk_id": chunk_id,
            "instance_idx": instance_idx,
            "type": "document",
            "task": "long_range"
        }
        retriever.add_document(chunk, metadata, f"long_range_doc_{chunk_id}")

def ingest_one_instance(instance_idx: int, chunk_size: int, overlap: int,shard: str):
    
    logger.info(f"=== Processing Instance {instance_idx} ===")
    # 加载数据（保留你原有逻辑）
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        json_data_path = project_root / "MemoryAgentBench" / "data" / f"Long_Range_Understanding-00000-of-{shard}.json"
        parquet_data_path = project_root / "MemoryAgentBench" / "data" / f"Long_Range_Understanding-00000-of-{shard}.parquet"

        
        if json_data_path.exists():
            data = load_benchmark_data(str(json_data_path), instance_idx)
        else:
            data = load_benchmark_data(str(parquet_data_path), instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    

    # 使用滑动窗口切分
    # 注意：这里我们知道它是小说文本，没有 "Document N:"，chunk_context 会自动回退到滑动窗口
    chunks = chunk_context(data["context"], chunk_size=chunk_size, overlap=overlap)

    logger.info(f"Ingesting entities into Neo4j")
    # 初始化Neo4j实体写入器
    neo4j_ingestor = Neo4jEntityIngestor()
    
    print(f"Starting ingestion of {len(chunks)} chunks (extract entities)...")
    for i, chunk in enumerate(chunks):
        # 写入Neo4j（提取实体+创建节点/关系）
        neo4j_ingestor.write_to_neo4j(chunk, i, instance_idx)
        if i % 100 == 0:
            print(f"Ingested {i}/{len(chunks)} chunks...", end="\r", flush=True)
    print(f"\nIngestion complete. {len(chunks)} chunks processed, entities stored in Neo4j.")
    
    # 关闭Neo4j连接
    # neo4j_ingestor.close()
    logger.warning("注意：Neo4j连接未自动关闭，请手动执行close()或在Neo4j端断开连接！")

def main():
    parser = argparse.ArgumentParser(description="Ingest Long_Range_Understanding data into Neo4j (Entity+Relation)")
    # 默认 Top 40 (0-39)
    parser.add_argument("--instance_idx", type=str, default="0-39", help="Index range (e.g., '0-39')")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size for sliding window")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap for sliding window")
    parser.add_argument("--shard", type=str, default="00001",
                    help="Shard id, e.g. 00001 or 00002")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(f"Config: Chunk Size={args.chunk_size}, Overlap={args.overlap}")

    for idx in indices:
        ingest_one_instance(idx, args.chunk_size, args.overlap, args.shard)
 
if __name__ == "__main__":
    main()
