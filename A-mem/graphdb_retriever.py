from typing import List, Dict, Any
from neo4j import GraphDatabase
import json
import tiktoken


class Neo4jRetriever:
    """
    Fully AMEM-compatible retriever but using Neo4j as backend.
    - NO CHUNKING
    - Document = one memory
    - Embedding uses enhanced_document (full AMEM logic)
    - Metadata processing identical to AMEM
    - Returns same structure as Chroma retriever
    - 新增：实体过滤+向量检索双模式，保留原有返回格式
    """

    def __init__(self,
                 collection_name: str = "memories",
                 model_name: str = None,
                 uri: str = None,
                 user: str = None,
                 password: str = None):

        # Load config (embedding provider etc.)
        from src.config import get_config
        cfg = get_config()

        # Neo4j config
        neo4j_cfg = cfg._app_config.get("neo4j", {})
        self.uri = uri or neo4j_cfg.get("uri")
        self.user = user or neo4j_cfg.get("user")
        self.password = password or neo4j_cfg.get("password")

        print(f"[Neo4jRetriever] connect: {self.uri} user={self.user}")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        # Embedding config
        self.embedding_cfg = cfg.embedding
        self.model_name = model_name or self.embedding_cfg.get("model")
        provider = self.embedding_cfg.get("provider")

        # Build embedding function
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.embedding_cfg.get("api_key"),
                base_url=self.embedding_cfg.get("base_url")
            )

            def embed_fn(texts):
                res = self.client.embeddings.create(
                    input=texts,
                    model=self.model_name,
                    dimensions=self.embedding_cfg.get("dim", 1536)
                )
                return [d.embedding for d in res.data]

            self.embedding_function = embed_fn

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        # Tokenizer (same as AMEM)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # 创建spacy实体提取模型（新增：用于查询的实体提取）
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")  # 中文/英文和Ingest保持一致
        except:
            raise Exception("请先安装spacy和实体模型：pip install spacy && python -m spacy download en_core_web_sm")

        # Create vector index
        self._create_vector_index()

    # -----------------------------------------------------------
    # Vector Index
    # -----------------------------------------------------------
    def _create_vector_index(self):
        dim = self.embedding_cfg.get("dim", 1536)
        with self.driver.session() as session:
            session.run("""
                CREATE VECTOR INDEX memory_vector_index IF NOT EXISTS
                FOR (m:Memory)
                ON (m.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """, dim=dim)

    # -----------------------------------------------------------
    # AMEM Metadata Processing (unchanged)
    # -----------------------------------------------------------
    def _process_metadata_for_storage(self, metadata: Dict):
        """Convert lists/dicts to JSON strings (AMEM behavior)."""
        result = {}
        for k, v in metadata.items():
            if isinstance(v, (list, dict)):
                result[k] = json.dumps(v, ensure_ascii=False)
            else:
                result[k] = v
        return result

    def _restore_metadata(self, metadata: Dict):
        """Convert JSON strings back into python types (AMEM behavior)."""
        restored = {}
        for k, v in metadata.items():
            if isinstance(v, str) and ((v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}"))):
                try:
                    restored[k] = json.loads(v)
                except:
                    restored[k] = v
            else:
                restored[k] = v
        return restored

    # -----------------------------------------------------------
    # 新增：提取查询的实体（用于检索过滤）
    # -----------------------------------------------------------
    def _extract_query_entities(self, query: str) -> list:
        """提取查询中的实体，返回实体名列表"""
        doc = self.nlp(query)
        entities = []
        seen_ents = set()
        for ent in doc.ents:
            ent_text = ent.text.strip()
            if len(ent_text) < 2 or ent_text in seen_ents:
                continue
            seen_ents.add(ent_text)
            entities.append(ent_text)
        return entities

    # -----------------------------------------------------------
    # AMEM Enhanced Document Building (unchanged)
    # -----------------------------------------------------------
    def _build_enhanced_document(self, document: str, metadata: Dict):
        """Reproduce AMEM's enhanced_document logic."""
        enhanced = document
        ctx = metadata.get("context", "General")
        if ctx != "General":
            enhanced += f"\nContext: {ctx}"
        keywords = metadata.get("keywords")
        if keywords:
            if isinstance(keywords, list):
                enhanced += f"\nKeywords: {', '.join(keywords)}"
            else:
                enhanced += f"\nKeywords: {keywords}"
        tags = metadata.get("tags")
        if tags:
            if isinstance(tags, list):
                enhanced += f"\nTags: {', '.join(tags)}"
            else:
                enhanced += f"\nTags: {tags}"
        return enhanced

    # -----------------------------------------------------------
    # Add Memory (AMEM style) (unchanged)
    # -----------------------------------------------------------
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        metadata_processed = self._process_metadata_for_storage(metadata)
        enhanced_document = self._build_enhanced_document(document, metadata)
        embedding = self.embedding_function([enhanced_document])[0]
        with self.driver.session() as session:
            session.run("""
                MERGE (m:Memory {id: $id})
                SET m.text = $text,
                    m.enhanced = $enhanced,
                    m.metadata = $metadata,
                    m.embedding = $embedding
            """, id=doc_id,
                 text=document,
                 enhanced=enhanced_document,
                 metadata=json.dumps(metadata_processed, ensure_ascii=False),
                 embedding=embedding)

    # -----------------------------------------------------------
    # Delete (unchanged)
    # -----------------------------------------------------------
    def delete_document(self, doc_id: str):
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Memory {id: $id})
                DETACH DELETE m
            """, id=doc_id)

    # -----------------------------------------------------------
    # Search (AMEM style) - 核心改造：保留格式+实体过滤+向量检索
    # -----------------------------------------------------------
    def search(self, query: str, k: int = 5):
        query_emb = self.embedding_function([query])[0]

        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)
                RETURN m.id AS id,
                       m.text AS text,
                       m.metadata AS metadata,
                       vector.similarity.cosine(m.embedding, $emb) AS sim,
                       0 AS entity_match_count
                ORDER BY sim DESC
                LIMIT $k
            """, emb=query_emb, k=k)

            ids = []
            docs = []
            metadatas = []
            distances = []

            for r in result:
                rid = r["id"]
                ids.append(rid)
                docs.append(r["text"])

                metadata_raw = json.loads(r["metadata"]) if r["metadata"] else {}
                metadata_final = self._restore_metadata(metadata_raw)
                metadata_final["similarity"] = r["sim"]
                metadata_final["entity_match_count"] = 0
                metadata_final["content"] = r["text"]
                metadatas.append(metadata_final)

                distances.append(r["sim"])

        return {
            "documents": [docs],
            "metadatas": [metadatas],
            "ids": [ids],
            "distances": [distances]
        }



    # -----------------------------------------------------------
    # Reset (unchanged)
    # -----------------------------------------------------------
    def reset(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")