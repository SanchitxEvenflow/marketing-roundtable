"""Semantic persona retrieval over Neo4j vector index.

Embeddings were built with Gemini (see vectorembedding.py), so queries must
use the same embedding model — the debate LLM (NVIDIA) is independent of this.
"""
from google import genai
from neo4j import GraphDatabase, NotificationMinimumSeverity

import config

_genai_client = genai.Client(api_key=config.GOOGLE_API_KEY)

EMBEDDING_MODEL = config.EMBEDDING_MODEL


class Neo4jRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
            notifications_min_severity=NotificationMinimumSeverity.OFF,
        )
        self.database = config.NEO4J_DATABASE

    def get_embedding(self, text: str):
        result = _genai_client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text
        )
        return result.embeddings[0].values

    def get_relevant_personas(self, topic: str, k: int = 5):
        query_embedding = self.get_embedding(topic)
        cypher = """
        CALL db.index.vector.queryNodes('personaEmbeddings', $k, $query_embedding)
        YIELD node, score
        RETURN node.name AS name,
               node.full_profile AS profile,
               score
        ORDER BY score DESC
        """
        with self.driver.session(database=self.database) as session:
            results = session.run(cypher, query_embedding=query_embedding, k=k)
            return [dict(record) for record in results]

    def close(self):
        self.driver.close()
