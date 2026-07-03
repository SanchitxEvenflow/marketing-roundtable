from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv

from neo4j.neo4j_retriever import Neo4jRetriever

load_dotenv()

class Neo4jClient:
    def __init__(self):
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=self.database
        )
        self.retriever = Neo4jRetriever(self)

    def get_relevant_personas(self, query: str, k: int = 5):
        """Semantic search for relevant personas"""
        cypher = """
        MATCH (p:Persona)
        CALL db.vector.annQuery('personaEmbeddings', $query_embedding, {k: $k})
        YIELD node, score
        RETURN node.name AS name, node.full_profile AS profile, score
        """
        # You'll need to generate embedding for query here
        return self.graph.query(cypher, {"query_embedding": ..., "k": k})  # we'll implement embedding later