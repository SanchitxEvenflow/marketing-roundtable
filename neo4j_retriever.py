from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

class Neo4jRetriever:
    def __init__(self, client):
        """client: Neo4jClient instance, driver shared not reopened"""
        self.driver = client.driver
        self.database = client.database

    def get_embedding(self, text: str):
        """Get Gemini embedding"""
        result = genai_client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
        )
        return result.embeddings[0].values

    def get_relevant_personas(self, topic: str, k: int = 5):
        """Get relevant personas based on semantic search"""
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