"""Unified persona source: Neo4j vector search first, local markdown fallback."""
import config


def get_personas(topic: str, k: int = None) -> list[dict]:
    k = k or config.NUM_PERSONAS
    try:
        from .neo4j_retriever import Neo4jRetriever
        retriever = Neo4jRetriever()
        try:
            results = retriever.get_relevant_personas(topic, k=k)
        finally:
            retriever.close()
        if results:
            print(f"[retrieval] Neo4j vector search → {[p['name'] for p in results]}")
            return results
        raise RuntimeError("Neo4j returned no personas")
    except Exception as e:
        print(f"[retrieval] Neo4j unavailable ({type(e).__name__}: {e}); "
              f"falling back to local profiles")
        from .local_loader import get_relevant_personas
        results = get_relevant_personas(topic, k=k)
        print(f"[retrieval] local fallback → {[p['name'] for p in results]}")
        return results
