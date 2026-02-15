"""
Retrieval module: wires Faiss vector search to SQLite summary lookup.

Usage:
    results = retrieve("What happened with the bandit problem?", k=3)
    for summary, score in results:
        print(f"[{score:.3f}] {summary.summary_text}")
"""

from config import CONTEXT_WINDOW
from dataclasses import dataclass
from schemas.core import SummaryMetadata
from memory.FaissVectorStore import FaissVectorStore
from memory.SQLiteSummaryStore import SQLiteSummaryStore
from utils.embeddings import generate_embedding


@dataclass
class RetrievalResult:
    """A single retrieval hit with its metadata and similarity score."""
    summary: SummaryMetadata
    score: float


def retrieve(query: str, k: int = CONTEXT_WINDOW) -> list[RetrievalResult]:
    """
    End-to-end retrieval: natural language query → ranked summary results.

    Pipeline:
        1. Embed the query text into a 768-dim vector.
        2. Search Faiss for the top-k most similar summary embeddings.
        3. Map Faiss results to SQLite IDs and fetch the full summary records.

    Args:
        query: Natural language search string.
        k: Number of results to return.

    Returns:
        List of RetrievalResult ordered by descending similarity score.
        Empty list if no summaries exist yet.
    """
    query_embedding = generate_embedding(query)

    with FaissVectorStore() as vector_store:
        faiss_results = vector_store.search(query_embedding, k=k)

    if not faiss_results:
        return []

    sqlite_ids = [sqlite_id for sqlite_id, _ in faiss_results]
    score_map = {sqlite_id: score for sqlite_id, score in faiss_results}

    with SQLiteSummaryStore() as summary_store:
        summaries = summary_store.get_summaries_by_ids(sqlite_ids)

    # Pair each summary with its Faiss similarity score, preserving rank order
    results = []
    for summary in summaries:
        # Match by position — get_summaries_by_ids preserves input order
        sid = sqlite_ids[summaries.index(summary)]
        results.append(RetrievalResult(summary=summary, score=score_map[sid]))

    return results
