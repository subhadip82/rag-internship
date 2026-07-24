"""Retrieval helpers for Week 4."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.embedding import compute_cosine_similarity


def build_retrieval_results(
    query_embedding: np.ndarray,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Build ranked retrieval results using cosine similarity."""
    if not chunks:
        return []

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Chunk count and embedding count do not match")

    similarities = compute_cosine_similarity(query_embedding, embeddings)
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for index in ranked_indices:
        chunk = dict(chunks[index])
        chunk["similarity_score"] = float(similarities[index])
        results.append(chunk)

    return results


def print_retrieval_results(
    results: List[Dict[str, Any]],
    logger: Optional[Any] = None,
) -> None:
    """Print a beginner-friendly summary of retrieval results."""
    if logger:
        logger.info("Displaying %s retrieval result(s)", len(results))

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        print(f"{rank}. Score={result.get('similarity_score', 0):.4f} | Chunk={result.get('chunk_id')} | File={metadata.get('source_file')} | Page={metadata.get('page_number')}")
        if logger:
            logger.info("Rank %s -> %s", rank, result.get("chunk_id"))
