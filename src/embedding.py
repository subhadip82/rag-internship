"""Embedding helpers for Week 4."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger("rag_pipeline")


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", logger: Optional[Any] = None) -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    if logger:
        logger.info("Loading embedding model: %s", model_name)
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger:
            logger.exception("Failed to load embedding model %s: %s", model_name, exc)
        raise

    if logger:
        logger.info("Embedding model loaded successfully")
    return model


def embed_texts(texts: List[str], model: SentenceTransformer, logger: Optional[Any] = None) -> np.ndarray:
    """Generate embeddings for a list of text chunks."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if logger:
        logger.info("Generating embeddings for %s text(s)", len(texts))

    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return np.asarray(embeddings, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger:
            logger.exception("Failed to generate embeddings: %s", exc)
        raise


def embed_query(text: str, model: SentenceTransformer, logger: Optional[Any] = None) -> np.ndarray:
    """Generate an embedding for a single user query."""
    if logger:
        logger.info("Generating embedding for query")
    embedding = embed_texts([text], model=model, logger=logger)
    return embedding[0]


def compute_cosine_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query embedding and many chunk embeddings."""
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

    if query_norm == 0 or np.any(chunk_norms == 0):
        return np.zeros(len(chunk_embeddings), dtype=np.float32)

    similarity = (chunk_embeddings @ query_embedding) / (chunk_norms * query_norm)
    return np.asarray(similarity, dtype=np.float32)
