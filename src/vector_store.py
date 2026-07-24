"""A lightweight local vector store for Week 4 retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.embedding import compute_cosine_similarity


class LocalVectorStore:
    """Store chunks, embeddings, and metadata in memory for simple retrieval."""

    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None, embeddings: Optional[np.ndarray] = None) -> None:
        self.chunks = chunks or []
        self.embeddings = embeddings if embeddings is not None else np.empty((0, 0), dtype=np.float32)

    def add(self, chunk: Dict[str, Any], embedding: np.ndarray) -> None:
        """Append a chunk and its embedding to the store."""
        self.chunks.append(chunk)
        if self.embeddings.size == 0:
            self.embeddings = np.array([embedding], dtype=np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

    def save(self, output_path: str | Path) -> None:
        """Persist the vector store payload to disk as JSON."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist(),
        }
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k matching chunks for the given query embedding."""
        if not self.chunks:
            return []

        if self.embeddings.shape[0] != len(self.chunks):
            raise ValueError("Chunk count and embedding count do not match")

        similarities = compute_cosine_similarity(query_embedding, self.embeddings)
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for index in ranked_indices:
            chunk = dict(self.chunks[index])
            chunk["similarity_score"] = float(similarities[index])
            results.append(chunk)

        return results
