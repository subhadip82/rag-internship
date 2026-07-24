import numpy as np
from unittest import TestCase

from src.embedding import compute_cosine_similarity
from src.retriever import build_retrieval_results
from src.vector_store import LocalVectorStore


class Week4Tests(TestCase):
    def test_compute_cosine_similarity(self) -> None:
        query_embedding = np.array([1.0, 0.0], dtype=np.float32)
        chunk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

        similarities = compute_cosine_similarity(query_embedding, chunk_embeddings)

        self.assertEqual(len(similarities), 3)
        self.assertAlmostEqual(similarities[0], 1.0)
        self.assertAlmostEqual(similarities[1], 0.0)
        self.assertAlmostEqual(similarities[2], 0.70710677, places=5)

    def test_local_vector_store_search(self) -> None:
        chunks = [
            {"chunk_id": "chunk-1", "text": "Eco travel tips", "metadata": {"source_file": "file-a.pdf"}},
            {"chunk_id": "chunk-2", "text": "Public transport advice", "metadata": {"source_file": "file-b.pdf"}},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store = LocalVectorStore(chunks=chunks, embeddings=embeddings)

        results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk-1")

    def test_build_retrieval_results(self) -> None:
        chunks = [
            {"chunk_id": "chunk-1", "text": "Eco travel tips", "metadata": {"source_file": "file-a.pdf"}},
            {"chunk_id": "chunk-2", "text": "Public transport advice", "metadata": {"source_file": "file-b.pdf"}},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        results = build_retrieval_results(
            query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            chunks=chunks,
            embeddings=embeddings,
            top_k=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chunk-1")
        self.assertGreaterEqual(results[0]["similarity_score"], 0.99)
