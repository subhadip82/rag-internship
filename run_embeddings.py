"""Run the Week 4 embedding and retrieval pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Dict

from src.embedding import embed_query, embed_texts, load_embedding_model
from src.retriever import build_retrieval_results, print_retrieval_results
from src.utils import ensure_directory, setup_logger
from src.vector_store import LocalVectorStore


def load_chunks(chunks_path: str | Path) -> List[Dict[str, Any]]:
    """Load chunk records from the JSON file produced in Week 3."""
    with Path(chunks_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Create embeddings, store them locally, and print retrieval results."""
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    chunks_path = data_dir / "chunks" / "chunks.json"
    output_dir = data_dir / "embeddings"
    ensure_directory(output_dir)

    logger = setup_logger(output_dir / "embedding_pipeline.log")

    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.warning("No chunks found in %s", chunks_path)
        return

    logger.info("Loaded %s chunk(s) from %s", len(chunks), chunks_path)

    model = load_embedding_model(logger=logger)
    chunk_texts = [chunk.get("text", "") for chunk in chunks]
    chunk_embeddings = embed_texts(chunk_texts, model=model, logger=logger)

    embedding_dimension = chunk_embeddings.shape[1] if chunk_embeddings.ndim > 1 else 0
    print(f"Total Chunks: {len(chunks)}")
    print(f"Embedding Dimension: {embedding_dimension}")
    logger.info("Embedding dimension: %s", embedding_dimension)

    query = "sustainable travel in South Korea with eco-friendly transport"
    query_embedding = embed_query(query, model=model, logger=logger)

    store = LocalVectorStore(chunks=chunks, embeddings=chunk_embeddings)
    store.save(output_dir / "vector_store.json")
    logger.info("Stored %s chunk(s) in the local vector store", len(chunks))

    top_1_results = build_retrieval_results(query_embedding, chunks, chunk_embeddings, top_k=1)
    top_3_results = build_retrieval_results(query_embedding, chunks, chunk_embeddings, top_k=3)
    top_5_results = build_retrieval_results(query_embedding, chunks, chunk_embeddings, top_k=5)

    print("Top 1 Retrieved Chunk:")
    print_retrieval_results(top_1_results, logger=logger)

    print("Top 3 Retrieved Chunks:")
    print_retrieval_results(top_3_results, logger=logger)

    print("Top 5 Retrieved Chunks:")
    print_retrieval_results(top_5_results, logger=logger)

    logger.info("Week 4 embedding retrieval pipeline completed")


if __name__ == "__main__":
    main()
