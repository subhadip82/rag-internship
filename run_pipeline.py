"""Run the Week 3 document preparation pipeline."""

from __future__ import annotations

from pathlib import Path

from src.cleaner import clean_document
from src.chunker import create_document_chunks
from src.loader import load_pdf_documents
from src.utils import ensure_directory, save_chunks_csv, save_chunks_json, setup_logger


def main() -> None:
    """Load PDFs, clean text, chunk documents, and save outputs."""
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    raw_documents_dir = data_dir / "raw_documents"
    chunks_dir = data_dir / "chunks"
    ensure_directory(chunks_dir)

    logger = setup_logger(chunks_dir / "pipeline.log")

    documents = load_pdf_documents(raw_documents_dir, logger=logger)
    cleaned_documents = []
    all_chunks = []

    for document in documents:
        cleaned_document = clean_document(document, logger=logger)
        cleaned_documents.append(cleaned_document)
        document_chunks = create_document_chunks(cleaned_document, logger=logger)
        all_chunks.extend(document_chunks)

    save_chunks_json(all_chunks, chunks_dir / "chunks.json")
    save_chunks_csv(all_chunks, chunks_dir / "chunks.csv")

    logger.info("Saved %s chunk(s) to %s", len(all_chunks), chunks_dir)


if __name__ == "__main__":
    main()
