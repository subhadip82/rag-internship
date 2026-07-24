"""Utility helpers for logging, directory setup, and chunk serialization."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_logger(log_file: Optional[str | Path] = None) -> logging.Logger:
    """Create a logger that writes to the console and optionally to a file."""
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = ensure_directory(Path(log_file).parent) / Path(log_file).name
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_chunks_json(chunks: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Save chunk records to a JSON file."""
    output = Path(output_path)
    ensure_directory(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, indent=2, ensure_ascii=False)


def save_chunks_csv(chunks: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Save chunk records to a CSV file with metadata columns."""
    output = Path(output_path)
    ensure_directory(output.parent)

    fieldnames = [
        "chunk_id",
        "text",
        "source_file",
        "page_number",
        "chunk_number",
        "document_name",
    ]

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            row = {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "source_file": metadata.get("source_file", ""),
                "page_number": metadata.get("page_number", ""),
                "chunk_number": metadata.get("chunk_number", ""),
                "document_name": metadata.get("document_name", ""),
            }
            writer.writerow(row)
