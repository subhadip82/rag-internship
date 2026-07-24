"""Chunking logic that groups cleaned paragraphs into retrieval-friendly text blocks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def count_words(text: str) -> int:
    """Return the approximate number of words in a text block."""
    return len(text.split())


def split_long_paragraph(paragraph: str, target_words: int) -> List[str]:
    """Break a paragraph into smaller pieces if it is unusually long."""
    words = paragraph.split()
    if len(words) <= target_words:
        return [paragraph]

    pieces = []
    for start in range(0, len(words), target_words):
        piece = " ".join(words[start : start + target_words])
        pieces.append(piece)
    return pieces


def build_chunks(paragraphs: List[Dict[str, Any]], target_words: int = 500, min_words: int = 250) -> List[Dict[str, Any]]:
    """Group paragraphs into chunks while keeping paragraph boundaries where possible."""
    chunks: List[Dict[str, Any]] = []
    current_paragraphs: List[Dict[str, Any]] = []
    current_word_count = 0

    def flush_current() -> None:
        nonlocal current_paragraphs, current_word_count
        if not current_paragraphs:
            return

        chunk_text = " ".join(item["text"] for item in current_paragraphs)
        page_numbers = [item["page_number"] for item in current_paragraphs]
        chunks.append({"text": chunk_text, "page_numbers": page_numbers})
        current_paragraphs = []
        current_word_count = 0

    for paragraph in paragraphs:
        paragraph_text = paragraph.get("text", "")
        paragraph_words = count_words(paragraph_text)

        if paragraph_words > target_words:
            if current_paragraphs:
                flush_current()
            for piece in split_long_paragraph(paragraph_text, target_words):
                chunks.append({"text": piece, "page_numbers": [paragraph.get("page_number")]})
            continue

        if current_paragraphs and current_word_count + paragraph_words > target_words and current_word_count >= min_words:
            flush_current()

        current_paragraphs.append(paragraph)
        current_word_count += paragraph_words

    flush_current()
    return chunks


def create_document_chunks(document: Dict[str, Any], target_words: int = 500, min_words: int = 250, logger: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Create chunk records with metadata for a single document."""
    paragraph_records = document.get("paragraphs", [])
    if not paragraph_records:
        if logger:
            logger.info("No paragraphs available for %s", document.get("source_file"))
        return []

    chunk_blocks = build_chunks(paragraph_records, target_words=target_words, min_words=min_words)
    chunk_records = []

    for index, chunk_block in enumerate(chunk_blocks, start=1):
        chunk_id = f"{document.get('document_name', 'document')}_chunk_{index:03d}"
        metadata = {
            "chunk_id": chunk_id,
            "source_file": document.get("source_file"),
            "page_number": chunk_block["page_numbers"][0] if chunk_block.get("page_numbers") else None,
            "chunk_number": index,
            "document_name": document.get("document_name"),
        }
        chunk_records.append({"chunk_id": chunk_id, "text": chunk_block["text"], "metadata": metadata})

    if logger:
        logger.info("Created %s chunk(s) for %s", len(chunk_records), document.get("source_file"))

    return chunk_records
