"""Text cleaning helpers for normalizing extracted PDF content."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters and convert odd spacing into standard characters."""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving sentence boundaries."""
    return re.sub(r"\s+", " ", text).strip()


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs using blank-line boundaries."""
    normalized_text = normalize_unicode(text)
    paragraphs = []
    for raw_paragraph in re.split(r"\n\s*\n", normalized_text):
        cleaned = normalize_whitespace(raw_paragraph)
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def clean_document(document: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:
    """Clean and structure the extracted text from a single document."""
    cleaned_pages = []
    paragraph_records = []

    for page in document.get("pages", []):
        page_number = page.get("page_number")
        page_text = normalize_unicode(page.get("text", ""))
        cleaned_paragraphs = split_into_paragraphs(page_text)

        cleaned_pages.append({"page_number": page_number, "text": "\n\n".join(cleaned_paragraphs)})
        for paragraph in cleaned_paragraphs:
            paragraph_records.append({"page_number": page_number, "text": paragraph})

    if logger:
        logger.info("Cleaned %s into %s paragraph(s)", document.get("source_file"), len(paragraph_records))

    return {
        "source_file": document.get("source_file"),
        "document_name": document.get("document_name"),
        "pages": cleaned_pages,
        "paragraphs": paragraph_records,
    }
