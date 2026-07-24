"""PDF loading utilities for the document preparation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader


def load_pdf_documents(raw_documents_dir: str | Path, logger: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Load every PDF file in a directory and return a list of structured document records."""
    raw_dir = Path(raw_documents_dir)
    documents: List[Dict[str, Any]] = []

    if not raw_dir.exists():
        if logger:
            logger.warning("Raw documents directory does not exist: %s", raw_dir)
        return documents

    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        if logger:
            logger.info("No PDF files found in %s", raw_dir)
        return documents

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page_number": page_number, "text": text})

            full_text = "\n\n".join(page["text"] for page in pages if page["text"])
            documents.append(
                {
                    "source_file": pdf_path.name,
                    "document_name": pdf_path.stem,
                    "pages": pages,
                    "raw_text": full_text,
                }
            )
            if logger:
                logger.info("Loaded %s with %s page(s)", pdf_path.name, len(pages))
        except Exception as exc:  # pragma: no cover - defensive logging
            if logger:
                logger.exception("Failed to load %s: %s", pdf_path.name, exc)

    return documents
