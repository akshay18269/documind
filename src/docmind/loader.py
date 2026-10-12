"""Load raw text from PDF / TXT / MD files."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: str | Path) -> List[Tuple[int, str]]:
    """Return a list of (page_number, text). Text files are a single page 1."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of {sorted(SUPPORTED)}")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i, text))
        return pages

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(1, text)] if text else []
