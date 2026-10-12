"""Recursive character text splitter with overlap (no external dependencies)."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int

    def to_dict(self) -> Dict:
        return asdict(self)


def _split(text: str, size: int, seps: List[str]) -> List[str]:
    """Split text into pieces <= size, preferring the highest-level separator."""
    if len(text) <= size:
        return [text]
    if not seps:  # hard split as a last resort
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    pieces: List[str] = []
    for i, p in enumerate(parts):
        if not p:
            continue
        last = i == len(parts) - 1
        # keep sentence/word separators attached (except after the final part)
        piece = p if (sep in ("\n\n", "\n") or last) else p + sep
        if len(piece) > size:
            pieces.extend(_split(piece, size, rest))
        else:
            pieces.append(piece)
    return pieces


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Greedily merge small pieces up to chunk_size, carrying an overlap tail."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    pieces = _split(text.strip(), chunk_size, _SEPARATORS)
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        joiner = "" if current.endswith(" ") else "\n"
        candidate = f"{current}{joiner}{piece}" if current else piece
        candidate = candidate.strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the next chunk at a word boundary: if the tail begins
            # mid-word, drop the partial first word
            if tail and not current[-overlap - 1 : -overlap][:1].isspace() and len(current) > overlap:
                parts = re.split(r"\s", tail, maxsplit=1)
                tail = parts[1] if len(parts) > 1 else ""
            joiner = "" if tail.endswith(" ") else "\n"
            current = f"{tail}{joiner}{piece}".strip() if tail else piece.strip()
            if len(current) > chunk_size:  # overlap + piece too large
                current = piece.strip()
        else:
            current = piece.strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def chunk_pages(pages, source: str, start_id: int = 0, chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    """Chunk a list of (page, text) into Chunk objects with source/page metadata."""
    out: List[Chunk] = []
    cid = start_id
    for page, text in pages:
        for piece in split_text(text, chunk_size, overlap):
            out.append(Chunk(text=piece, source=source, page=page, chunk_id=cid))
            cid += 1
    return out
