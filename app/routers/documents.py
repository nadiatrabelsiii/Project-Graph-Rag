"""
/api/documents — chunking & document processing endpoints.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models import ChunkRequest, ChunkResponse, ChunkInfo

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Documents"])

CHUNKS_DIR = os.environ.get("CHUNKS_DIR", "./chunks")


def _detect_doc_type(path: str) -> str:
    """Guess document type from filename / content heuristics."""
    p = path.lower()
    if "note" in p or "مذكرة" in p:
        return "notes"
    return "law"


@router.post("/documents/chunk", response_model=ChunkResponse)
async def chunk_document(req: ChunkRequest):
    """
    Chunk an OCR'd HTML file into Graph RAG–ready JSON chunks.

    - doc_type='law'   → uses chunk_graphrag.py logic
    - doc_type='notes' → uses chunk_notes.py logic
    - doc_type='auto'  → auto-detect from filename
    """
    input_path = Path(req.input_path)
    if not input_path.exists():
        raise HTTPException(404, f"File not found: {req.input_path}")

    doc_type = req.doc_type if req.doc_type != "auto" else _detect_doc_type(req.input_path)

    # Determine output path
    if req.output_path:
        output_path = req.output_path
    else:
        Path(CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
        output_path = f"{CHUNKS_DIR}/chunks_{input_path.stem}.json"

    try:
        if doc_type == "notes":
            from app.services.chunk_notes import run as run_notes_chunker
            run_notes_chunker(
                input_path=str(input_path),
                output_path=output_path,
                max_tokens=req.max_tokens,
            )
        else:
            from app.services.chunk_law import run as run_law_chunker
            run_law_chunker(
                input_path=str(input_path),
                output_path=output_path,
                max_tokens=req.max_tokens,
            )
    except Exception as exc:
        log.exception("Chunking failed")
        raise HTTPException(500, f"Chunking error: {exc}")

    # Load result for response
    chunks_data = json.loads(Path(output_path).read_text("utf-8"))
    total_tokens = sum(c.get("token_count", 0) for c in chunks_data)

    chunk_infos = [
        ChunkInfo(
            chunk_id=c["chunk_id"],
            chunk_type=c.get("chunk_type", ""),
            section_path=c.get("section_path", ""),
            section_number=c.get("section_number") or c.get("article_number"),
            token_count=c.get("token_count", 0),
            text_preview=c.get("text", "")[:120],
        )
        for c in chunks_data
    ]

    return ChunkResponse(
        total_chunks=len(chunks_data),
        total_tokens=total_tokens,
        output_path=output_path,
        chunks=chunk_infos,
    )


@router.get("/documents/chunks/{filename}")
async def get_chunks(filename: str):
    """Return the contents of a previously generated chunks JSON file."""
    for dir_path in [".", CHUNKS_DIR]:
        p = Path(dir_path) / filename
        if p.exists():
            return json.loads(p.read_text("utf-8"))

    raise HTTPException(404, f"Chunks file not found: {filename}")
