"""
Pydantic models for API request / response schemas.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ─── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question (Arabic or English)")


class SourceInfo(BaseModel):
    index: int
    source_type: str = ""
    section_path: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    response: str
    sources: list[SourceInfo] = []
    intent: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""


# ─── Chunking ─────────────────────────────────────────────────────────────────

class ChunkRequest(BaseModel):
    input_path: str = Field(..., description="Path to the OCR HTML file")
    output_path: str = Field("", description="Output JSON path (auto-generated if empty)")
    doc_type: str = Field("auto", description="'law', 'notes', or 'auto' to detect")
    max_tokens: int = Field(1500, ge=100, le=5000)


class ChunkInfo(BaseModel):
    chunk_id: str
    chunk_type: str
    section_path: str = ""
    section_number: Optional[str] = None
    token_count: int = 0
    text_preview: str = ""


class ChunkResponse(BaseModel):
    total_chunks: int
    total_tokens: int
    output_path: str
    chunks: list[ChunkInfo] = []


# ─── OCR ──────────────────────────────────────────────────────────────────────

class OCRProcessRequest(BaseModel):
    input_pdf_path: str = Field(..., description="Path to input PDF file")
    dpi: int = Field(170, ge=72, le=600, description="Rasterization DPI for PDF pages")
    output_html_path: str = Field("", description="Optional OCR HTML output path (auto if empty)")
    output_json_path: str = Field("", description="Optional OCR JSON output path (auto if empty)")


class OCRProcessResponse(BaseModel):
    source_file: str
    page_count: int
    total_text_blocks: int = 0
    total_tables: int = 0
    output_html_path: str = ""
    output_json_path: str = ""


# ─── Graph ────────────────────────────────────────────────────────────────────

class GraphBuildRequest(BaseModel):
    law_chunks_path: str = "chunks_graphrag.json"
    note_chunks_path: str = "chunks_notes.json"
    clear_graph: bool = False
    skip_embeddings: bool = False


class GraphBuildResponse(BaseModel):
    status: str
    law_chunks_count: int = 0
    note_chunks_count: int = 0
    message: str = ""


class GraphStatsResponse(BaseModel):
    chunks: int = 0
    entities: int = 0
    relationships: int = 0
    relationship_types: dict[str, int] = {}


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    neo4j_connected: bool = False
    models_loaded: bool = False
