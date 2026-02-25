"""
/api/graph — graph building, stats, and health endpoints.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.config import load_environment
from app.models import (
    GraphBuildRequest,
    GraphBuildResponse,
    GraphStatsResponse,
    HealthResponse,
)
from app.services.neo4j_service import cypher, check_connection
from app.services.rag_agent import get_agent

load_environment()

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Graph"])

def _build_graph_task(
    law_chunks: list[dict],
    note_chunks: list[dict],
    clear: bool,
    skip_embeddings: bool,
):
    """Heavy background task — builds the full Neo4j knowledge graph."""
    from app.services.graph_builder import GraphBuilder, generate_embeddings

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    database = os.environ.get("NEO4J_DATABASE") or None
    embed_model = os.environ.get("EMBED_MODEL_ID", "intfloat/multilingual-e5-base")

    gb = GraphBuilder(uri, user, password, database=database)
    try:
        if clear:
            gb.clear()

        gb.create_schema()
        gb.insert_chunks(law_chunks, source="law")
        gb.insert_chunks(note_chunks, source="notes")
        gb.insert_entities(law_chunks, note_chunks)
        gb.create_mentions(law_chunks, note_chunks)
        gb.create_next(law_chunks)
        gb.create_next(note_chunks)
        gb.create_cross_references(law_chunks, note_chunks)
        gb.create_explains(law_chunks, note_chunks)
        gb.create_relates_to(law_chunks, note_chunks)
        gb.create_part_of(law_chunks, note_chunks)

        if not skip_embeddings:
            embeddings: Dict[str, list] = generate_embeddings(
                law_chunks + note_chunks, embed_model,
            )
            gb.store_embeddings(embeddings)
            dims = len(next(iter(embeddings.values())))
            gb.create_vector_index(dimensions=dims)
            gb.create_similar_to(threshold=0.78, max_neighbors=5)

        gb.stats()
        log.info("Graph build complete")
    except Exception:
        log.exception("Graph build failed")
    finally:
        gb.close()


@router.post("/graph/build", response_model=GraphBuildResponse)
async def build_graph(req: GraphBuildRequest, bg: BackgroundTasks):
    """
    Build/rebuild the Neo4j knowledge graph from chunk JSON files.

    Runs as a background task — returns immediately.
    """
    law_path = Path(req.law_chunks_path)
    note_path = Path(req.note_chunks_path)

    law_chunks = json.loads(law_path.read_text("utf-8")) if law_path.exists() else []
    note_chunks = json.loads(note_path.read_text("utf-8")) if note_path.exists() else []

    if not law_chunks and not note_chunks:
        raise HTTPException(400, "No chunks found. Run chunking first.")

    bg.add_task(
        _build_graph_task,
        law_chunks, note_chunks, req.clear_graph, req.skip_embeddings,
    )

    return GraphBuildResponse(
        status="building",
        law_chunks_count=len(law_chunks),
        note_chunks_count=len(note_chunks),
        message="Graph build started in background. Check /api/graph/stats for progress.",
    )


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats():
    """Return current graph statistics (node/edge counts)."""
    if not check_connection():
        raise HTTPException(503, "Neo4j is not reachable")

    try:
        rows = cypher(
            "MATCH (c:Chunk) WITH count(c) AS chunks "
            "MATCH (e:Entity) WITH chunks, count(e) AS entities "
            "OPTIONAL MATCH ()-[r]->() "
            "RETURN chunks, entities, count(r) AS rels"
        )
        stats = rows[0] if rows else {"chunks": 0, "entities": 0, "rels": 0}

        rel_rows = cypher(
            "MATCH ()-[r]->() "
            "RETURN type(r) AS rel_type, count(r) AS cnt "
            "ORDER BY cnt DESC"
        )
        rel_types = {r["rel_type"]: r["cnt"] for r in rel_rows}

        return GraphStatsResponse(
            chunks=stats.get("chunks", 0),
            entities=stats.get("entities", 0),
            relationships=stats.get("rels", 0),
            relationship_types=rel_types,
        )
    except Exception as exc:
        raise HTTPException(500, f"Stats query error: {exc}")

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verifies Neo4j connectivity and model readiness."""
    neo4j_ok = check_connection()
    agent = get_agent()

    return HealthResponse(
        status="ok" if neo4j_ok else "degraded",
        neo4j_connected=neo4j_ok,
        models_loaded=agent.is_ready,
    )
