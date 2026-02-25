"""
/api/query — RAG query endpoint.
"""

from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse
from app.services.rag_agent import get_agent

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Ask a legal question — the agentic Graph RAG pipeline will:
    1. Analyze intent & extract entities
    2. Retrieve from Neo4j (keyword, vector, graph traversal)
    3. Evaluate relevance via LLM
    4. Generate a grounded French legal answer
    """
    agent = get_agent()
    if not agent.is_ready:
        raise HTTPException(503, "Les modeles sont en cours de chargement. Veuillez reessayer dans un instant.")

    try:
        result = agent.query(req.query)
    except Exception as exc:
        log.exception("query failed")
        raise HTTPException(500, f"Erreur lors du traitement de la requete: {exc}")

    return QueryResponse(
        query=result.get("query", req.query),
        response=result.get("response", ""),
        sources=result.get("sources", []),
        intent=result.get("intent", ""),
        needs_clarification=result.get("needs_clarification", False),
        clarification_question=result.get("clarification_question", ""),
    )
