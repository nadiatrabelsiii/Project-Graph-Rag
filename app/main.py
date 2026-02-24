"""
Graph RAG — FastAPI application factory.

This module creates and configures the FastAPI app. It can be used both:
  • On Modal (via modal_app.py — GPU A100, models loaded in container)
  • Locally  (uvicorn app.main:create_app --factory --reload)
"""

from __future__ import annotations
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import load_environment

load_environment()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and return the FastAPI application (no model loading here)."""
    from app.routers import query, documents, graph, ocr

    application = FastAPI(
        title="Graph RAG API",
        description=(
            "Graph RAG API for Tunisian legal documents. "
            "Includes PaddleOCR-VL document processing, "
            "Combines Neo4j knowledge graph traversal with LLM-powered "
            "retrieval-augmented generation to answer legal questions in Arabic. "
            "Deployed on Modal (GPU A100)."
        ),
        version="1.0.0",
    )

    # CORS — allow all origins (tighten in production)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(query.router)
    application.include_router(documents.router)
    application.include_router(ocr.router)
    application.include_router(graph.router)

    @application.get("/")
    async def root():
        return {
            "app": "Graph RAG API",
            "docs": "/docs",
            "ocr": "/api/ocr/process",
            "health": "/api/health",
        }

    return application
