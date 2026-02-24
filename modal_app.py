"""
modal_app.py — Deploy the Graph RAG FastAPI backend on Modal (GPU A100).

This serves the full FastAPI application on a Modal container with:
  • Qwen 7B LLM (float16) on A100 GPU
  • multilingual-e5-base embedding model
  • Neo4j connection (credentials via Modal secrets)
  • All /api/* endpoints (query, chunking, graph build, health)

Deploy:
    modal deploy modal_app.py

Dev (hot-reload):
    modal serve modal_app.py

Test locally:
    curl https://<your-modal-app>.modal.run/docs
"""

from __future__ import annotations
import modal

from app.config import load_environment

load_environment()

# ═══════════════════════════════════════════════════════════════════════
# Modal configuration
# ═══════════════════════════════════════════════════════════════════════

app = modal.App("graph-rag-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "poppler-utils",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        # FastAPI
        "fastapi>=0.115.0",
        "pydantic>=2.0",
        "python-dotenv>=1.0.1",
        # ML / LLM
        "torch>=2.4.0",
        "transformers>=4.46.0",
        "accelerate>=0.34.0",
        "sentencepiece",
        "sentence-transformers>=3.0.0",
        # OCR
        "pdf2image>=1.17.0",
        "Pillow>=10.0.0",
        "paddlepaddle>=3.0.0",
        "paddleocr>=3.0.0",
        # Agent
        "langgraph>=0.2.53",
        "langchain-core>=0.3.0",
        # Neo4j
        "neo4j>=5.0.0",
    )
    # Copy the entire app package into the container
    .add_local_dir("app", remote_path="/root/app")
)

# Persistent volume for model weights (survives between deploys)
models_vol = modal.Volume.from_name("graph-rag-models", create_if_missing=True)


# ═══════════════════════════════════════════════════════════════════════
# GPU-backed FastAPI server
# ═══════════════════════════════════════════════════════════════════════

@app.cls(
    image=image,
    gpu="A100",
    volumes={"/models": models_vol},
    secrets=[modal.Secret.from_name("neo4j-credentials")],
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=4)
class GraphRAGServer:
    """Modal class that loads models on startup and serves FastAPI."""

    @modal.enter()
    def startup(self):
        """Load LLM + embedding model once when the container starts."""
        import sys
        sys.path.insert(0, "/root")

        from app.services.rag_agent import get_agent

        agent = get_agent()
        agent.startup()
        print("✓ Graph RAG agent ready (LLM + embeds loaded on A100)")

    @modal.asgi_app()
    def serve(self):
        """Return the FastAPI ASGI app (Modal serves it automatically)."""
        import sys
        sys.path.insert(0, "/root")

        from app.main import create_app
        return create_app()


# ═══════════════════════════════════════════════════════════════════════
# CLI entrypoint  (modal run modal_app.py --query "...")
# ═══════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    query: str = "ما هو النظام الجبائي للشركات الأهلية في مادة الأداء على القيمة المضافة؟",
):
    """Quick test: run a single query against the deployed agent."""
    import urllib.request
    import json

    # When running locally, hit the deployed endpoint
    url = GraphRAGServer.serve.web_url + "/api/query"
    data = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    print("\n" + "═" * 70)
    print(f"  السؤال: {result['query']}")
    print("═" * 70)
    print(f"\n{result['response']}")

    if result.get("sources"):
        print("\n── المصادر " + "─" * 55)
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['source_type']} — {s['section_path']}")

    if result.get("needs_clarification"):
        print(f"\n⚠  للتوضيح: {result.get('clarification_question', '')}")
    print()
