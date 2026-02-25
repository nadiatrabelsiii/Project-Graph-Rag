"""Modal deployment entrypoint for the Graph RAG FastAPI backend."""

from __future__ import annotations
import modal

from app.config import load_environment

load_environment()

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
        "fastapi>=0.115.0",
        "pydantic>=2.0",
        "python-dotenv>=1.0.1",
        "torch>=2.4.0",
        "transformers>=4.46.0",
        "accelerate>=0.34.0",
        "sentencepiece",
        "sentence-transformers>=3.0.0",
        "pdf2image>=1.17.0",
        "Pillow>=10.0.0",
        "paddlepaddle>=3.0.0",
        "paddleocr>=3.0.0",
        "langgraph>=0.2.53",
        "langchain-core>=0.3.0",
        "neo4j>=5.0.0",
    )
    .add_local_dir("app", remote_path="/root/app")
    # Make chunk files available for /api/graph/build in production.
    .add_local_file("chunks_graphrag.json", remote_path="/root/chunks_graphrag.json")
    .add_local_file("chunks_notes.json", remote_path="/root/chunks_notes.json")
)

models_vol = modal.Volume.from_name("graph-rag-models", create_if_missing=True)

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
    """ASGI server with model startup hooks."""

    @modal.enter()
    def startup(self):
        """Load models once per container."""
        import sys
        sys.path.insert(0, "/root")

        from app.services.rag_agent import get_agent

        agent = get_agent()
        agent.startup()
        print("✓ Graph RAG agent ready (LLM + embeds loaded on A100)")

    @modal.asgi_app()
    def serve(self):
        """Return the FastAPI ASGI app."""
        import sys
        sys.path.insert(0, "/root")

        from app.main import create_app
        return create_app()

@app.local_entrypoint()
def main(
    query: str = "ما هو النظام الجبائي للشركات الأهلية في مادة الأداء على القيمة المضافة؟",
    endpoint: str = "",
):
    """Run one query against the deployed API endpoint."""
    import urllib.request
    import json

    if endpoint:
        base_url = endpoint.rstrip("/")
    else:
        serve_fn = modal.Function.from_name("graph-rag-api", "GraphRAGServer.serve")
        base_url = (serve_fn.web_url or "").rstrip("/")
        if not base_url:
            raise RuntimeError(
                "Unable to resolve deployed web URL. "
                "Provide one explicitly with --endpoint."
            )

    url = base_url + "/api/query"
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
