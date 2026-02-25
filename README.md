# Agentic Graph RAG Backend for Tunisian Legal and Tax Documents

This repository contains the backend of an agentic Graph RAG system for Tunisian legal and tax content (laws and explanatory notes).

The system ingests PDF documents, extracts text with OCR, builds structured chunks, stores links in Neo4j, and answers questions with grounded sources.

## What this project does

- OCR on legal PDFs with PaddleOCR-VL
- Law and notes chunking (Arabic and French)
- Neo4j knowledge graph construction
- Multi-step retrieval (keyword, graph traversal, optional vector)
- Answer generation with grounding checks and citation-first behavior
- Web API + simple UI (`/ui`)

## Graph RAG + agentic behavior

This is Graph RAG, not plain RAG.

- Graph relations are used for retrieval expansion (`EXPLAINS`, `RELATES_TO`, `CROSS_REFERENCES`, `NEXT_CHUNK`, `CITES_EXTERNAL_ARTICLE`, etc.)
- Query processing is multi-step (intent -> retrieve -> validate -> answer/fallback)
- If evidence is weak or missing, the system asks for clarification instead of hallucinating

## Tech stack

- FastAPI
- Neo4j
- Qwen2.5-7B-Instruct (inference)
- `intfloat/multilingual-e5-base` (embeddings)
- PaddleOCR-VL 1.5
- Modal (A100 deployment target)

## Repository structure

```text
app/
  main.py
  config.py
  models.py
  routers/
    query.py
    graph.py
    documents.py
    ocr.py
    ui.py
  services/
    rag_agent.py
    graph_builder.py
    graph_quality_report.py
    neo4j_service.py
    ocr_service.py
    chunk_law.py
    chunk_notes.py
Data/
  Law/
  Notes/
OCR_Law/
OCR_Notes/
modal_app.py
requirements.txt
.env
```

## Local setup

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Set at least:

- `NEO4J_URI`
- `NEO4J_USER` (or `NEO4J_USERNAME`)
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (for Aura)

### 3. Run API

```bash
uvicorn app.main:create_app --factory --reload
```

Open:

- Swagger: `http://127.0.0.1:8000/docs`
- UI: `http://127.0.0.1:8000/ui`

## End-to-end runbook

### A) OCR one PDF via API

```bash
curl -X POST http://127.0.0.1:8000/api/ocr/process \
  -H "Content-Type: application/json" \
  -d '{
    "input_pdf_path":"Data/Law/2026/Loi2025_17Arabe.pdf",
    "dpi":170,
    "output_html_path":"OCR_Law/2026/ocr_output2026.html",
    "output_json_path":"OCR_Law/2026/ocr_output2026.json"
  }'
```

### B) Chunk OCR outputs

```bash
python3 -m app.services.chunk_law --input-dir OCR_Law --output chunks_graphrag.json
python3 -m app.services.chunk_notes --input-dir OCR_Notes --output chunks_notes.json
```

### C) Build graph in Neo4j

```bash
python3 -m app.services.graph_builder \
  --law-chunks chunks_graphrag.json \
  --note-chunks chunks_notes.json \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "${NEO4J_USER:-$NEO4J_USERNAME}" \
  --neo4j-password "$NEO4J_PASSWORD" \
  --neo4j-database "$NEO4J_DATABASE" \
  --skip-embeddings --clear
```

### D) Query API

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Que prevoit l article 36 de la loi de finances 2025 ?"}'
```

## Modal deployment

### 1) Create/update secret

```bash
./.venv/bin/modal secret create neo4j-credentials --force \
  NEO4J_URI="neo4j+s://<your-aura-id>.databases.neo4j.io" \
  NEO4J_USER="<neo4j-user>" \
  NEO4J_PASSWORD="<neo4j-password>" \
  NEO4J_DATABASE="<neo4j-database>"
```

### 2) Deploy

```bash
./.venv/bin/modal deploy modal_app.py
```

### 3) Health and graph status

```bash
curl -sS https://<your-modal-url>/api/health
curl -sS https://<your-modal-url>/api/graph/stats
```

### 4) Build graph on deployed app

`modal_app.py` mounts `chunks_graphrag.json` and `chunks_notes.json` into the container.

```bash
curl -X POST https://<your-modal-url>/api/graph/build \
  -H "Content-Type: application/json" \
  -d '{
    "law_chunks_path":"/root/chunks_graphrag.json",
    "note_chunks_path":"/root/chunks_notes.json",
    "clear_graph":true,
    "skip_embeddings":true
  }'
```

## Quality checks

Run graph-level diagnostics:

```bash
PYTHONPATH=. python3 app/services/graph_quality_report.py \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "${NEO4J_USER:-$NEO4J_USERNAME}" \
  --neo4j-password "$NEO4J_PASSWORD" \
  --neo4j-database "$NEO4J_DATABASE"
```

## Important behavior notes

- Current answer language is enforced to French.
- Questions outside indexed legal evidence return safe fallback/clarification.
- This repository does not include model fine-tuning in the current version (inference pipeline only).

## Data and compliance note

`Data/` contains sample law and notes PDFs for reproducible runs.

Before large-scale collection or redistribution from external sources (for example `https://jibaya.tn/documentation/`), verify legal and terms-of-use requirements.

## Author
Nadia Trabelsi
