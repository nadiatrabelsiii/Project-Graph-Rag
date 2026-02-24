# Agentic Graph RAG Backend for Tunisian Legal Documents

This repository is the **backend** of an **agentic Graph RAG** project focused on Tunisian legal and tax documents (laws + explanatory notes).

It is designed to ingest documents (including data sourced from `https://jibaya.tn/documentation/`), build a Neo4j knowledge graph, and serve grounded Arabic answers through an API.

## What This Project Is About

This backend solves one problem:

- Given legal documents and notes, answer user questions with **grounded references** instead of free-form guessing.

It combines:

- OCR for PDFs
- structure-aware chunking
- graph construction in Neo4j
- vector + graph retrieval
- an agentic LLM pipeline for final answer generation

## Is It RAG or Graph RAG?

It is **Graph RAG**.

Why:

- It does standard retrieval (text + vector), and
- It also uses **graph relationships** in Neo4j (`MENTIONS`, `EXPLAINS`, `RELATES_TO`, `NEXT_CHUNK`, etc.) for retrieval expansion and reasoning.

## Is It Agentic?

Yes.

The query pipeline is multi-step and stateful (not one-shot prompting):

1. Analyze query intent/entities
2. Retrieve from multiple channels (direct lookup, full-text, vectors, graph traversal)
3. Evaluate relevance
4. Generate grounded answer with citations
5. Self-check and clarification fallback

## Data Source and Input Files

- `Data/` contains sample law and note PDFs as examples.
- You can upload your own documents manually (for example from `jibeya.com`) and run the same pipeline.

## Main Technical Components

- App factory: `app/main.py`
- Query endpoint: `app/routers/query.py`
- OCR endpoint: `app/routers/ocr.py`
- Document chunking endpoint: `app/routers/documents.py`
- Graph build/stats/health endpoints: `app/routers/graph.py`
- Agentic pipeline: `app/services/rag_agent.py`
- OCR service (PaddleOCR-VL): `app/services/ocr_service.py`
- Graph builder: `app/services/graph_builder.py`
- Neo4j connection: `app/services/neo4j_service.py`

## How It Works End-to-End

```text
PDF (law or note)
  -> OCR (PaddleOCR-VL)
  -> OCR HTML/JSON
  -> Chunking (law chunker or notes chunker)
  -> Neo4j graph build (+ optional embeddings)
  -> Agentic Graph RAG query pipeline
  -> API response in Arabic with sources
```

## API Overview

### Root and Health

- `GET /` -> service info
- `GET /api/health` -> Neo4j connectivity + model readiness

### OCR

- `POST /api/ocr/process`
- Input: PDF path, optional output paths, DPI
- Output: OCR metadata + saved output paths

### Document Chunking

- `POST /api/documents/chunk`
- Input: OCR HTML path, document type (`law|notes|auto`), token settings
- Output: chunk metadata and output file path

### Graph Build and Stats

- `POST /api/graph/build` -> background graph build
- `GET /api/graph/stats` -> current graph counts

### Query

- `POST /api/query`
- Input: user legal question
- Output: answer, sources, intent, and clarification flags

## Project Setup

## 1) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure environment variables

```bash
cp .env.example .env
```

Set at least:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

Optional model/config variables are documented in `.env.example`.

## 3) Run backend locally

```bash
uvicorn app.main:create_app --factory --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`

## How To Make It Work (Practical Runbook)

## Step A: OCR a PDF

```bash
python3 ocr.py \
  --input-pdf Data/Loi2025_17Arabe.pdf \
  --output-html OCR_Law/ocr_output.html \
  --output-json OCR_Law/ocr_output.json \
  --dpi 170
```

## Step B: Chunk the OCR output

Law documents:

```bash
python3 app/services/chunk_law.py --input-dir OCR_Law --output chunks_graphrag.json
```

Notes documents:

```bash
python3 app/services/chunk_notes.py --input-dir OCR_Notes --output chunks_notes.json
```

## Step C: Build Neo4j graph

```bash
python3 app/services/graph_builder.py \
  --law-chunks chunks_graphrag.json \
  --note-chunks chunks_notes.json \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"
```

## Step D: Ask questions through API

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"ما هو الفصل 1؟"}'
```

## Deployment (Modal)

Deploy with:

```bash
./.venv/bin/modal deploy modal_app.py
```

This project is configured as a GPU-backed backend service on Modal.

## Reliability and Safety Design

The backend includes safeguards to improve answer quality:

- evidence-strength gating before answer generation
- citation-oriented output format (`[S#]`)
- grounding checks and fallback rewrites
- Arabic-output enforcement
- safe fallback if evidence is insufficient

## Current Scope

This backend is optimized for the legal/tax corpus you ingest.

If a question is outside the available corpus, the system should return a clarification/safe fallback instead of fabricating unsupported legal claims.

## Directory Structure

```text
app/
  main.py
  models.py
  routers/
    query.py
    documents.py
    graph.py
    ocr.py
  services/
    rag_agent.py
    ocr_service.py
    chunk_law.py
    chunk_notes.py
    graph_builder.py
    neo4j_service.py
Data/
  (sample PDFs)
OCR_Law/
OCR_Notes/
modal_app.py
ocr.py
requirements.txt
.env.example
```

## Note on Data Usage

Before large-scale scraping or redistribution of materials from `https://jibaya.tn/documentation/`, verify legal/terms-of-use requirements for that source.

Project realised by: Nadia Trabelsi
