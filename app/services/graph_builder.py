#!/usr/bin/env python3
"""
build_graph.py — Build a Neo4j knowledge graph from chunked legal documents.

v2: Source-aware entity extraction and relationship building.
    Fixes the article conflation bug (e.g., فصل 92 of Finance Law vs
    فصل 92 of Decree 15/2022 are now properly distinguished).

Creates:
  • Chunk nodes  (law articles, budget tables, note sections)
  • Entity nodes (source-qualified article refs, decree/law refs, tax concepts)
  • Section nodes (law document structural hierarchy)
  • Relationships: NEXT_CHUNK, MENTIONS, CROSS_REFERENCES, EXPLAINS,
                   RELATES_TO, PART_OF, SIMILAR_TO

Prerequisites:
  pip install neo4j sentence-transformers

Usage:
  python build_graph.py \
    --law-chunks  chunks_graphrag.json \
    --note-chunks chunks_notes.json \
    --neo4j-uri   bolt://localhost:7687 \
    --neo4j-user  neo4j \
    --neo4j-password password123

  Add --clear to wipe the graph before building.
  Add --skip-embeddings to skip vector embedding generation.
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAW_SOURCE_LABEL = "قانون المالية"   # label for our primary law document

TAX_CONCEPTS = [
    "الضريبة على الشركات",
    "الأداء على القيمة المضافة",
    "المعلوم على الاستهلاك",
    "المعلوم على المؤسسات ذات الصبغة الصناعية أو التجارية أو المهنية",
    "الأداء على التكوين المهني",
    "المساهمة في صندوق النهوض بالمسكن",
    "المعلوم على الأراضي غير المبنية",
    "معاليم التسجيل",
    "الطابع الجبائي",
    "الخصم من المورد",
    "الشركات الأهلية",
    "الإعفاء",
    "توقيف العمل",
    "الضريبة على الدخل",
    "الأقساط الاحتياطية",
    "ميزانية الدولة",
    "صناديق الخزينة",
    "الموارد الذاتية",
    "قروض",
]

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

RE_ARTICLE = re.compile(r"(?:الفصل|فصل)\s+(\d+)")
RE_DECREE  = re.compile(r"(?:المرسوم|مرسوم)\s+عدد\s+(\d+)\s+لسنة\s+(\d+)")
RE_LAW     = re.compile(r"(?:القانون|قانون)\s+عدد\s+(\d+)\s+لسنة\s+(\d+)")

# Patterns to identify the source law/decree/code AFTER an article reference
RE_SRC_DECREE      = re.compile(r"من\s*(?:المرسوم|مرسوم)\s+عدد\s+(\d+)\s+لسنة\s+(\d+)")
RE_SRC_SAME_DECREE = re.compile(r"(?:نفس المرسوم|المرسوم المذكور|المرسوم المشار)")
RE_SRC_FINANCE_LAW = re.compile(r"من\s*قانون المالية\s+لسنة\s+(\d+)")
RE_SRC_NAMED_LAW   = re.compile(r"من\s*(?:القانون|قانون)\s+عدد\s+(\d+)\s+لسنة\s+(\d+)")
RE_SRC_CODE        = re.compile(
    r"من\s*("
    r"مجلة الأداء على القيمة المضافة"
    r"|مجلة الضريبة على دخل الأشخاص الطبيعيين والضريبة على الشركات"
    r"|مجلة الضريبة"
    r"|مجلة الجباية المحلية"
    r"|مجلة الشغل"
    r"|مجلة\s+[\u0600-\u06FF]+"       # fallback: مجلة + one word
    r")"
)

# ---------------------------------------------------------------------------
# Source identification for article references
# ---------------------------------------------------------------------------

def identify_article_source(text: str, match_end: int,
                            default_decree: str = "المرسوم عدد 15 لسنة 2022"
                            ) -> Optional[str]:
    """Given text and the end position of an article regex match,
    determine which law/decree/code the article belongs to.

    Returns a source string like "المرسوم عدد 15 لسنة 2022" or None
    if no source qualifier is found (bare article reference).
    """
    window = text[match_end:match_end + 80]

    m = RE_SRC_DECREE.search(window[:60])
    if m:
        return f"المرسوم عدد {m.group(1)} لسنة {m.group(2)}"

    m = RE_SRC_SAME_DECREE.search(window[:50])
    if m:
        return default_decree

    m = RE_SRC_FINANCE_LAW.search(window[:60])
    if m:
        return f"قانون المالية لسنة {m.group(1)}"

    m = RE_SRC_NAMED_LAW.search(window[:60])
    if m:
        return f"القانون عدد {m.group(1)} لسنة {m.group(2)}"

    m = RE_SRC_CODE.search(window[:80])
    if m:
        return m.group(1).strip()

    if re.search(r"^\s*من\s", window[:15]):
        return "مصدر خارجي"

    return None


# ---------------------------------------------------------------------------
# Entity extraction (source-aware)
# ---------------------------------------------------------------------------

def extract_entities_law(chunk: dict) -> List[Tuple[str, str]]:
    """Extract entities from a LAW chunk.
    The chunk's own article is tagged as a Finance Law article."""
    entities: Set[Tuple[str, str]] = set()
    text = chunk["text"]

    art_num = chunk.get("article_number")
    if art_num:
        entities.add((f"فصل {art_num} — {LAW_SOURCE_LABEL}", "law_article"))

    for m in RE_DECREE.finditer(text):
        entities.add((f"مرسوم عدد {m.group(1)} لسنة {m.group(2)}", "decree_ref"))
    for m in RE_LAW.finditer(text):
        entities.add((f"قانون عدد {m.group(1)} لسنة {m.group(2)}", "law_ref"))

    for concept in TAX_CONCEPTS:
        if concept in text:
            entities.add((concept, "tax_concept"))

    return list(entities)


def extract_entities_note(chunk: dict) -> List[Tuple[str, str]]:
    """Extract entities from a NOTE chunk.
    Article references are qualified with their source law/decree."""
    entities: Set[Tuple[str, str]] = set()
    text = chunk["text"]

    for m in RE_ARTICLE.finditer(text):
        art_num = m.group(1)
        source = identify_article_source(text, m.end())
        if source:
            entities.add((f"فصل {art_num} — {source}", "external_article_ref"))
        else:
            entities.add((f"فصل {art_num}", "article_ref"))

    for m in RE_DECREE.finditer(text):
        entities.add((f"مرسوم عدد {m.group(1)} لسنة {m.group(2)}", "decree_ref"))
    for m in RE_LAW.finditer(text):
        entities.add((f"قانون عدد {m.group(1)} لسنة {m.group(2)}", "law_ref"))

    for concept in TAX_CONCEPTS:
        if concept in text:
            entities.add((concept, "tax_concept"))

    return list(entities)


# ---------------------------------------------------------------------------
# Embedding generation (optional)
# ---------------------------------------------------------------------------

def generate_embeddings(
    chunks: List[dict],
    model_name: str = "intfloat/multilingual-e5-base",
) -> Dict[str, List[float]]:
    from sentence_transformers import SentenceTransformer

    print(f"  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [f"passage: {c['text'][:512]}" for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]

    print(f"  Encoding {len(texts)} chunks …")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    return {cid: emb.tolist() for cid, emb in zip(ids, embeddings)}


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Batch-oriented graph builder — uses UNWIND for fast AuraDB inserts."""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _run(self, cypher: str, **params):
        with self.driver.session() as s:
            return s.run(cypher, **params).data()

    # ── Schema ──────────────────────────────────────────────────────

    def create_schema(self):
        stmts = [
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX chunk_type_idx IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_type)",
            "CREATE INDEX chunk_source_idx IF NOT EXISTS FOR (c:Chunk) ON (c.source)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX chunk_article_idx IF NOT EXISTS FOR (c:Chunk) ON (c.article_number)",
        ]
        for q in stmts:
            try:
                self._run(q)
            except Exception as e:
                print(f"    ⚠ {e}")

        try:
            self._run(
                "CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.text, c.section_path]"
            )
        except Exception as e:
            print(f"    ⚠ fulltext: {e}")

    def create_vector_index(self, dimensions: int = 768):
        try:
            self._run(
                f"CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
                f"FOR (c:Chunk) ON (c.embedding) "
                f"OPTIONS {{indexConfig: {{"
                f"  `vector.dimensions`: {dimensions},"
                f"  `vector.similarity_function`: 'cosine'"
                f"}}}}"
            )
            print("  ✓ Vector index created")
        except Exception as e:
            print(f"    ⚠ vector index: {e}")

    def clear(self):
        self._run("MATCH (n) DETACH DELETE n")
        print("  ✓ Graph cleared")

    # ── Batch Nodes ─────────────────────────────────────────────────

    def insert_chunks(
        self,
        chunks: List[dict],
        source: str,
        embeddings: Optional[Dict[str, list]] = None,
    ):
        rows = []
        for c in chunks:
            props = {
                "chunk_id":       c["chunk_id"],
                "chunk_type":     c["chunk_type"],
                "source":         source,
                "section_path":   c.get("section_path", ""),
                "article_number": c.get("article_number") or c.get("section_number"),
                "schedule_name":  c.get("schedule_name"),
                "document_title": c.get("document_title", ""),
                "zone":           c.get("zone", ""),
                "text":           c["text"],
                "page_start":     c.get("page_start"),
                "page_end":       c.get("page_end"),
                "token_count":    c.get("token_count", 0),
            }
            if embeddings and c["chunk_id"] in embeddings:
                props["embedding"] = embeddings[c["chunk_id"]]
            rows.append(props)

        BATCH = 50
        for i in range(0, len(rows), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MERGE (c:Chunk {chunk_id: p.chunk_id}) SET c += p",
                batch=rows[i:i+BATCH],
            )
        print(f"  ✓ {len(chunks)} chunks inserted  [source={source}]")

    def insert_entities(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Insert entity nodes using source-aware extraction."""
        all_ents: list = []
        seen: set = set()

        for c in law_chunks:
            for name, etype in extract_entities_law(c):
                if name not in seen:
                    seen.add(name)
                    all_ents.append({"name": name, "entity_type": etype})

        for c in note_chunks:
            for name, etype in extract_entities_note(c):
                if name not in seen:
                    seen.add(name)
                    all_ents.append({"name": name, "entity_type": etype})

        BATCH = 50
        for i in range(0, len(all_ents), BATCH):
            self._run(
                "UNWIND $batch AS e "
                "MERGE (ent:Entity {name: e.name}) "
                "SET ent.entity_type = e.entity_type",
                batch=all_ents[i:i+BATCH],
            )
        print(f"  ✓ {len(all_ents)} entities inserted")

        from collections import Counter
        types = Counter(e["entity_type"] for e in all_ents)
        for t, n in types.most_common():
            print(f"      {t}: {n}")

    # ── Batch Relationships ─────────────────────────────────────────

    def create_mentions(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Create MENTIONS relationships using source-aware extraction."""
        pairs = []

        for c in law_chunks:
            for name, _ in extract_entities_law(c):
                pairs.append({"cid": c["chunk_id"], "name": name})

        for c in note_chunks:
            for name, _ in extract_entities_note(c):
                pairs.append({"cid": c["chunk_id"], "name": name})

        BATCH = 100
        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (c:Chunk {chunk_id: p.cid}) "
                "MATCH (e:Entity {name: p.name}) "
                "MERGE (c)-[:MENTIONS]->(e)",
                batch=pairs[i:i+BATCH],
            )
        print(f"  ✓ {len(pairs)} MENTIONS relationships")

    def create_next(self, chunks: List[dict]):
        pairs = [
            {"a": chunks[i]["chunk_id"], "b": chunks[i+1]["chunk_id"]}
            for i in range(len(chunks) - 1)
        ]
        BATCH = 100
        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (a:Chunk {chunk_id: p.a}) "
                "MATCH (b:Chunk {chunk_id: p.b}) "
                "MERGE (a)-[:NEXT_CHUNK]->(b)",
                batch=pairs[i:i+BATCH],
            )
        print(f"  ✓ {len(pairs)} NEXT_CHUNK relationships")

    def create_cross_references(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Create CROSS_REFERENCES (source-aware).

        - Law→Law: bare article refs link to other finance law articles
        - Note→Law: only if cross_ref explicitly mentions قانون المالية
        - Decree/law entity refs are already handled by MENTIONS
        """
        pairs = []

        for c in law_chunks:
            for ref in c.get("cross_references", []):
                art_m = RE_ARTICLE.search(ref)
                if art_m and not RE_DECREE.search(ref) and not RE_LAW.search(ref):
                    pairs.append({
                        "src": c["chunk_id"],
                        "num": art_m.group(1),
                        "ref": ref,
                        "tgt_source": "law",
                    })

        for c in note_chunks:
            for ref in c.get("cross_references", []):
                art_m = RE_ARTICLE.search(ref)
                if art_m and "قانون المالية" in ref:
                    pairs.append({
                        "src": c["chunk_id"],
                        "num": art_m.group(1),
                        "ref": ref,
                        "tgt_source": "law",
                    })

        BATCH = 50
        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (src:Chunk {chunk_id: p.src}) "
                "MATCH (tgt:Chunk) "
                "WHERE tgt.source = p.tgt_source "
                "  AND tgt.article_number = p.num "
                "  AND tgt.chunk_id <> p.src "
                "MERGE (src)-[:CROSS_REFERENCES {ref_text: p.ref}]->(tgt)",
                batch=pairs[i:i+BATCH],
            )
        print(f"  ✓ {len(pairs)} CROSS_REFERENCES attempted")

    def create_explains(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Link note sections → law articles they ACTUALLY reference.

        CRITICAL FIX: Only creates EXPLAINS when a note explicitly references
        an article from our Finance Law (قانون المالية). References to other
        laws/decrees (المرسوم, مجلة, etc.) are NOT linked to our law chunks.

        Previously, any note mentioning "فصل 92" was linked to law article 92,
        even when the note referenced فصل 92 of المرسوم عدد 15 — a completely
        different law. This caused wrong answers.
        """
        pairs = []

        for nc in note_chunks:
            text = nc["text"]
            for m in RE_ARTICLE.finditer(text):
                art_num = m.group(1)
                source = identify_article_source(text, m.end())

                if source and "قانون المالية" in source:
                    pairs.append({"nid": nc["chunk_id"], "num": art_num})
                # If source is decree/code/other law/bare ref → skip

        BATCH = 50
        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (note:Chunk {chunk_id: p.nid}) "
                "MATCH (law:Chunk) "
                "WHERE law.source = 'law' AND law.article_number = p.num "
                "MERGE (note)-[:EXPLAINS]->(law)",
                batch=pairs[i:i+BATCH],
            )
        print(f"  ✓ {len(pairs)} EXPLAINS attempted (source-verified)")

    # ── RELATES_TO (shared concept bridging) ─────────────────────────

    def create_relates_to(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Bridge notes ↔ law chunks that share the same tax concepts.

        This creates the crucial cross-document links that were missing.
        E.g., a note about الأداء على القيمة المضافة gets linked to
        law articles also mentioning الأداء على القيمة المضافة.
        """
        # Build index: concept → chunk_ids (per source)
        law_concepts: Dict[str, List[str]] = {}  # concept → [chunk_ids]
        note_concepts: Dict[str, List[str]] = {}  # concept → [chunk_ids]

        for c in law_chunks:
            for concept in TAX_CONCEPTS:
                if concept in c["text"]:
                    law_concepts.setdefault(concept, []).append(c["chunk_id"])

        for c in note_chunks:
            for concept in TAX_CONCEPTS:
                if concept in c["text"]:
                    note_concepts.setdefault(concept, []).append(c["chunk_id"])

        # Create RELATES_TO between note↔law pairs sharing concepts
        pairs = []
        seen_pairs = set()
        for concept in TAX_CONCEPTS:
            law_ids = law_concepts.get(concept, [])
            note_ids = note_concepts.get(concept, [])
            for nid in note_ids:
                for lid in law_ids:
                    key = (nid, lid)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        pairs.append({
                            "nid": nid,
                            "lid": lid,
                            "concept": concept,
                        })

        BATCH = 100
        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (note:Chunk {chunk_id: p.nid}) "
                "MATCH (law:Chunk {chunk_id: p.lid}) "
                "MERGE (note)-[:RELATES_TO {concept: p.concept}]->(law)",
                batch=pairs[i:i+BATCH],
            )
        print(f"  ✓ {len(pairs)} RELATES_TO relationships (note↔law via shared concepts)")

    # ── PART_OF (structural hierarchy) ──────────────────────────────

    def create_part_of(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Create Section nodes and PART_OF hierarchy.

        Law: section_path like "المحور السابع - الإصلاح الجبائي ... > العنوان ..."
        Notes: section_path like "1. الإطار القانوني > الضريبة على الشركات"
        """
        sections: dict = {}  # (source, section_name) → section_name

        # Extract top-level sections from law chunks
        for c in law_chunks:
            sp = c.get("section_path", "")
            if not sp:
                continue
            # Take the first segment before " > " as the top section
            top = sp.split(" > ")[0].strip()
            if top and len(top) > 5:
                sections[("law", top)] = top

        # Extract top-level sections from note chunks
        for c in note_chunks:
            sp = c.get("section_path", "")
            if not sp:
                continue
            top = sp.split(" > ")[0].strip()
            if top and len(top) > 5:
                sections[("notes", top)] = top

        # Create Section nodes
        sec_rows = [{"name": name, "source": src} for (src, name), name in sections.items()]
        BATCH = 50
        for i in range(0, len(sec_rows), BATCH):
            self._run(
                "UNWIND $batch AS s "
                "MERGE (sec:Section {name: s.name, source: s.source}) ",
                batch=sec_rows[i:i+BATCH],
            )
        print(f"  ✓ {len(sec_rows)} Section nodes created")

        # Link chunks → their top section
        pairs = []
        for c in law_chunks:
            sp = c.get("section_path", "")
            if sp:
                top = sp.split(" > ")[0].strip()
                if top and len(top) > 5:
                    pairs.append({"cid": c["chunk_id"], "sec": top, "src": "law"})

        for c in note_chunks:
            sp = c.get("section_path", "")
            if sp:
                top = sp.split(" > ")[0].strip()
                if top and len(top) > 5:
                    pairs.append({"cid": c["chunk_id"], "sec": top, "src": "notes"})

        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (c:Chunk {chunk_id: p.cid}) "
                "MATCH (s:Section {name: p.sec, source: p.src}) "
                "MERGE (c)-[:PART_OF]->(s)",
                batch=pairs[i:i+BATCH],
            )
        print(f"  ✓ {len(pairs)} PART_OF relationships")

    # ── Embeddings → Neo4j (for Modal-based generation) ─────────────

    def store_embeddings(self, embeddings: Dict[str, list]):
        """Store pre-computed embeddings on Chunk nodes."""
        rows = [{"cid": cid, "emb": emb} for cid, emb in embeddings.items()]
        BATCH = 20
        for i in range(0, len(rows), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (c:Chunk {chunk_id: p.cid}) "
                "SET c.embedding = p.emb",
                batch=rows[i:i+BATCH],
            )
        print(f"  ✓ {len(rows)} embeddings stored")

    def create_similar_to(self, threshold: float = 0.78, max_neighbors: int = 5):
        """Create SIMILAR_TO edges between semantically similar chunks.

        Uses embeddings already stored on Chunk nodes.
        Computes cosine similarity in a Cypher batch.
        """
        # Get all chunks with embeddings
        rows = self._run(
            "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
            "RETURN c.chunk_id AS id, c.embedding AS emb, c.source AS source"
        )
        if len(rows) < 2:
            print("  ⚠ Not enough embeddings for SIMILAR_TO")
            return

        print(f"  Computing similarities for {len(rows)} embedded chunks …")

        import math
        def cosine_sim(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            return dot / (na * nb) if na > 0 and nb > 0 else 0

        pairs = []
        for i, ri in enumerate(rows):
            sims = []
            for j, rj in enumerate(rows):
                if i == j:
                    continue
                sim = cosine_sim(ri["emb"], rj["emb"])
                if sim >= threshold:
                    sims.append((rj["id"], sim))
            # Keep top N neighbors
            sims.sort(key=lambda x: x[1], reverse=True)
            for tid, sim in sims[:max_neighbors]:
                pairs.append({
                    "src": ri["id"],
                    "tgt": tid,
                    "score": round(sim, 4),
                })

        # Deduplicate (keep only one direction)
        seen = set()
        deduped = []
        for p in pairs:
            key = tuple(sorted([p["src"], p["tgt"]]))
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        BATCH = 100
        for i in range(0, len(deduped), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (a:Chunk {chunk_id: p.src}) "
                "MATCH (b:Chunk {chunk_id: p.tgt}) "
                "MERGE (a)-[:SIMILAR_TO {score: p.score}]->(b)",
                batch=deduped[i:i+BATCH],
            )
        print(f"  ✓ {len(deduped)} SIMILAR_TO relationships (threshold={threshold})")

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self):
        rows = self._run(
            "MATCH (c:Chunk) WITH count(c) AS chunks "
            "MATCH (e:Entity) WITH chunks, count(e) AS entities "
            "OPTIONAL MATCH ()-[r]->() "
            "RETURN chunks, entities, count(r) AS rels"
        )
        if rows:
            s = rows[0]
            print(f"\n  📊 Chunks: {s['chunks']}  |  Entities: {s['entities']}  |  Relationships: {s['rels']}")

        rows = self._run(
            "MATCH ()-[r]->() "
            "RETURN type(r) AS rel_type, count(r) AS cnt "
            "ORDER BY cnt DESC"
        )
        for r in rows:
            print(f"      {r['rel_type']}: {r['cnt']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build Neo4j knowledge graph for Graph RAG")
    ap.add_argument("--law-chunks",      default="chunks_graphrag.json")
    ap.add_argument("--note-chunks",     default="chunks_notes.json")
    ap.add_argument("--neo4j-uri",       default="neo4j+s://f6b29f07.databases.neo4j.io")
    ap.add_argument("--neo4j-user",      default="neo4j")
    ap.add_argument("--neo4j-password",  default="h9lAbhKlanyl_zahfllUTrNZ82ADTNpbcntqgAyBAsU")
    ap.add_argument("--embed-model",     default="intfloat/multilingual-e5-base")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--clear",           action="store_true", help="Wipe graph first")
    args = ap.parse_args()

    # Load chunks
    print("═" * 60)
    print("Loading chunks …")
    law_chunks  = json.loads(Path(args.law_chunks).read_text("utf-8"))  if Path(args.law_chunks).exists()  else []
    note_chunks = json.loads(Path(args.note_chunks).read_text("utf-8")) if Path(args.note_chunks).exists() else []
    print(f"  Law: {len(law_chunks)}   Notes: {len(note_chunks)}   Total: {len(law_chunks)+len(note_chunks)}")

    # Embeddings
    embeddings: Dict[str, list] = {}
    if not args.skip_embeddings:
        print("\nGenerating embeddings …")
        embeddings = generate_embeddings(law_chunks + note_chunks, args.embed_model)
    else:
        print("\n(skipping embeddings)")

    # Build graph
    print("\nConnecting to Neo4j …")
    gb = GraphBuilder(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        if args.clear:
            gb.clear()

        print("\n[1/9] Schema")
        gb.create_schema()

        print("\n[2/9] Chunk nodes")
        gb.insert_chunks(law_chunks,  source="law",   embeddings=embeddings)
        gb.insert_chunks(note_chunks, source="notes", embeddings=embeddings)

        print("\n[3/9] Entity nodes (source-aware)")
        gb.insert_entities(law_chunks, note_chunks)

        print("\n[4/9] MENTIONS (source-aware)")
        gb.create_mentions(law_chunks, note_chunks)

        print("\n[5/9] NEXT_CHUNK + CROSS_REFERENCES")
        gb.create_next(law_chunks)
        gb.create_next(note_chunks)
        gb.create_cross_references(law_chunks, note_chunks)

        print("\n[6/9] EXPLAINS (source-verified: notes → law articles)")
        gb.create_explains(law_chunks, note_chunks)

        print("\n[7/9] RELATES_TO (shared concept bridging)")
        gb.create_relates_to(law_chunks, note_chunks)

        print("\n[8/9] PART_OF (structural hierarchy)")
        gb.create_part_of(law_chunks, note_chunks)

        print("\n[9/9] Embeddings + SIMILAR_TO")
        if embeddings:
            gb.store_embeddings(embeddings)
            dims = len(next(iter(embeddings.values())))
            gb.create_vector_index(dimensions=dims)
            gb.create_similar_to(threshold=0.78, max_neighbors=5)
        else:
            print("  (skipped — run embed_graph.py on Modal to add embeddings)")

        gb.stats()
        print("\n✅  Graph build complete!")
    finally:
        gb.close()


if __name__ == "__main__":
    main()
