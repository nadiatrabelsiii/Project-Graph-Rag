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
    --neo4j-password "$NEO4J_PASSWORD" \
    --neo4j-database "$NEO4J_DATABASE"

  Add --clear to wipe the graph before building.
  Add --skip-embeddings to skip vector embedding generation.
"""

import json
import re
import os
import argparse
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

from app.config import load_environment

load_environment()

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

# Canonical legal topics (French labels) with Arabic/French aliases.
TOPIC_LEXICON: Dict[str, List[str]] = {
    "TVA": [
        "الأداء على القيمة المضافة", "tva", "taxe sur la valeur ajoutee",
        "taxe sur la valeur ajoutée", "valeur ajoutee", "valeur ajoutée",
    ],
    "IS": [
        "الضريبة على الشركات", "impot sur les societes", "impot sur les sociétés", "is",
    ],
    "IRPP": [
        "الضريبة على الدخل", "impot sur le revenu", "impot sur le revenu des personnes physiques",
    ],
    "DroitsEnregistrement": [
        "معاليم التسجيل", "droits d'enregistrement", "droits denregistrement",
    ],
    "RetentionSource": [
        "الخصم من المورد", "retenue a la source", "retenue à la source",
    ],
    "TimbreFiscal": [
        "الطابع الجبائي", "timbre fiscal",
    ],
    "DroitsConsommation": [
        "المعلوم على الاستهلاك", "droits de consommation",
    ],
    "SocietesCommunautaires": [
        "الشركات الأهلية", "societes communautaires", "sociétés communautaires",
    ],
    "Exoneration": [
        "الإعفاء", "exoneration", "exonération",
    ],
    "BudgetEtat": [
        "ميزانية الدولة", "budget de l'etat", "budget de l'état",
    ],
}


def _normalize_latin(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _is_arabic_fragment(term: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", term))


def _extract_year(value: Optional[str], fallback_text: str = "") -> Optional[str]:
    if value:
        m = re.search(r"(19\d{2}|20\d{2})", str(value))
        if m:
            return m.group(1)
    m = re.search(r"(19\d{2}|20\d{2})", fallback_text or "")
    return m.group(1) if m else None


def _chunk_year(chunk: dict) -> Optional[str]:
    return _extract_year(chunk.get("document_year"), chunk.get("source_file", ""))


def extract_topics(text: str) -> Set[str]:
    raw = text or ""
    raw_low = raw.lower()
    raw_norm = _normalize_latin(raw)
    matched: Set[str] = set()

    for topic, aliases in TOPIC_LEXICON.items():
        for alias in aliases:
            a = alias.strip()
            if not a:
                continue
            if _is_arabic_fragment(a):
                if a in raw:
                    matched.add(topic)
                    break
            else:
                an = _normalize_latin(a)
                if an in raw_norm or an in raw_low:
                    matched.add(topic)
                    break
    return matched


def extract_chunk_topics(chunk: dict) -> Set[str]:
    pool = " ".join([
        str(chunk.get("document_title", "")),
        str(chunk.get("section_path", "")),
        str(chunk.get("text", "")),
    ])
    return extract_topics(pool)


def _tokenize_legal_terms(text: str) -> Set[str]:
    stop = {
        "dans", "avec", "pour", "les", "des", "une", "sur", "par", "aux", "est",
        "ce", "cet", "cette", "dans", "lors", "ainsi", "selon",
        "من", "على", "في", "الى", "إلى", "هذا", "هذه", "ذلك", "تلك", "مع", "عن",
    }
    toks = re.findall(r"[A-Za-z\u00C0-\u017F\u0600-\u06FF0-9]{3,}", text.lower())
    return {t for t in toks if t not in stop}


def _article_base_number(chunk: dict) -> Optional[str]:
    raw = str(chunk.get("article_number") or "").split("_")[0].strip()
    return raw if raw.isdigit() else None


def _build_law_article_anchor(law_chunks: List[dict]) -> Dict[str, Dict[str, str]]:
    """Choose one canonical law chunk per (year, article_number).

    Priority:
    1) chunk_type == article
    2) lower token_count
    3) stable lexical order by chunk_id
    """
    best: Dict[Tuple[str, str], Tuple[int, int, str, str]] = {}
    for c in law_chunks:
        year = _chunk_year(c)
        art = _article_base_number(c)
        if not year or not art:
            continue
        rank = 0 if c.get("chunk_type") == "article" else 1
        tok = int(c.get("token_count", 0) or 0)
        cid = str(c["chunk_id"])
        key = (year, art)
        cur = (rank, tok, cid, cid)
        if key not in best or cur < best[key]:
            best[key] = cur

    out: Dict[str, Dict[str, str]] = defaultdict(dict)
    for (year, art), (_, _, _, cid) in best.items():
        out[year][art] = cid
    return out


def _build_law_doc_identity_map(law_chunks: List[dict]) -> Dict[Tuple[str, str, str], str]:
    """Map (source_type, source_number, source_year) -> corpus document_year.

    source_year is the year written in the referenced law/decree identifier
    (e.g., القانون عدد 17 لسنة 2025), while document_year is the fiscal/corpus year
    we use on chunks (e.g., 2026).
    """
    out: Dict[Tuple[str, str, str], str] = {}
    seen_source_files: set[str] = set()
    for c in law_chunks:
        source_file = str(c.get("source_file", "")).strip()
        if not source_file or source_file in seen_source_files:
            continue
        seen_source_files.add(source_file)

        doc_year = _chunk_year(c)
        if not doc_year:
            continue
        title = str(c.get("document_title", ""))
        if not title:
            continue

        m_ar = RE_DOC_LAW_ID_AR.search(title)
        if m_ar:
            out[("law", m_ar.group(1), m_ar.group(2))] = doc_year

        m_fr = RE_DOC_LAW_ID_FR.search(title)
        if m_fr:
            out[("law", m_fr.group(1), m_fr.group(2))] = doc_year
    return out


def _parse_cross_ref_entry(ref_text: str) -> Optional[dict]:
    ref = re.sub(r"\s+", " ", str(ref_text or "")).strip()
    if not ref:
        return None

    m_ar = RE_REF_ARTICLE_SRC_AR.search(ref)
    if m_ar:
        return {
            "article_number": m_ar.group(1),
            "source_type": "law" if m_ar.group(2) == "القانون" else "decree",
            "source_number": m_ar.group(3),
            "source_year": m_ar.group(4),
            "ref_text": ref,
        }

    m_fr = RE_REF_ARTICLE_SRC_FR.search(ref)
    if m_fr:
        src_kind = m_fr.group(2).lower()
        return {
            "article_number": "1" if m_fr.group(1).lower().startswith("prem") or m_fr.group(1).lower() == "1er" else re.sub(r"[^\d]", "", m_fr.group(1)),
            "source_type": "law" if "loi" in src_kind else "decree",
            "source_number": re.sub(r"[–—]", "-", m_fr.group(3)),
            "source_year": m_fr.group(4),
            "ref_text": ref,
        }

    m_fr_alt = RE_REF_ARTICLE_SRC_FR_ALT.search(ref)
    if m_fr_alt:
        src_kind = m_fr_alt.group(2).lower()
        src_num = re.sub(r"[–—]", "-", m_fr_alt.group(3))
        src_year = None
        y_m = re.match(r"^(19\d{2}|20\d{2})-", src_num)
        if y_m:
            src_year = y_m.group(1)
        if src_year:
            return {
                "article_number": "1" if m_fr_alt.group(1).lower().startswith("prem") or m_fr_alt.group(1).lower() == "1er" else re.sub(r"[^\d]", "", m_fr_alt.group(1)),
                "source_type": "law" if "loi" in src_kind else "decree",
                "source_number": src_num,
                "source_year": src_year,
                "ref_text": ref,
            }

    # Fallback to bare article mention (no explicit source law/decree id).
    m_bare = RE_ARTICLE.search(ref)
    if m_bare:
        return {
            "article_number": m_bare.group(1),
            "source_type": None,
            "source_number": None,
            "source_year": None,
            "ref_text": ref,
        }
    return None

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

RE_ARTICLE = re.compile(r"(?:الفصل|فصل|article|art\.?)\s*[:\-\.]?\s*(\d+)", re.IGNORECASE)
RE_DECREE  = re.compile(
    r"(?:المرسوم|مرسوم|decret|décret)\s*(?:عدد|n[°oº]?)?\s*(\d+)\s*(?:لسنة|de|/)\s*(\d{4})",
    re.IGNORECASE,
)
RE_LAW     = re.compile(
    r"(?:القانون|قانون|loi)\s*(?:عدد|n[°oº]?)?\s*(\d+)\s*(?:لسنة|de|/)\s*(\d{4})",
    re.IGNORECASE,
)

# Patterns to identify the source law/decree/code AFTER an article reference
RE_SRC_DECREE      = re.compile(r"من\s*(?:المرسوم|مرسوم)\s+عدد\s+(\d+)\s+لسنة\s+(\d+)")
RE_SRC_SAME_DECREE = re.compile(r"(?:نفس المرسوم|المرسوم المذكور|المرسوم المشار)")
RE_SRC_FINANCE_LAW = re.compile(r"من\s*قانون المالية\s+لسنة\s+(\d+)")
RE_SRC_NAMED_LAW   = re.compile(r"من\s*(?:القانون|قانون)\s+عدد\s+(\d+)\s+لسنة\s+(\d+)")
RE_SRC_DECREE_FR = re.compile(r"(?:decret|décret)\s+n?[°o]?\s*(\d+)\s*(?:de|/)\s*(\d{4})", re.IGNORECASE)
RE_SRC_FINANCE_LAW_FR = re.compile(r"(?:loi de finances)\s*(?:pour|de)\s*(\d{4})", re.IGNORECASE)
RE_SRC_NAMED_LAW_FR = re.compile(r"(?:loi)\s+n?[°o]?\s*(\d+)\s*(?:de|/)\s*(\d{4})", re.IGNORECASE)
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
RE_DOC_LAW_ID_AR = re.compile(r"قانون\s+عدد\s+(\d+)\s+لسنة\s+(\d{4})")
RE_DOC_LAW_ID_FR = re.compile(r"loi\s+n[°oº]?\s*([\d\-]+)\s+du\s+[^,\n]*?(19\d{2}|20\d{2})", re.IGNORECASE)

RE_REF_ARTICLE_SRC_AR = re.compile(
    r"(?:الفصل|فصل)\s+(\d+)\s*(?:مكرر|مكرّر|\(جديد\)|\(مكرر\))?\s*"
    r"(?:من|بالفصل)\s+"
    r"(القانون|المرسوم|الأمر)\s+عدد\s+(\d+)\s+لسنة\s+(\d{4})",
    re.IGNORECASE,
)
RE_REF_ARTICLE_SRC_FR = re.compile(
    r"(?:article|art\.?)\s*(\d+|1er|premi(?:er|ere))\s*"
    r"(?:du|de la|de l['’])\s*"
    r"(loi|decret|décret)\s+n?[°oº]?\s*([\d\-]+)\s*(?:de|/)\s*(\d{4})",
    re.IGNORECASE,
)
RE_REF_ARTICLE_SRC_FR_ALT = re.compile(
    r"(?:article|art\.?)\s*(\d+|1er|premi(?:er|ere)).{0,60}?"
    r"(loi|decret|décret)\s+n?[°oº]?\s*([\d]+(?:[-–—]\d+)?)",
    re.IGNORECASE,
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

    m = RE_SRC_DECREE_FR.search(window[:80])
    if m:
        return f"decret n {m.group(1)} de {m.group(2)}"

    m = RE_SRC_FINANCE_LAW_FR.search(window[:80])
    if m:
        return f"loi de finances {m.group(1)}"

    m = RE_SRC_NAMED_LAW_FR.search(window[:80])
    if m:
        return f"loi n {m.group(1)} de {m.group(2)}"

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
    for topic in extract_chunk_topics(chunk):
        entities.add((f"topic:{topic}", "tax_concept"))

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
    for topic in extract_chunk_topics(chunk):
        entities.add((f"topic:{topic}", "tax_concept"))

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

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def _run(self, cypher: str, **params):
        session_kwargs = {"database": self.database} if self.database else {}
        with self.driver.session(**session_kwargs) as s:
            return s.run(cypher, **params).data()

    # ── Schema ──────────────────────────────────────────────────────

    def create_schema(self):
        stmts = [
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT external_article_id IF NOT EXISTS FOR (x:ExternalArticle) REQUIRE x.external_id IS UNIQUE",
            "CREATE INDEX chunk_type_idx IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_type)",
            "CREATE INDEX chunk_source_idx IF NOT EXISTS FOR (c:Chunk) ON (c.source)",
            "CREATE INDEX chunk_year_idx IF NOT EXISTS FOR (c:Chunk) ON (c.document_year)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX chunk_article_idx IF NOT EXISTS FOR (c:Chunk) ON (c.article_number)",
            "CREATE INDEX document_year_idx IF NOT EXISTS FOR (d:Document) ON (d.year)",
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
            doc_year = _chunk_year(c)
            props = {
                "chunk_id":       c["chunk_id"],
                "chunk_type":     c["chunk_type"],
                "source":         source,
                "source_file":    c.get("source_file", ""),
                "document_year":  doc_year,
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

    def create_documents(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Create Document nodes, link chunks to documents, and connect same-year docs."""
        doc_map: Dict[Tuple[str, str], dict] = {}
        chunk_doc_pairs: list[dict] = []

        def _ingest(chunks: List[dict], source: str):
            for c in chunks:
                source_file = str(c.get("source_file", "")).strip()
                if not source_file:
                    continue
                year = _chunk_year(c)
                key = (source, source_file)
                if key not in doc_map:
                    doc_map[key] = {
                        "doc_id": f"{source}:{source_file}",
                        "source": source,
                        "source_file": source_file,
                        "title": c.get("document_title", ""),
                        "year": year,
                    }
                chunk_doc_pairs.append({
                    "cid": c["chunk_id"],
                    "doc_id": f"{source}:{source_file}",
                })

        _ingest(law_chunks, "law")
        _ingest(note_chunks, "notes")

        docs = list(doc_map.values())
        BATCH = 50
        for i in range(0, len(docs), BATCH):
            self._run(
                "UNWIND $batch AS d "
                "MERGE (doc:Document {doc_id: d.doc_id}) "
                "SET doc.source = d.source, "
                "    doc.source_file = d.source_file, "
                "    doc.title = d.title, "
                "    doc.year = d.year",
                batch=docs[i:i+BATCH],
            )

        for i in range(0, len(chunk_doc_pairs), 100):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (c:Chunk {chunk_id: p.cid}) "
                "MATCH (d:Document {doc_id: p.doc_id}) "
                "MERGE (c)-[:IN_DOCUMENT]->(d)",
                batch=chunk_doc_pairs[i:i+100],
            )

        self._run(
            "MATCH (l:Document {source:'law'}), (n:Document {source:'notes'}) "
            "WHERE l.year IS NOT NULL AND n.year = l.year "
            "MERGE (n)-[:SAME_YEAR_AS]->(l)"
        )
        print(f"  ✓ {len(docs)} Document nodes, {len(chunk_doc_pairs)} IN_DOCUMENT links")

    def create_topics_and_object_links(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Create topic graph and year-aligned note->law article links by shared topics."""
        topic_rows = [{"name": t} for t in sorted(TOPIC_LEXICON.keys())]
        self._run(
            "UNWIND $topics AS t "
            "MERGE (tp:Topic {name: t.name})",
            topics=topic_rows,
        )

        has_topic_pairs: list[dict] = []
        law_by_year_topic: Dict[str, Dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        note_by_year_topic: Dict[str, Dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        law_map: Dict[str, dict] = {}
        object_note_map: Dict[str, dict] = {}
        law_article_anchor = _build_law_article_anchor(law_chunks)

        def _is_object_chunk(chunk: dict) -> bool:
            section_path = str(chunk.get("section_path", ""))
            text = str(chunk.get("text", "")).strip()
            section_leaf = section_path.split(">")[-1].strip() if section_path else ""
            if re.match(r"^(?:objet|الموضوع)\b", section_leaf, re.IGNORECASE):
                return True
            first_line = text.splitlines()[0].strip() if text else ""
            if re.match(r"^(?:objet|الموضوع)\s*[:\-]", first_line, re.IGNORECASE):
                return True
            return False

        for c in law_chunks:
            topics = extract_chunk_topics(c)
            year = _chunk_year(c)
            cid = c["chunk_id"]
            law_map[cid] = c
            for t in topics:
                has_topic_pairs.append({"cid": cid, "topic": t})
                art = _article_base_number(c)
                if year and art:
                    anchor = law_article_anchor.get(year, {}).get(art)
                    if anchor == cid:
                        law_by_year_topic[year][t].append(cid)

        for c in note_chunks:
            topics = extract_chunk_topics(c)
            year = _chunk_year(c)
            cid = c["chunk_id"]
            is_object = _is_object_chunk(c)
            if year and is_object:
                object_note_map[cid] = c
            for t in topics:
                has_topic_pairs.append({"cid": cid, "topic": t})
                if year and is_object:
                    note_by_year_topic[year][t].append(cid)

        for i in range(0, len(has_topic_pairs), 120):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (c:Chunk {chunk_id: p.cid}) "
                "MATCH (t:Topic {name: p.topic}) "
                "MERGE (c)-[:HAS_TOPIC]->(t)",
                batch=has_topic_pairs[i:i+120],
            )

        pair_topics: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
        explicit_marker = "__EXPLICIT_ARTICLE_REF__"

        # High-confidence links: explicit article references in note object sections
        # -> same-year canonical law article chunk.
        note_has_explicit: Dict[str, bool] = defaultdict(bool)
        for nid, note in object_note_map.items():
            year = _chunk_year(note)
            if not year:
                continue
            text = str(note.get("text", ""))
            for m in RE_ARTICLE.finditer(text):
                art_num = m.group(1)
                src = identify_article_source(text, m.end())
                if src:
                    src_low = src.lower()
                    if "مرسوم" in src or "decret" in src_low or "مجلة" in src:
                        continue
                lid = law_article_anchor.get(year, {}).get(art_num)
                if lid:
                    pair_topics[(nid, lid, year)].add(explicit_marker)
                    note_has_explicit[nid] = True

        for year, topic_map in note_by_year_topic.items():
            for topic, note_ids in topic_map.items():
                law_ids = law_by_year_topic.get(year, {}).get(topic, [])
                for nid in note_ids:
                    if not law_ids:
                        continue
                    note_terms = _tokenize_legal_terms(str(object_note_map.get(nid, {}).get("text", "")))
                    scored: list[Tuple[float, str]] = []
                    if note_has_explicit.get(nid):
                        candidate_law_ids = [lid for lid in law_ids if (nid, lid, year) in pair_topics]
                    else:
                        candidate_law_ids = law_ids
                    for lid in candidate_law_ids:
                        law_terms = _tokenize_legal_terms(str(law_map.get(lid, {}).get("text", "")))
                        overlap = len(note_terms & law_terms)
                        if overlap >= (1 if note_has_explicit.get(nid) else 3):
                            scored.append((float(overlap), lid))
                    scored.sort(reverse=True)
                    kept_k = 2 if note_has_explicit.get(nid) else 1
                    kept = [lid for _, lid in scored[:kept_k]] if scored else []
                    for lid in kept:
                        pair_topics[(nid, lid, year)].add(topic)

        object_links = []
        for (nid, lid, year), topics in pair_topics.items():
            explicit_ref = explicit_marker in topics
            topic_list = sorted([t for t in topics if t != explicit_marker])
            if not explicit_ref and not topic_list:
                continue
            object_links.append(
                {
                    "nid": nid,
                    "lid": lid,
                    "year": year,
                    "topics": topic_list,
                    "overlap": len(topic_list),
                    "explicit_ref": explicit_ref,
                }
            )

        for i in range(0, len(object_links), 80):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (n:Chunk {chunk_id: p.nid, source: 'notes'}) "
                "MATCH (l:Chunk {chunk_id: p.lid, source: 'law'}) "
                "WHERE n.document_year = p.year AND l.document_year = p.year "
                "MERGE (n)-[r:ABOUT_ARTICLE]->(l) "
                "SET r.year = p.year, "
                "    r.shared_topics = p.topics, "
                "    r.topic_overlap = p.overlap, "
                "    r.explicit_article_ref = p.explicit_ref",
                batch=object_links[i:i+80],
            )
        print(
            f"  ✓ {len(has_topic_pairs)} HAS_TOPIC links, "
            f"{len(object_links)} ABOUT_ARTICLE links (same-year topic overlap)"
        )

    def create_next(self, chunks: List[dict]):
        pairs = []
        for i in range(len(chunks) - 1):
            a = chunks[i]
            b = chunks[i + 1]
            if a.get("source_file") != b.get("source_file"):
                continue
            pairs.append({"a": a["chunk_id"], "b": b["chunk_id"]})
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
        """Create CROSS_REFERENCES with article+year aware routing.

        - If referenced law/decree article exists in corpus -> link to that law chunk
        - Else -> link to ExternalArticle node (keeps relation for future data growth)
        """
        law_article_anchor = _build_law_article_anchor(law_chunks)
        law_doc_id_map = _build_law_doc_identity_map(law_chunks)
        pair_map: Dict[Tuple[str, str], dict] = {}
        external_map: Dict[Tuple[str, str], dict] = {}

        def _collect_from_chunk(chunk: dict):
            src_id = str(chunk["chunk_id"])
            src_doc_year = _chunk_year(chunk)
            if not src_doc_year:
                return
            src_art = _article_base_number(chunk)

            for ref_text in chunk.get("cross_references", []) or []:
                parsed = _parse_cross_ref_entry(str(ref_text))
                if not parsed:
                    continue
                art_num = str(parsed.get("article_number") or "").strip()
                if not art_num:
                    continue
                if src_art and src_art == art_num:
                    continue

                target_doc_year = None
                src_type = parsed.get("source_type")
                src_num = parsed.get("source_number")
                src_year = parsed.get("source_year")
                if src_type == "law" and src_num and src_year:
                    target_doc_year = law_doc_id_map.get(("law", str(src_num), str(src_year)))
                if target_doc_year is None:
                    target_doc_year = src_doc_year if parsed.get("source_type") is None else str(src_year or "")
                if not target_doc_year:
                    target_doc_year = src_doc_year

                tgt_id = law_article_anchor.get(target_doc_year, {}).get(art_num)
                if tgt_id and tgt_id != src_id:
                    key = (src_id, tgt_id)
                    if key not in pair_map:
                        pair_map[key] = {
                            "src": src_id,
                            "tgt": tgt_id,
                            "ref": parsed["ref_text"],
                            "count": 1,
                            "source_type": src_type or "unspecified",
                            "source_number": src_num,
                            "source_year": src_year,
                            "target_doc_year": target_doc_year,
                            "same_year": str(src_doc_year) == str(target_doc_year),
                        }
                    else:
                        pair_map[key]["count"] += 1
                    continue

                # Keep external references explicitly for laws/decrees not loaded yet.
                if src_type in ("law", "decree") and src_num and src_year:
                    ext_id = f"{src_type}:{src_num}:{src_year}:art:{art_num}"
                    key = (src_id, ext_id)
                    if key not in external_map:
                        external_map[key] = {
                            "src": src_id,
                            "external_id": ext_id,
                            "source_type": src_type,
                            "source_number": str(src_num),
                            "source_year": str(src_year),
                            "article_number": art_num,
                            "ref": parsed["ref_text"],
                            "count": 1,
                        }
                    else:
                        external_map[key]["count"] += 1

        for c in law_chunks:
            _collect_from_chunk(c)
        for c in note_chunks:
            _collect_from_chunk(c)

        pairs = list(pair_map.values())
        BATCH = 80
        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (src:Chunk {chunk_id: p.src}) "
                "MATCH (tgt:Chunk {chunk_id: p.tgt}) "
                "MERGE (src)-[r:CROSS_REFERENCES]->(tgt) "
                "SET r.ref_text = coalesce(r.ref_text, p.ref), "
                "    r.mentions = coalesce(r.mentions, 0) + p.count, "
                "    r.source_type = p.source_type, "
                "    r.source_number = p.source_number, "
                "    r.source_year = p.source_year, "
                "    r.target_doc_year = p.target_doc_year, "
                "    r.same_year = p.same_year",
                batch=pairs[i:i+BATCH],
            )

        external_pairs = list(external_map.values())
        for i in range(0, len(external_pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (src:Chunk {chunk_id: p.src}) "
                "MERGE (x:ExternalArticle {external_id: p.external_id}) "
                "SET x.source_type = p.source_type, "
                "    x.source_number = p.source_number, "
                "    x.source_year = p.source_year, "
                "    x.article_number = p.article_number "
                "MERGE (src)-[r:CITES_EXTERNAL_ARTICLE]->(x) "
                "SET r.ref_text = coalesce(r.ref_text, p.ref), "
                "    r.mentions = coalesce(r.mentions, 0) + p.count",
                batch=external_pairs[i:i+BATCH],
            )

        print(
            f"  ✓ {len(pairs)} CROSS_REFERENCES relationships, "
            f"{len(external_pairs)} CITES_EXTERNAL_ARTICLE relationships"
        )

    def create_explains(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Link note sections → law articles they ACTUALLY reference.

        CRITICAL FIX: Only creates EXPLAINS when a note explicitly references
        an article from our Finance Law (قانون المالية). References to other
        laws/decrees (المرسوم, مجلة, etc.) are NOT linked to our law chunks.

        Previously, any note mentioning "فصل 92" was linked to law article 92,
        even when the note referenced فصل 92 of المرسوم عدد 15 — a completely
        different law. This caused wrong answers.
        """
        law_article_anchor = _build_law_article_anchor(law_chunks)
        pairs: list[dict] = []
        seen_pairs: set[Tuple[str, str]] = set()

        for nc in note_chunks:
            text = nc["text"]
            note_year = _chunk_year(nc)
            if not note_year:
                continue
            for m in RE_ARTICLE.finditer(text):
                art_num = m.group(1)
                source = identify_article_source(text, m.end())

                # Reject known non-finance-law mentions.
                if source:
                    src_low = source.lower()
                    if "مرسوم" in source or "decret" in src_low:
                        continue
                    if "مجلة" in source:
                        continue

                tgt_id = law_article_anchor.get(note_year, {}).get(art_num)
                if not tgt_id:
                    continue
                key = (nc["chunk_id"], tgt_id)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append({"nid": nc["chunk_id"], "lid": tgt_id, "year": note_year})

        BATCH = 50
        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (note:Chunk {chunk_id: p.nid}) "
                "MATCH (law:Chunk {chunk_id: p.lid}) "
                "WHERE note.document_year = p.year AND law.document_year = p.year "
                "MERGE (note)-[:EXPLAINS {year: p.year}]->(law)",
                batch=pairs[i:i+BATCH],
            )
        print(f"  ✓ {len(pairs)} EXPLAINS attempted (same-year constrained)")

    # ── RELATES_TO (shared concept bridging) ─────────────────────────

    def create_relates_to(self, law_chunks: List[dict], note_chunks: List[dict]):
        """Bridge notes ↔ law chunks that share bilingual legal topics in same year.

        Uses TOPIC_LEXICON (Arabic + French aliases) and lexical overlap ranking
        to avoid dense noisy Cartesian links.
        """
        law_by_year_topic: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        note_by_year_topic: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        law_map: Dict[str, dict] = {}
        note_map: Dict[str, dict] = {}

        for c in law_chunks:
            year = _chunk_year(c)
            if not year:
                continue
            # Keep RELATES_TO focused on legal article content, not budget tables.
            if c.get("chunk_type") not in ("article", "article_subchunk"):
                continue
            cid = c["chunk_id"]
            law_map[cid] = c
            for topic in extract_chunk_topics(c):
                law_by_year_topic[year][topic].append(cid)

        for c in note_chunks:
            year = _chunk_year(c)
            if not year:
                continue
            cid = c["chunk_id"]
            note_map[cid] = c
            for topic in extract_chunk_topics(c):
                note_by_year_topic[year][topic].append(cid)

        pairs: list[dict] = []
        seen_pairs: set[Tuple[str, str, str]] = set()
        for year in sorted(set(law_by_year_topic.keys()) & set(note_by_year_topic.keys())):
            for topic in sorted(set(law_by_year_topic[year].keys()) & set(note_by_year_topic[year].keys())):
                law_ids = law_by_year_topic[year][topic]
                note_ids = note_by_year_topic[year][topic]
                for nid in note_ids:
                    note_terms = _tokenize_legal_terms(str(note_map.get(nid, {}).get("text", "")))
                    scored: list[Tuple[int, str]] = []
                    for lid in law_ids:
                        law_terms = _tokenize_legal_terms(str(law_map.get(lid, {}).get("text", "")))
                        overlap = len(note_terms & law_terms)
                        if overlap >= 1:
                            scored.append((overlap, lid))
                    scored.sort(reverse=True)
                    for _, lid in scored[:1]:
                        key = (nid, lid, topic)
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        pairs.append({"nid": nid, "lid": lid, "concept": topic, "year": year})

        BATCH = 100
        for i in range(0, len(pairs), BATCH):
            self._run(
                "UNWIND $batch AS p "
                "MATCH (note:Chunk {chunk_id: p.nid}) "
                "MATCH (law:Chunk {chunk_id: p.lid}) "
                "WHERE note.document_year = p.year AND law.document_year = p.year "
                "MERGE (note)-[:RELATES_TO {concept: p.concept, year: p.year}]->(law)",
                batch=pairs[i:i+BATCH],
            )
        print(f"  ✓ {len(pairs)} RELATES_TO relationships (same-year concept bridging)")

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
    ap.add_argument("--neo4j-uri",       default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--neo4j-user",      default=os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "neo4j"))
    ap.add_argument("--neo4j-password",  default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--neo4j-database",  default=os.environ.get("NEO4J_DATABASE", ""))
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
    if not args.skip_embeddings and (law_chunks or note_chunks):
        print("\nGenerating embeddings …")
        embeddings = generate_embeddings(law_chunks + note_chunks, args.embed_model)
    else:
        print("\n(skipping embeddings)")

    # Build graph
    print("\nConnecting to Neo4j …")
    gb = GraphBuilder(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.neo4j_database or None,
    )
    try:
        if args.clear:
            gb.clear()

        print("\n[1/11] Schema")
        gb.create_schema()

        print("\n[2/11] Chunk nodes")
        gb.insert_chunks(law_chunks, source="law", embeddings=embeddings if embeddings else None)
        gb.insert_chunks(note_chunks, source="notes", embeddings=embeddings if embeddings else None)

        print("\n[3/11] Document graph (year-aware)")
        gb.create_documents(law_chunks, note_chunks)

        print("\n[4/11] Entity nodes (source-aware)")
        gb.insert_entities(law_chunks, note_chunks)

        print("\n[5/11] MENTIONS (source-aware)")
        gb.create_mentions(law_chunks, note_chunks)

        print("\n[6/11] NEXT_CHUNK + CROSS_REFERENCES")
        gb.create_next(law_chunks)
        gb.create_next(note_chunks)
        gb.create_cross_references(law_chunks, note_chunks)

        print("\n[7/11] EXPLAINS (same-year notes -> law articles)")
        gb.create_explains(law_chunks, note_chunks)

        print("\n[8/11] RELATES_TO (same-year concept bridging)")
        gb.create_relates_to(law_chunks, note_chunks)

        print("\n[9/11] Topic/Object links")
        gb.create_topics_and_object_links(law_chunks, note_chunks)

        print("\n[10/11] PART_OF (structural hierarchy)")
        gb.create_part_of(law_chunks, note_chunks)

        print("\n[11/11] Embeddings + SIMILAR_TO")
        if embeddings:
            gb.create_vector_index(dimensions=len(next(iter(embeddings.values()))))
            gb.create_similar_to(threshold=0.78, max_neighbors=5)
        else:
            print("  (skipped)")

        gb.stats()
        print("\n✅  Graph build complete!")
    finally:
        gb.close()


if __name__ == "__main__":
    main()
