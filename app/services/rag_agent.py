"""
Graph RAG agent — runs inside Modal container with GPU A100.

Provides an agentic pipeline with strong grounding controls:
  - multi-strategy graph retrieval
  - hybrid relevance scoring (LLM + lexical heuristics)
  - citation-constrained generation
  - French-output enforcement
  - safe fallback on weak evidence
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List, Literal, TypedDict

from app.config import load_environment
from app.services.neo4j_service import cypher

load_environment()

log = logging.getLogger(__name__)

# Model IDs — can be overridden by env vars
MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "intfloat/multilingual-e5-base")
MODELS_CACHE = os.environ.get("MODELS_CACHE_DIR", "/models")

# Quality knobs
MIN_RELEVANCE_SCORE = float(os.environ.get("RAG_MIN_RELEVANCE_SCORE", "6.5"))
MIN_EVIDENCE_STRENGTH = int(os.environ.get("RAG_MIN_EVIDENCE_STRENGTH", "2"))
MAX_CONTEXT_CHUNKS = int(os.environ.get("RAG_MAX_CONTEXT_CHUNKS", "8"))


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    intent: str
    search_entities: List[str]
    search_concepts: List[str]
    search_keywords: List[str]
    retrieved_chunks: List[dict]
    relevant_chunks: List[dict]
    response: str
    iteration: int
    needs_clarification: bool
    clarification_question: str
    sources: List[dict]
    evidence_strength: int


# ─── Prompt templates ─────────────────────────────────────────────────────────

ANALYZE_PROMPT = """\
أنت محلل استعلامات قانونية متخصص في القانون التونسي والنظام الجبائي.

حلل السؤال التالي واستخرج المعلومات التالية بصيغة JSON فقط (بدون أي نص إضافي):

{{
  "intent": "وصف موجز لنية السؤال",
  "entities": ["أرقام الفصول أو القوانين أو المراسيم المذكورة"],
  "concepts": ["المفاهيم الضريبية أو القانونية المعنية"],
  "keywords": ["كلمات البحث الأساسية"]
}}

السؤال: {query}

أجب بصيغة JSON فقط:"""

BATCH_EVAL_PROMPT = """\
أنت مقيّم صلة النصوص القانونية.

لكل نص أدناه، قيّم الصلة بالسؤال من 0 إلى 10.
أجب بمصفوفة JSON فقط:
[{{"index": 1, "score": X, "relevant": true/false}}, ...]

السؤال: {query}

النصوص:
{chunks}

JSON فقط:"""

GENERATE_PROMPT = """\
Vous etes un conseiller juridique tunisien precis.

Repondez a la question en vous basant UNIQUEMENT sur les references (S1..Sn) ci-dessous.

Regles strictes:
1) Repondez uniquement en francais.
2) N'ajoutez aucune information absente des references.
3) Chaque paragraphe doit contenir au moins une citation [S#].
4) Si les preuves sont insuffisantes, dites-le explicitement et demandez une reference plus precise.
5) N'inventez pas des articles, annees ou numeros de lois.

References:
{context}

Question: {query}

Reponse juridique en francais:"""

CITATION_REWRITE_PROMPT = """\
Reecrivez la reponse suivante en francais seulement, avec des citations [S#] dans chaque paragraphe.
N'ajoutez aucune information en dehors des references.

References:
{context}

Reponse actuelle:
{answer}

Reponse finale:"""

FRENCH_REWRITE_PROMPT = """\
Transformez le texte suivant en francais clair, sans changer le sens et sans ajouter de nouvelles informations.

Texte:
{answer}

Texte francais:"""

GROUNDING_CHECK_PROMPT = """\
Verifiez que la reponse est supportee uniquement par les references.
Repondez en JSON uniquement:
{{
  "grounded": true or false,
  "unsupported_claims": integer
}}

References:
{context}

Reponse:
{answer}
"""

GROUNDED_REWRITE_PROMPT = """\
Reecrivez la reponse suivante pour qu'elle soit strictement fondee sur les references, en francais,
avec des citations [S#], et sans information externe.

References:
{context}

Reponse actuelle:
{answer}

Reponse corrigee:"""


# ─── RAG Agent class ──────────────────────────────────────────────────────────

class GraphRAGAgent:
    """
    Local Graph RAG agent using LangGraph + Qwen + Neo4j.

    Call `startup()` once at app boot (lifespan), then use `query()`.
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.embed_model = None
        self.agent = None
        self._ready = False

    # ── Lifecycle ──────────────────────────────────────────────────

    def startup(self) -> None:
        """Load LLM + embedding model and compile the LangGraph agent."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sentence_transformers import SentenceTransformer

        log.info("Loading LLM: %s (cache: %s)", MODEL_ID, MODELS_CACHE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, cache_dir=f"{MODELS_CACHE}/qwen", trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=f"{MODELS_CACHE}/qwen",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        log.info("LLM ready")

        log.info("Loading embed model: %s", EMBED_MODEL_ID)
        self.embed_model = SentenceTransformer(
            EMBED_MODEL_ID, cache_folder=f"{MODELS_CACHE}/embed",
        )
        log.info("Embed model ready")

        self.agent = self._build_agent()
        self._ready = True
        log.info("Agent compiled — ready to serve")

    def shutdown(self) -> None:
        del self.model, self.tokenizer, self.embed_model
        self.model = self.tokenizer = self.embed_model = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── LLM helpers ────────────────────────────────────────────────

    def _llm(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.85,
                repetition_penalty=1.1,
            )
        return self.tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True,
        ).strip()

    def _embed(self, text: str) -> list:
        return self.embed_model.encode(f"query: {text}").tolist()

    @staticmethod
    def _parse_json(raw: str) -> Any:
        for pattern in (r'\[.*\]', r'\{.*\}'):
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    continue
        return {}

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        stop = {
            "ما", "ماذا", "كيف", "هل", "في", "من", "على", "الى", "إلى", "عن", "مع",
            "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "ماهو", "ماهي",
            "the", "what", "how", "is", "are", "in", "of", "to",
        }
        tokens = re.findall(r"[\u0600-\u06FFA-Za-z0-9]{2,}", text.lower())
        return {t for t in tokens if t not in stop}

    def _heuristic_relevance(self, query: str, chunk: dict) -> float:
        q_terms = self._query_terms(query)
        q_years = set(self._extract_years(query))
        if not q_terms:
            return 0.0

        text = (chunk.get("text") or "")[:1400].lower()
        if not text:
            return 0.0

        hits = sum(1 for t in q_terms if t in text)
        coverage = hits / max(1, len(q_terms))
        score = coverage * 10.0

        if chunk.get("_direct_hit"):
            score = max(score, 9.5)

        if chunk.get("source") == "law":
            score += 0.3

        if q_years:
            cy = str(chunk.get("document_year") or "")
            if cy and cy in q_years:
                score += 1.0
            elif cy:
                score -= 0.8

        if chunk.get("_topic_overlap"):
            score += min(1.2, 0.4 * float(chunk.get("_topic_overlap", 0)))

        # Bonus when chunk contains explicit legal reference requested by user.
        for ref in self._extract_reference_signals(query):
            if ref in text:
                score += 1.2

        return min(10.0, round(score, 2))

    @staticmethod
    def _has_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    @staticmethod
    def _is_mostly_french(text: str) -> bool:
        letters = re.findall(r"[A-Za-z\u00C0-\u017F\u0600-\u06FF]", text or "")
        if not letters:
            return True
        joined = "".join(letters)
        latin = re.findall(r"[A-Za-z\u00C0-\u017F]", joined)
        arabic = re.findall(r"[\u0600-\u06FF]", joined)
        latin_ratio = len(latin) / len(letters)
        return latin_ratio >= 0.45 and len(latin) >= len(arabic)

    def _enforce_french(self, answer: str) -> str:
        if answer and self._is_mostly_french(answer) and not self._has_cjk(answer):
            return answer

        rewritten = self._llm(
            FRENCH_REWRITE_PROMPT.format(answer=answer),
            max_tokens=900,
            temperature=0.0,
        )
        candidate = rewritten.strip() or answer
        # Hard strip for occasional foreign script leakage.
        candidate = re.sub(r"[\u4e00-\u9fff]+", " ", candidate)
        candidate = re.sub(r"\s{2,}", " ", candidate).strip()
        return candidate

    @staticmethod
    def _has_source_citations(answer: str) -> bool:
        return bool(re.search(r"\[S\d+\]", answer or ""))

    @staticmethod
    def _cleanup_answer(answer: str) -> str:
        cleaned = answer or ""
        cleaned = cleaned.replace("\\[", "").replace("\\]", "")
        cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _extract_reference_signals(text: str) -> list[str]:
        out: list[str] = []
        lowered = (text or "").lower()

        for m in re.finditer(r"(?:المرسوم|مرسوم)\s+عدد\s+(\d+)\s+لسنة\s+(\d{4})", lowered):
            out.append(f"مرسوم عدد {m.group(1)} لسنة {m.group(2)}")
            out.append(f"{m.group(1)} لسنة {m.group(2)}")

        for m in re.finditer(r"(?:القانون|قانون)\s+عدد\s+(\d+)\s+لسنة\s+(\d{4})", lowered):
            out.append(f"قانون عدد {m.group(1)} لسنة {m.group(2)}")
            out.append(f"{m.group(1)} لسنة {m.group(2)}")

        for m in re.finditer(r"(?:الفصل|فصل)\s+(\d+)", lowered):
            out.append(f"الفصل {m.group(1)}")
            out.append(f"فصل {m.group(1)}")

        for m in re.finditer(r"(?:loi|decret|décret)\s+n?[°o]?\s*(\d+)\s*(?:de|/)\s*(\d{4})", lowered):
            out.append(f"{m.group(1)} de {m.group(2)}")
            out.append(f"{m.group(1)}/{m.group(2)}")

        for m in re.finditer(r"article\s+(\d+)", lowered):
            out.append(f"article {m.group(1)}")

        return list(dict.fromkeys(out))

    @staticmethod
    def _extract_years(text: str) -> list[str]:
        years = re.findall(r"(19\d{2}|20\d{2})", text or "")
        return list(dict.fromkeys(years))

    def _references_supported(self, query: str, chunks: list[dict]) -> bool:
        refs = self._extract_reference_signals(query)
        if not refs:
            return True

        joined = " ".join((c.get("text") or "").lower() for c in chunks[:MAX_CONTEXT_CHUNKS])
        return any(ref in joined for ref in refs)

    @staticmethod
    def _required_number_year_pairs(text: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        lowered = (text or "").lower()
        for m in re.finditer(r"(?:المرسوم|مرسوم|القانون|قانون)\s+عدد\s+(\d+)\s+لسنة\s+(\d{4})", lowered):
            pairs.append((m.group(1), m.group(2)))
        for m in re.finditer(r"(?:loi|decret|décret)\s+n?[°o]?\s*(\d+)\s*(?:de|/)\s*(\d{4})", lowered):
            pairs.append((m.group(1), m.group(2)))
        return list(dict.fromkeys(pairs))

    def _answer_mentions_required_refs(self, query: str, answer: str) -> bool:
        pairs = self._required_number_year_pairs(query)
        if not pairs:
            return True
        lowered_answer = (answer or "").lower()
        for num, year in pairs:
            patterns = [
                f"{num} لسنة {year}",
                f"{num} de {year}",
                f"n {num} de {year}",
                f"{num}/{year}",
            ]
            if not any(p in lowered_answer for p in patterns):
                return False
        return True

    def _ensure_source_citations(self, answer: str, context: str) -> str:
        if self._has_source_citations(answer):
            return answer

        rewritten = self._llm(
            CITATION_REWRITE_PROMPT.format(context=context, answer=answer),
            max_tokens=1000,
            temperature=0.0,
        )
        return rewritten.strip() or answer

    def _verify_grounding(self, answer: str, context: str) -> tuple[bool, int]:
        raw = self._llm(
            GROUNDING_CHECK_PROMPT.format(context=context, answer=answer),
            max_tokens=200,
            temperature=0.0,
        )
        parsed = self._parse_json(raw)
        if isinstance(parsed, dict):
            unsupported = int(parsed.get("unsupported_claims", 0) or 0)
            grounded = bool(parsed.get("grounded", unsupported == 0))
            return grounded, unsupported
        # If verifier parsing fails, keep conservative behavior.
        return False, 1

    @staticmethod
    def _safe_fallback_answer(sources: list[dict]) -> str:
        if not sources:
            return (
                "La base de connaissances actuelle ne contient pas assez de references juridiques "
                "pour repondre de facon fiable a cette question. "
                "Veuillez preciser la reference (article/loi/decret) ou fournir le texte legal concerne."
            )

        refs = "، ".join(f"[S{s['index']}]" for s in sources[:3])
        return (
            "Les elements disponibles sont insuffisants pour produire une reponse juridique fiable "
            "sans extrapolation hors references. "
            f"References actuellement disponibles: {refs}. "
            "Veuillez fournir une reference legale plus precise."
        )

    @staticmethod
    def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        for c in chunks:
            cid = c.get("id") or ""
            key = cid or f"{c.get('source','')}|{(c.get('text') or '')[:120]}"
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _build_context(self, chunks: list[dict]) -> tuple[str, list[dict]]:
        context_parts: list[str] = []
        sources: list[dict] = []

        for i, c in enumerate(chunks, 1):
            path = c.get("path") or c.get("section_path", "non specifie")
            label = "Texte de loi" if c.get("source") == "law" else "Note explicative"
            txt = (c.get("text") or "")[:1400]
            context_parts.append(
                f"S{i} | {label} | chemin: {path} | pertinence: {c.get('relevance_score', 0)}\n{txt}"
            )
            sources.append({
                "index": i,
                "source_type": c.get("source", ""),
                "section_path": path,
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
            })

        context = "\n\n---\n\n".join(context_parts) if context_parts else "(aucune reference disponible)"
        return context, sources

    # ── LangGraph nodes ────────────────────────────────────────────

    def _node_analyze(self, state: AgentState) -> dict:
        prompt = ANALYZE_PROMPT.format(query=state["query"])
        raw = self._llm(prompt, max_tokens=400, temperature=0.05)
        parsed_raw = self._parse_json(raw)
        parsed = parsed_raw if isinstance(parsed_raw, dict) else {}

        keywords = parsed.get("keywords", state["query"].split())
        if not isinstance(keywords, list):
            keywords = state["query"].split()

        return {
            "intent": parsed.get("intent", state["query"]),
            "search_entities": parsed.get("entities", []),
            "search_concepts": parsed.get("concepts", []),
            "search_keywords": keywords,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _node_retrieve(self, state: AgentState) -> dict:
        """Multi-strategy retrieval from Neo4j knowledge graph."""
        seen: dict[str, dict] = {}
        query_years = self._extract_years(state["query"])

        def _add(rows: list, is_direct: bool = False):
            for r in rows:
                if r.get("id") and r["id"] not in seen:
                    seen[r["id"]] = r
                    if is_direct:
                        r["_direct_hit"] = True
                elif r.get("id") and r["id"] in seen and r.get("topic_overlap"):
                    seen[r["id"]]["_topic_overlap"] = max(
                        float(seen[r["id"]].get("_topic_overlap", 0)),
                        float(r.get("topic_overlap", 0)),
                    )

        # 0) Direct article lookup
        art_pat = re.compile(r'(?:فصل|article|مادة|الفصل)\s*(\d+)', re.IGNORECASE)
        for src in state.get("search_entities", []) + [state["query"]]:
            for match in art_pat.finditer(src):
                _add(cypher(
                    "MATCH (c:Chunk {article_number: $num, source: 'law'}) "
                    "WHERE (size($years) = 0 OR c.document_year IN $years) "
                    "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                    "       c.chunk_type AS type, c.source AS source, "
                    "       c.document_year AS document_year, "
                    "       c.page_start AS page_start, c.page_end AS page_end "
                    "LIMIT 3",
                    num=match.group(1),
                    years=query_years,
                ), is_direct=True)

        # 1) Entity-based retrieval
        for ent in state.get("search_entities", [])[:6]:
            _add(cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
                "WHERE e.name CONTAINS $ent "
                "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                "       c.chunk_type AS type, c.source AS source, "
                "       c.document_year AS document_year, "
                "       c.page_start AS page_start, c.page_end AS page_end "
                "LIMIT 6",
                ent=ent,
            ))

        # 2) Concept-based retrieval
        for concept in state.get("search_concepts", [])[:6]:
            _add(cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
                "WHERE e.name CONTAINS $concept "
                "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                "       c.chunk_type AS type, c.source AS source, "
                "       c.document_year AS document_year, "
                "       c.page_start AS page_start, c.page_end AS page_end "
                "LIMIT 6",
                concept=concept,
            ))

        # 3) Full-text keyword search
        kw_terms = [str(k).strip() for k in state.get("search_keywords", []) if str(k).strip()]
        kw_terms = [k for k in kw_terms if len(k) >= 2][:8]
        kw_text = " OR ".join(kw_terms)
        if kw_text:
            try:
                _add(cypher(
                    "CALL db.index.fulltext.queryNodes('chunk_text_ft', $kw) "
                    "YIELD node, score "
                    "RETURN node.chunk_id AS id, node.text AS text, "
                    "       node.section_path AS path, node.chunk_type AS type, "
                    "       node.source AS source, "
                    "       node.document_year AS document_year, "
                    "       node.page_start AS page_start, node.page_end AS page_end, "
                    "       score "
                    "ORDER BY score DESC LIMIT 12",
                    kw=kw_text,
                ))
            except Exception:
                pass

        # 4) Vector similarity search
        try:
            qemb = self._embed(state["query"])
            _add(cypher(
                "CALL db.index.vector.queryNodes('chunk_embeddings', 12, $emb) "
                "YIELD node, score "
                "RETURN node.chunk_id AS id, node.text AS text, "
                "       node.section_path AS path, node.chunk_type AS type, "
                "       node.source AS source, "
                "       node.document_year AS document_year, "
                "       node.page_start AS page_start, node.page_end AS page_end, "
                "       score",
                emb=qemb,
            ))
        except Exception:
            pass

        # 5) Year-focused retrieval
        for y in query_years[:3]:
            _add(cypher(
                "MATCH (c:Chunk) "
                "WHERE c.document_year = $y "
                "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                "       c.chunk_type AS type, c.source AS source, "
                "       c.document_year AS document_year, "
                "       c.page_start AS page_start, c.page_end AS page_end "
                "LIMIT 8",
                y=y,
            ))

        # 6) Graph expansion: EXPLAINS, ABOUT_ARTICLE, RELATES_TO
        law_ids = [cid for cid, c in seen.items() if c.get("source") == "law"]
        for lid in law_ids[:5]:
            _add(cypher(
                "MATCH (law:Chunk {chunk_id: $id})<-[:EXPLAINS]-(note:Chunk) "
                "RETURN note.chunk_id AS id, note.text AS text, "
                "       note.section_path AS path, note.chunk_type AS type, "
                "       note.source AS source, "
                "       note.document_year AS document_year, "
                "       note.page_start AS page_start, note.page_end AS page_end",
                id=lid,
            ))
            _add(cypher(
                "MATCH (note:Chunk)-[r:ABOUT_ARTICLE]->(law:Chunk {chunk_id: $id}) "
                "RETURN note.chunk_id AS id, note.text AS text, "
                "       note.section_path AS path, note.chunk_type AS type, "
                "       note.source AS source, "
                "       note.document_year AS document_year, "
                "       note.page_start AS page_start, note.page_end AS page_end, "
                "       r.topic_overlap AS topic_overlap "
                "LIMIT 6",
                id=lid,
            ))

        note_ids = [cid for cid, c in seen.items() if c.get("source") == "notes"]
        for nid in note_ids[:5]:
            _add(cypher(
                "MATCH (note:Chunk {chunk_id: $id})-[r:ABOUT_ARTICLE]->(law:Chunk) "
                "RETURN law.chunk_id AS id, law.text AS text, "
                "       law.section_path AS path, law.chunk_type AS type, "
                "       law.source AS source, "
                "       law.document_year AS document_year, "
                "       law.page_start AS page_start, law.page_end AS page_end, "
                "       r.topic_overlap AS topic_overlap "
                "LIMIT 6",
                id=nid,
            ))

        for cid in list(seen.keys())[:8]:
            _add(cypher(
                "MATCH (c:Chunk {chunk_id: $id})-[:RELATES_TO]-(other:Chunk) "
                "RETURN other.chunk_id AS id, other.text AS text, "
                "       other.section_path AS path, other.chunk_type AS type, "
                "       other.source AS source, "
                "       other.document_year AS document_year, "
                "       other.page_start AS page_start, other.page_end AS page_end "
                "LIMIT 5",
                id=cid,
            ))

        # 7) Neighbor chunks (NEXT_CHUNK)
        for cid in list(seen.keys())[:5]:
            _add(cypher(
                "MATCH (c:Chunk {chunk_id: $id})-[:NEXT_CHUNK]->(nxt:Chunk) "
                "RETURN nxt.chunk_id AS id, nxt.text AS text, "
                "       nxt.section_path AS path, nxt.chunk_type AS type, "
                "       nxt.source AS source, "
                "       nxt.document_year AS document_year, "
                "       nxt.page_start AS page_start, nxt.page_end AS page_end "
                "UNION "
                "MATCH (prv:Chunk)-[:NEXT_CHUNK]->(c:Chunk {chunk_id: $id}) "
                "RETURN prv.chunk_id AS id, prv.text AS text, "
                "       prv.section_path AS path, prv.chunk_type AS type, "
                "       prv.source AS source, "
                "       prv.document_year AS document_year, "
                "       prv.page_start AS page_start, prv.page_end AS page_end",
                id=cid,
            ))

        # 8) Topic two-hop + same-year document expansion
        for cid in list(seen.keys())[:6]:
            _add(cypher(
                "MATCH (c:Chunk {chunk_id: $id})-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(other:Chunk) "
                "WHERE other.chunk_id <> $id "
                "RETURN other.chunk_id AS id, other.text AS text, "
                "       other.section_path AS path, other.chunk_type AS type, "
                "       other.source AS source, other.document_year AS document_year, "
                "       other.page_start AS page_start, other.page_end AS page_end, "
                "       1 AS topic_overlap "
                "LIMIT 5",
                id=cid,
            ))
            _add(cypher(
                "MATCH (c:Chunk {chunk_id: $id})-[:IN_DOCUMENT]->(d:Document) "
                "MATCH (d)-[:SAME_YEAR_AS]-(peer:Document) "
                "MATCH (other:Chunk)-[:IN_DOCUMENT]->(peer) "
                "RETURN other.chunk_id AS id, other.text AS text, "
                "       other.section_path AS path, other.chunk_type AS type, "
                "       other.source AS source, other.document_year AS document_year, "
                "       other.page_start AS page_start, other.page_end AS page_end "
                "LIMIT 5",
                id=cid,
            ))

        return {"retrieved_chunks": list(seen.values())}

    def _node_evaluate(self, state: AgentState) -> dict:
        retrieved = state.get("retrieved_chunks", [])
        if not retrieved:
            return {"relevant_chunks": []}

        direct_hits = [c for c in retrieved if c.get("_direct_hit")]
        for dh in direct_hits:
            dh["relevance_score"] = 10.0

        candidates = [
            c for c in retrieved
            if len(c.get("text", "")) >= 40 and not c.get("_direct_hit")
        ]
        if not candidates:
            return {"relevant_chunks": direct_hits[:MAX_CONTEXT_CHUNKS]}

        # Heuristic relevance
        for c in candidates:
            c["_heuristic_score"] = self._heuristic_relevance(state["query"], c)

        # LLM-based relevance
        numbered: list[str] = []
        for i, c in enumerate(candidates[:24], 1):
            path = c.get("path") or c.get("section_path", "")
            txt = (c.get("text") or "")[:320]
            numbered.append(f"[{i}] ({path}): {txt}")

        llm_scores: dict[int, float] = {}
        llm_relevant: dict[int, bool] = {}

        if numbered:
            prompt = BATCH_EVAL_PROMPT.format(
                query=state["query"],
                chunks="\n\n".join(numbered),
            )
            raw = self._llm(prompt, max_tokens=650, temperature=0.05)
            parsed = self._parse_json(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    idx = int(item.get("index", 0)) - 1
                    if 0 <= idx < len(candidates[:24]):
                        try:
                            llm_scores[idx] = float(item.get("score", 0) or 0)
                        except (TypeError, ValueError):
                            llm_scores[idx] = 0.0
                        llm_relevant[idx] = bool(item.get("relevant", False))

        ranked: list[dict] = []
        for idx, c in enumerate(candidates):
            llm_score = llm_scores.get(idx, 0.0)
            combined = max(float(c.get("_heuristic_score", 0.0)), llm_score)
            if llm_relevant.get(idx, False):
                combined = max(combined, MIN_RELEVANCE_SCORE)

            if c.get("source") == "law":
                combined += 0.2

            c["relevance_score"] = round(min(10.0, combined), 2)
            if c["relevance_score"] >= (MIN_RELEVANCE_SCORE - 0.5):
                ranked.append(c)

        if not ranked:
            fallback_ranked = sorted(
                candidates,
                key=lambda x: x.get("_heuristic_score", 0),
                reverse=True,
            )[:3]
            for c in fallback_ranked:
                c["relevance_score"] = max(4.0, float(c.get("_heuristic_score", 0)))
            ranked = fallback_ranked

        ranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        final = self._dedupe_chunks(direct_hits + ranked)
        final = sorted(final, key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Source diversity (law + notes) where possible.
        if final:
            for source_name in ("law", "notes"):
                if not any(c.get("source") == source_name for c in final):
                    repl = next((c for c in ranked if c.get("source") == source_name), None)
                    if repl:
                        if len(final) < MAX_CONTEXT_CHUNKS:
                            final.append(repl)
                        else:
                            final[-1] = repl

        final = self._dedupe_chunks(final)
        final = sorted(final, key=lambda x: x.get("relevance_score", 0), reverse=True)
        return {"relevant_chunks": final[:MAX_CONTEXT_CHUNKS]}

    def _node_generate(self, state: AgentState) -> dict:
        chunks = sorted(
            state.get("relevant_chunks", []),
            key=lambda x: x.get("relevance_score", 0),
            reverse=True,
        )[:MAX_CONTEXT_CHUNKS]

        context, sources = self._build_context(chunks)

        strong_chunks = [c for c in chunks if c.get("relevance_score", 0) >= MIN_RELEVANCE_SCORE]
        direct_hits = [c for c in chunks if c.get("_direct_hit")]
        evidence_strength = len(strong_chunks) + (2 * len(direct_hits))

        # Hard guard: weak evidence -> safe fallback, no hallucination.
        if not chunks or evidence_strength < MIN_EVIDENCE_STRENGTH:
            return {
                "response": self._safe_fallback_answer(sources),
                "sources": sources,
                "evidence_strength": evidence_strength,
                "needs_clarification": True,
                "clarification_question": "Veuillez preciser l'article, la loi ou le decret exact.",
            }

        # If the user requested an explicit legal reference that is not present
        # in retrieved evidence, force safe fallback to avoid wrong citation drift.
        if not self._references_supported(state["query"], chunks):
            return {
                "response": self._safe_fallback_answer(sources),
                "sources": sources,
                "evidence_strength": 0,
                "needs_clarification": True,
                "clarification_question": "La reference demandee n'a pas ete trouvee dans le corpus actuel. Merci de fournir le texte associe.",
            }

        prompt = GENERATE_PROMPT.format(context=context, query=state["query"])
        answer = self._llm(prompt, max_tokens=1400, temperature=0.15)

        answer = self._ensure_source_citations(answer, context)
        answer = self._cleanup_answer(answer)
        answer = self._enforce_french(answer)

        grounded, _ = self._verify_grounding(answer, context)
        if not grounded:
            answer = self._llm(
                GROUNDED_REWRITE_PROMPT.format(context=context, answer=answer),
                max_tokens=1200,
                temperature=0.0,
            )
            answer = self._ensure_source_citations(answer, context)
            answer = self._cleanup_answer(answer)
            answer = self._enforce_french(answer)

            grounded, _ = self._verify_grounding(answer, context)
            if not grounded:
                answer = self._safe_fallback_answer(sources)

        if not self._answer_mentions_required_refs(state["query"], answer):
            return {
                "response": self._safe_fallback_answer(sources),
                "sources": sources,
                "evidence_strength": 0,
                "needs_clarification": True,
                "clarification_question": "La reponse ne couvre pas correctement la reference numerique demandee. Merci de fournir le texte exact.",
            }

        if self._has_cjk(answer):
            return {
                "response": self._safe_fallback_answer(sources),
                "sources": sources,
                "evidence_strength": 0,
                "needs_clarification": True,
                "clarification_question": "Une sortie linguistique invalide a ete detectee. Veuillez reformuler la question de facon plus precise.",
            }

        return {
            "response": answer,
            "sources": sources,
            "evidence_strength": evidence_strength,
        }

    def _node_check(self, state: AgentState) -> dict:
        response = state.get("response", "")
        sources = state.get("sources", [])
        evidence_strength = state.get("evidence_strength", 0)

        weak = evidence_strength < MIN_EVIDENCE_STRENGTH
        missing_sources = len(sources) == 0
        non_french = not self._is_mostly_french(response)

        needs = weak or missing_sources or non_french

        clarification_question = ""
        if weak or missing_sources:
            clarification_question = "Veuillez preciser la reference legale (article/loi/decret)."
        elif non_french:
            clarification_question = "Veuillez reformuler votre question; la reponse doit etre produite en francais."

        return {
            "needs_clarification": needs,
            "clarification_question": clarification_question,
        }

    def _route_after_eval(self, state: AgentState) -> Literal["retry", "generate"]:
        if len(state.get("relevant_chunks", [])) < 1 and state.get("iteration", 0) < 3:
            return "retry"
        return "generate"

    # ── Build LangGraph ────────────────────────────────────────────

    def _build_agent(self):
        from langgraph.graph import StateGraph, START, END

        g = StateGraph(AgentState)
        g.add_node("analyze", self._node_analyze)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("evaluate", self._node_evaluate)
        g.add_node("generate", self._node_generate)
        g.add_node("check", self._node_check)

        g.add_edge(START, "analyze")
        g.add_edge("analyze", "retrieve")
        g.add_edge("retrieve", "evaluate")
        g.add_conditional_edges(
            "evaluate",
            self._route_after_eval,
            {"retry": "analyze", "generate": "generate"},
        )
        g.add_edge("generate", "check")
        g.add_edge("check", END)

        return g.compile()

    # ── Public API ─────────────────────────────────────────────────

    def query(self, user_query: str) -> dict:
        """Run the full agentic Graph RAG pipeline."""
        if not self._ready:
            return {"error": "Agent indisponible pour le moment — les modeles sont en cours de chargement."}

        initial: AgentState = {
            "query": user_query,
            "intent": "",
            "search_entities": [],
            "search_concepts": [],
            "search_keywords": [],
            "retrieved_chunks": [],
            "relevant_chunks": [],
            "response": "",
            "iteration": 0,
            "needs_clarification": False,
            "clarification_question": "",
            "sources": [],
            "evidence_strength": 0,
        }

        result = self.agent.invoke(initial)

        response = self._cleanup_answer(result.get("response", ""))
        if response:
            response = self._enforce_french(response)

        clarification_question = self._cleanup_answer(result.get("clarification_question", ""))
        if clarification_question:
            clarification_question = self._enforce_french(clarification_question)

        return {
            "query": user_query,
            "response": response,
            "sources": result.get("sources", []),
            "intent": result.get("intent", ""),
            "needs_clarification": result.get("needs_clarification", False),
            "clarification_question": clarification_question,
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_agent: GraphRAGAgent | None = None


def get_agent() -> GraphRAGAgent:
    global _agent
    if _agent is None:
        _agent = GraphRAGAgent()
    return _agent
