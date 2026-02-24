"""
Graph RAG agent — runs inside Modal container with GPU A100.

Provides an agentic pipeline with strong grounding controls:
  - multi-strategy graph retrieval
  - hybrid relevance scoring (LLM + lexical heuristics)
  - citation-constrained generation
  - Arabic-output enforcement
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
أنت مستشار قانوني تونسي دقيق.

أجب على السؤال اعتمادًا حصريًا على المراجع (S1..Sn) أدناه.

قواعد صارمة:
1) أجب باللغة العربية فقط.
2) لا تستخدم أي معلومة غير موجودة في المراجع.
3) كل فقرة يجب أن تتضمن إحالة مرجعية مثل [S1] أو [S2].
4) إذا كانت الأدلة غير كافية، صرّح بذلك بوضوح واطلب مرجعًا أدق.
5) لا تخمّن مواد قانونية أو سنوات أو أرقام فصول غير مذكورة في المراجع.

المراجع:
{context}

السؤال: {query}

الإجابة العربية الموثقة:"""

CITATION_REWRITE_PROMPT = """\
أعد صياغة الإجابة التالية باللغة العربية الفصحى فقط، مع إضافة إحالات [S#] في كل فقرة.
لا تضف أي معلومة غير موجودة في المراجع.

المراجع:
{context}

الإجابة الحالية:
{answer}

الإجابة النهائية:"""

ARABIC_REWRITE_PROMPT = """\
حوّل النص التالي إلى العربية الفصحى فقط مع الحفاظ على المعنى وعدم إضافة أي معلومات جديدة.

النص:
{answer}

النص العربي:"""

GROUNDING_CHECK_PROMPT = """\
تحقق من كون الإجابة مدعومة بالمراجع فقط.
أجب JSON فقط:
{{
  "grounded": true أو false,
  "unsupported_claims": عدد الادعاءات غير المدعومة
}}

المراجع:
{context}

الإجابة:
{answer}
"""

GROUNDED_REWRITE_PROMPT = """\
أعد كتابة الإجابة التالية بحيث تكون مدعومة حصريًا بالمراجع المعطاة، وبالعربية فقط،
ومع إحالات [S#] واضحة، ومن دون أي معلومة خارج المراجع.

المراجع:
{context}

الإجابة الحالية:
{answer}

الإجابة المصححة:"""


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

        # Bonus when chunk contains explicit legal reference requested by user.
        for ref in self._extract_reference_signals(query):
            if ref in text:
                score += 1.2

        return min(10.0, round(score, 2))

    @staticmethod
    def _has_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    @staticmethod
    def _is_mostly_arabic(text: str) -> bool:
        letters = re.findall(r"[A-Za-z\u0600-\u06FF]", text or "")
        if not letters:
            return True
        arabic = re.findall(r"[\u0600-\u06FF]", "".join(letters))
        return (len(arabic) / len(letters)) >= 0.75

    def _enforce_arabic(self, answer: str) -> str:
        if answer and self._is_mostly_arabic(answer) and not self._has_cjk(answer):
            return answer

        rewritten = self._llm(
            ARABIC_REWRITE_PROMPT.format(answer=answer),
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

        return list(dict.fromkeys(out))

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
        return list(dict.fromkeys(pairs))

    def _answer_mentions_required_refs(self, query: str, answer: str) -> bool:
        pairs = self._required_number_year_pairs(query)
        if not pairs:
            return True
        lowered_answer = (answer or "").lower()
        return all(f"{num} لسنة {year}" in lowered_answer for num, year in pairs)

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
                "لا تتوفر في قاعدة المعرفة الحالية نصوص قانونية كافية للإجابة بدقة على هذا السؤال. "
                "يرجى تحديد مرجع أدق (رقم فصل/قانون/مرسوم) أو تزويدي بالنص القانوني المرتبط بالسؤال."
            )

        refs = "، ".join(f"[S{s['index']}]" for s in sources[:3])
        return (
            "المعطيات المتاحة غير كافية لاستخلاص جواب قانوني موثوق دون الاستناد إلى افتراضات خارج النصوص. "
            f"المراجع المتوفرة حاليًا: {refs}. "
            "يرجى تزويدي بمرجع قانوني أدق لاستكمال الإجابة بشكل صحيح."
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
            path = c.get("path") or c.get("section_path", "غير محدد")
            label = "نص قانوني" if c.get("source") == "law" else "مذكرة توضيحية"
            txt = (c.get("text") or "")[:1400]
            context_parts.append(
                f"S{i} | {label} | المسار: {path} | الصلة: {c.get('relevance_score', 0)}\n{txt}"
            )
            sources.append({
                "index": i,
                "source_type": c.get("source", ""),
                "section_path": path,
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
            })

        context = "\n\n---\n\n".join(context_parts) if context_parts else "(لا توجد نصوص مرجعية)"
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

        def _add(rows: list, is_direct: bool = False):
            for r in rows:
                if r.get("id") and r["id"] not in seen:
                    seen[r["id"]] = r
                    if is_direct:
                        r["_direct_hit"] = True

        # 0) Direct article lookup
        art_pat = re.compile(r'(?:فصل|article|مادة|الفصل)\s*(\d+)', re.IGNORECASE)
        for src in state.get("search_entities", []) + [state["query"]]:
            for match in art_pat.finditer(src):
                _add(cypher(
                    "MATCH (c:Chunk {article_number: $num, source: 'law'}) "
                    "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                    "       c.chunk_type AS type, c.source AS source, "
                    "       c.page_start AS page_start, c.page_end AS page_end "
                    "LIMIT 3",
                    num=match.group(1),
                ), is_direct=True)

        # 1) Entity-based retrieval
        for ent in state.get("search_entities", [])[:6]:
            _add(cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
                "WHERE e.name CONTAINS $ent "
                "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                "       c.chunk_type AS type, c.source AS source, "
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
                "       node.page_start AS page_start, node.page_end AS page_end, "
                "       score",
                emb=qemb,
            ))
        except Exception:
            pass

        # 5) Graph expansion: EXPLAINS and RELATES_TO
        law_ids = [cid for cid, c in seen.items() if c.get("source") == "law"]
        for lid in law_ids[:5]:
            _add(cypher(
                "MATCH (law:Chunk {chunk_id: $id})<-[:EXPLAINS]-(note:Chunk) "
                "RETURN note.chunk_id AS id, note.text AS text, "
                "       note.section_path AS path, note.chunk_type AS type, "
                "       note.source AS source, "
                "       note.page_start AS page_start, note.page_end AS page_end",
                id=lid,
            ))

        for cid in list(seen.keys())[:8]:
            _add(cypher(
                "MATCH (c:Chunk {chunk_id: $id})-[:RELATES_TO]-(other:Chunk) "
                "RETURN other.chunk_id AS id, other.text AS text, "
                "       other.section_path AS path, other.chunk_type AS type, "
                "       other.source AS source, "
                "       other.page_start AS page_start, other.page_end AS page_end "
                "LIMIT 5",
                id=cid,
            ))

        # 6) Neighbor chunks (NEXT_CHUNK)
        for cid in list(seen.keys())[:5]:
            _add(cypher(
                "MATCH (c:Chunk {chunk_id: $id})-[:NEXT_CHUNK]->(nxt:Chunk) "
                "RETURN nxt.chunk_id AS id, nxt.text AS text, "
                "       nxt.section_path AS path, nxt.chunk_type AS type, "
                "       nxt.source AS source, "
                "       nxt.page_start AS page_start, nxt.page_end AS page_end "
                "UNION "
                "MATCH (prv:Chunk)-[:NEXT_CHUNK]->(c:Chunk {chunk_id: $id}) "
                "RETURN prv.chunk_id AS id, prv.text AS text, "
                "       prv.section_path AS path, prv.chunk_type AS type, "
                "       prv.source AS source, "
                "       prv.page_start AS page_start, prv.page_end AS page_end",
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
                "clarification_question": "يرجى تحديد الفصل أو القانون أو المرسوم المطلوب بدقة.",
            }

        # If the user requested an explicit legal reference that is not present
        # in retrieved evidence, force safe fallback to avoid wrong citation drift.
        if not self._references_supported(state["query"], chunks):
            return {
                "response": self._safe_fallback_answer(sources),
                "sources": sources,
                "evidence_strength": 0,
                "needs_clarification": True,
                "clarification_question": "لم أجد المرجع القانوني المذكور داخل النصوص الحالية. يرجى تزويدي بالنص المرتبط به.",
            }

        prompt = GENERATE_PROMPT.format(context=context, query=state["query"])
        answer = self._llm(prompt, max_tokens=1400, temperature=0.15)

        answer = self._ensure_source_citations(answer, context)
        answer = self._cleanup_answer(answer)
        answer = self._enforce_arabic(answer)

        grounded, _ = self._verify_grounding(answer, context)
        if not grounded:
            answer = self._llm(
                GROUNDED_REWRITE_PROMPT.format(context=context, answer=answer),
                max_tokens=1200,
                temperature=0.0,
            )
            answer = self._ensure_source_citations(answer, context)
            answer = self._cleanup_answer(answer)
            answer = self._enforce_arabic(answer)

            grounded, _ = self._verify_grounding(answer, context)
            if not grounded:
                answer = self._safe_fallback_answer(sources)

        if not self._answer_mentions_required_refs(state["query"], answer):
            return {
                "response": self._safe_fallback_answer(sources),
                "sources": sources,
                "evidence_strength": 0,
                "needs_clarification": True,
                "clarification_question": "الإجابة الحالية لا تغطي المرجع الرقمي المطلوب بدقة. يرجى تزويدي بالنص المرتبط به مباشرة.",
            }

        if self._has_cjk(answer):
            return {
                "response": self._safe_fallback_answer(sources),
                "sources": sources,
                "evidence_strength": 0,
                "needs_clarification": True,
                "clarification_question": "تم اكتشاف مخرجات لغوية غير عربية. يرجى إعادة المحاولة بصياغة أكثر تحديدًا.",
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
        non_arabic = not self._is_mostly_arabic(response)

        needs = weak or missing_sources or non_arabic

        clarification_question = ""
        if weak or missing_sources:
            clarification_question = "يرجى تحديد المرجع القانوني بدقة (رقم الفصل/القانون/المرسوم)."
        elif non_arabic:
            clarification_question = "يرجى إعادة صياغة السؤال بالعربية القانونية للتأكد من دقة الجواب."

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
            return {"error": "Agent not ready — models still loading"}

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

        return {
            "query": user_query,
            "response": result.get("response", ""),
            "sources": result.get("sources", []),
            "intent": result.get("intent", ""),
            "needs_clarification": result.get("needs_clarification", False),
            "clarification_question": result.get("clarification_question", ""),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_agent: GraphRAGAgent | None = None


def get_agent() -> GraphRAGAgent:
    global _agent
    if _agent is None:
        _agent = GraphRAGAgent()
    return _agent
