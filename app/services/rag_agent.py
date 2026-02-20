"""
Graph RAG agent — runs inside Modal container with GPU A100.

Provides the same agentic LangGraph pipeline as modal_agent.py,
but served via FastAPI routes instead of Modal web_endpoint.

Models are cached on the Modal volume mounted at /models.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List, Literal, TypedDict

from app.services.neo4j_service import cypher

log = logging.getLogger(__name__)

# Model IDs — can be overridden by env vars
MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "intfloat/multilingual-e5-base")
MODELS_CACHE = os.environ.get("MODELS_CACHE_DIR", "/models")  # Modal volume mount

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

لكل نص من النصوص أدناه، قيّم مدى صلته بالسؤال من 0 إلى 10.
أجب بمصفوفة JSON فقط:
[{{"index": 1, "score": X, "relevant": true/false}}, ...]

السؤال: {query}

النصوص:
{chunks}

أجب بمصفوفة JSON فقط:"""

GENERATE_PROMPT = """\
أنت مستشار قانوني متخصص في القانون التونسي والنظام الجبائي.
أجب على سؤال المستخدم بناءً على النصوص القانونية المقدمة أدناه.

⚠ تعليمات صارمة:
- أجب حصريًا باللغة العربية.

قواعد الإجابة:
1. أجب باللغة العربية بشكل مباشر ودقيق وعملي.
2. استند فقط إلى النصوص القانونية المقدمة.
3. اذكر أرقام الفصول والقوانين والمراسيم ذات الصلة.
4. قدم إجابة قابلة للتطبيق فوراً.
5. إذا كانت المعلومات غير كافية، اذكر ذلك بوضوح.
6. رتّب الإجابة بحسب الأهمية.
7. ميّز بوضوح بين أحكام قانون المالية وأحكام المراسيم والمجلات الأخرى.

النصوص القانونية المرجعية:
{context}

السؤال: {query}

الإجابة:"""

CHECK_PROMPT = """\
أنت مراجع جودة للاستشارات القانونية. قيّم الإجابة وأجب بصيغة JSON فقط:

{{
  "is_complete": true أو false,
  "needs_clarification": true أو false,
  "clarification_question": "سؤال توضيحي إن لزم الأمر أو null",
  "quality_score": رقم من 1 إلى 10
}}

السؤال: {query}
الإجابة: {response}

JSON فقط:"""


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
        """Load LLM + embedding model and compile the LangGraph agent.
        Models are cached on the Modal volume at /models."""
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

    # ── LLM helper ─────────────────────────────────────────────────

    def _llm(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.3) -> str:
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
                top_p=0.9,
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

    # ── LangGraph nodes ────────────────────────────────────────────

    def _node_analyze(self, state: AgentState) -> dict:
        prompt = ANALYZE_PROMPT.format(query=state["query"])
        raw = self._llm(prompt, max_tokens=512, temperature=0.1)
        parsed = self._parse_json(raw) if isinstance(self._parse_json(raw), dict) else {}
        return {
            "intent":          parsed.get("intent", state["query"]),
            "search_entities": parsed.get("entities", []),
            "search_concepts": parsed.get("concepts", []),
            "search_keywords": parsed.get("keywords", state["query"].split()),
            "iteration":       state.get("iteration", 0) + 1,
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
        _art_pat = re.compile(r'(?:فصل|article|مادة|الفصل)\s*(\d+)', re.IGNORECASE)
        for src in state.get("search_entities", []) + [state["query"]]:
            for _m in _art_pat.finditer(src):
                _add(cypher(
                    "MATCH (c:Chunk {article_number: $num, source: 'law'}) "
                    "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                    "       c.chunk_type AS type, c.source AS source, "
                    "       c.page_start AS page_start, c.page_end AS page_end "
                    "LIMIT 3",
                    num=_m.group(1),
                ), is_direct=True)

        # 1) Entity-based retrieval
        for ent in state.get("search_entities", []):
            _add(cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
                "WHERE e.name CONTAINS $ent "
                "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                "       c.chunk_type AS type, c.source AS source, "
                "       c.page_start AS page_start, c.page_end AS page_end "
                "LIMIT 5",
                ent=ent,
            ))

        # 2) Concept-based retrieval
        for concept in state.get("search_concepts", []):
            _add(cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
                "WHERE e.name CONTAINS $concept "
                "RETURN c.chunk_id AS id, c.text AS text, c.section_path AS path, "
                "       c.chunk_type AS type, c.source AS source, "
                "       c.page_start AS page_start, c.page_end AS page_end "
                "LIMIT 5",
                concept=concept,
            ))

        # 3) Full-text keyword search
        kw_text = " OR ".join(state.get("search_keywords", []))
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
                    "ORDER BY score DESC LIMIT 10",
                    kw=kw_text,
                ))
            except Exception:
                pass

        # 4) Vector similarity search
        try:
            qemb = self._embed(state["query"])
            _add(cypher(
                "CALL db.index.vector.queryNodes('chunk_embeddings', 10, $emb) "
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

        # 5) Graph expansion: EXPLAINS, RELATES_TO, SIMILAR_TO, PART_OF
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
            dh["relevance_score"] = 10

        candidates = [c for c in retrieved
                      if len(c.get("text", "")) >= 15 and not c.get("_direct_hit")]
        if not candidates and direct_hits:
            return {"relevant_chunks": direct_hits}
        if not candidates:
            return {"relevant_chunks": []}

        numbered = []
        for i, c in enumerate(candidates[:20], 1):
            path = c.get("path") or c.get("section_path", "")
            txt = c["text"][:300]
            numbered.append(f"[{i}] ({path}): {txt}")

        prompt = BATCH_EVAL_PROMPT.format(
            query=state["query"],
            chunks="\n\n".join(numbered),
        )
        raw = self._llm(prompt, max_tokens=600, temperature=0.1)
        parsed = self._parse_json(raw)

        relevant = []
        if isinstance(parsed, list):
            for item in parsed:
                idx = item.get("index", 0) - 1
                score = item.get("score", 0)
                if 0 <= idx < len(candidates) and (item.get("relevant") or score >= 5):
                    candidates[idx]["relevance_score"] = score
                    relevant.append(candidates[idx])
        else:
            for c in candidates[:10]:
                c["relevance_score"] = 5
                relevant.append(c)

        relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        final = direct_hits + [c for c in relevant
                               if c.get("id") not in {d.get("id") for d in direct_hits}]
        return {"relevant_chunks": final[:10]}

    def _node_generate(self, state: AgentState) -> dict:
        chunks = state.get("relevant_chunks", [])
        context_parts, sources = [], []

        for i, c in enumerate(chunks, 1):
            path = c.get("path") or c.get("section_path", "غير محدد")
            label = "نص قانوني" if c.get("source") == "law" else "مذكرة توضيحية"
            context_parts.append(f"[{i}] ({label} — {path}):\n{c['text']}")
            sources.append({
                "index": i,
                "source_type": c.get("source", ""),
                "section_path": path,
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
            })

        context = "\n\n---\n\n".join(context_parts) if context_parts else "(لا توجد نصوص مرجعية)"
        prompt = GENERATE_PROMPT.format(context=context, query=state["query"])
        response = self._llm(prompt, max_tokens=2048, temperature=0.4)

        return {"response": response, "sources": sources}

    def _node_check(self, state: AgentState) -> dict:
        prompt = CHECK_PROMPT.format(
            query=state["query"],
            response=state.get("response", ""),
        )
        raw = self._llm(prompt, max_tokens=300, temperature=0.1)
        parsed = self._parse_json(raw) if isinstance(self._parse_json(raw), dict) else {}
        return {
            "needs_clarification":    parsed.get("needs_clarification", False),
            "clarification_question": parsed.get("clarification_question", ""),
        }

    def _route_after_eval(self, state: AgentState) -> Literal["retry", "generate"]:
        if len(state.get("relevant_chunks", [])) < 2 and state.get("iteration", 0) < 3:
            return "retry"
        return "generate"

    # ── Build LangGraph ────────────────────────────────────────────

    def _build_agent(self):
        from langgraph.graph import StateGraph, START, END

        g = StateGraph(AgentState)
        g.add_node("analyze",  self._node_analyze)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("evaluate", self._node_evaluate)
        g.add_node("generate", self._node_generate)
        g.add_node("check",    self._node_check)

        g.add_edge(START,      "analyze")
        g.add_edge("analyze",  "retrieve")
        g.add_edge("retrieve", "evaluate")
        g.add_conditional_edges(
            "evaluate",
            self._route_after_eval,
            {"retry": "analyze", "generate": "generate"},
        )
        g.add_edge("generate", "check")
        g.add_edge("check",    END)

        return g.compile()

    # ── Public API ─────────────────────────────────────────────────

    def query(self, user_query: str) -> dict:
        """Run the full agentic Graph RAG pipeline."""
        if not self._ready:
            return {"error": "Agent not ready — models still loading"}

        initial: AgentState = {
            "query":                  user_query,
            "intent":                 "",
            "search_entities":        [],
            "search_concepts":        [],
            "search_keywords":        [],
            "retrieved_chunks":       [],
            "relevant_chunks":        [],
            "response":               "",
            "iteration":              0,
            "needs_clarification":    False,
            "clarification_question": "",
            "sources":                [],
        }

        result = self.agent.invoke(initial)

        out = {
            "query":    user_query,
            "response": result.get("response", ""),
            "sources":  result.get("sources", []),
            "intent":   result.get("intent", ""),
            "needs_clarification": result.get("needs_clarification", False),
        }
        if result.get("needs_clarification"):
            out["clarification_question"] = result.get("clarification_question", "")
        return out


# ─── Singleton ────────────────────────────────────────────────────────────────

_agent: GraphRAGAgent | None = None


def get_agent() -> GraphRAGAgent:
    global _agent
    if _agent is None:
        _agent = GraphRAGAgent()
    return _agent
