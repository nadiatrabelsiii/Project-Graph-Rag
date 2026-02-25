"""
/ui — simple browser frontend for asking Graph RAG questions.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI"])


UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Graph RAG Legal Assistant</title>
  <style>
    :root {
      --bg: #f7f2e8;
      --panel: #fffef9;
      --ink: #1f1f1f;
      --muted: #6a645a;
      --line: #ddd2be;
      --accent: #0e5a45;
      --accent-2: #d18d2f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans Arabic", "Noto Sans Arabic", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 20% 0%, #fef4dd 0%, transparent 45%),
        radial-gradient(circle at 90% 15%, #e9f6f0 0%, transparent 35%),
        var(--bg);
      min-height: 100vh;
    }
    .wrap {
      max-width: 980px;
      margin: 2rem auto;
      padding: 0 1rem 2rem;
    }
    .hero {
      background: linear-gradient(130deg, #0f5d47 0%, #0f6e55 55%, #1f8a6e 100%);
      color: #fff;
      border-radius: 20px;
      padding: 1.3rem 1.2rem;
      box-shadow: 0 14px 35px rgba(14, 90, 69, 0.25);
    }
    .hero h1 {
      margin: 0 0 .35rem 0;
      font-size: 1.25rem;
      letter-spacing: .2px;
    }
    .hero p {
      margin: 0;
      color: #e9fff7;
      font-size: .95rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
      margin-top: 1rem;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 1rem;
      box-shadow: 0 8px 18px rgba(0,0,0,.04);
    }
    label {
      font-size: .9rem;
      color: var(--muted);
      display: block;
      margin-bottom: .35rem;
    }
    textarea {
      width: 100%;
      min-height: 130px;
      resize: vertical;
      border: 1px solid #cabda8;
      border-radius: 12px;
      padding: .8rem .9rem;
      font: inherit;
      background: #fffdfa;
    }
    textarea:focus {
      border-color: var(--accent);
      outline: 2px solid rgba(14,90,69,.15);
      outline-offset: 0;
    }
    .actions {
      margin-top: .75rem;
      display: flex;
      gap: .55rem;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: .62rem 1rem;
      font: inherit;
      cursor: pointer;
    }
    .primary {
      background: var(--accent);
      color: #fff;
      font-weight: 600;
    }
    .ghost {
      background: #efe6d7;
      color: #423d34;
    }
    .status {
      min-height: 1.25rem;
      margin-top: .55rem;
      font-size: .88rem;
      color: var(--muted);
    }
    .answer {
      white-space: pre-wrap;
      line-height: 1.65;
      font-size: .98rem;
      direction: rtl;
      text-align: right;
      background: #fff;
      border: 1px solid #ece2d2;
      border-radius: 12px;
      padding: .9rem;
    }
    .meta {
      margin-top: .65rem;
      font-size: .9rem;
      color: #4f483d;
      background: #fbf6ec;
      border: 1px dashed #e2d5bf;
      border-radius: 10px;
      padding: .6rem .7rem;
    }
    .sources {
      margin-top: .75rem;
      display: grid;
      gap: .5rem;
    }
    .src {
      border: 1px solid #e7dbc8;
      border-left: 6px solid var(--accent-2);
      border-radius: 10px;
      padding: .5rem .65rem;
      background: #fffefb;
      font-size: .9rem;
    }
    .empty {
      color: var(--muted);
      font-size: .92rem;
      border: 1px dashed #d9ccb7;
      border-radius: 10px;
      padding: .8rem;
      background: #fffdfa;
    }
    @media (min-width: 900px) {
      .grid { grid-template-columns: 1fr 1.1fr; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>Agentic Graph RAG Legal Assistant</h1>
      <p>Ask in Arabic or English. You will get an answer with supporting sources from your legal graph.</p>
    </section>

    <section class="grid">
      <article class="card">
        <label for="query">Your legal question</label>
        <textarea id="query" placeholder="مثال: ما هو الفصل 1؟"></textarea>
        <div class="actions">
          <button id="ask" class="primary">Ask</button>
          <button id="sample1" class="ghost">Sample: الفصل 1</button>
          <button id="sample2" class="ghost">Sample: الفرق بين قانون المالية والمرسوم 15</button>
        </div>
        <div id="status" class="status"></div>
      </article>

      <article class="card">
        <div id="resultEmpty" class="empty">No response yet. Submit a question to visualize the result.</div>
        <div id="resultBox" style="display:none;">
          <div id="answer" class="answer"></div>
          <div id="meta" class="meta"></div>
          <div id="sources" class="sources"></div>
        </div>
      </article>
    </section>
  </main>

  <script>
    const queryInput = document.getElementById("query");
    const askBtn = document.getElementById("ask");
    const sample1 = document.getElementById("sample1");
    const sample2 = document.getElementById("sample2");
    const statusBox = document.getElementById("status");
    const resultEmpty = document.getElementById("resultEmpty");
    const resultBox = document.getElementById("resultBox");
    const answerBox = document.getElementById("answer");
    const metaBox = document.getElementById("meta");
    const sourcesBox = document.getElementById("sources");

    function renderSources(sources) {
      sourcesBox.innerHTML = "";
      if (!sources || !sources.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No sources returned.";
        sourcesBox.appendChild(empty);
        return;
      }
      for (const s of sources) {
        const el = document.createElement("div");
        el.className = "src";
        el.textContent = `[S${s.index}] ${s.source_type || "source"} | ${s.section_path || "n/a"} | pages: ${s.page_start ?? "-"}-${s.page_end ?? "-"}`;
        sourcesBox.appendChild(el);
      }
    }

    async function askQuestion() {
      const query = queryInput.value.trim();
      if (!query) {
        statusBox.textContent = "Please type a question first.";
        return;
      }
      askBtn.disabled = true;
      statusBox.textContent = "Running query...";
      try {
        const res = await fetch("/api/query", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query})
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`HTTP ${res.status} - ${text}`);
        }
        const data = await res.json();
        resultEmpty.style.display = "none";
        resultBox.style.display = "block";
        answerBox.textContent = data.response || "(empty response)";
        metaBox.textContent = `Intent: ${data.intent || "n/a"} | Clarification needed: ${data.needs_clarification ? "yes" : "no"}${data.clarification_question ? " | Clarification: " + data.clarification_question : ""}`;
        renderSources(data.sources || []);
        statusBox.textContent = "Done.";
      } catch (err) {
        statusBox.textContent = `Error: ${err.message}`;
      } finally {
        askBtn.disabled = false;
      }
    }

    askBtn.addEventListener("click", askQuestion);
    queryInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") askQuestion();
    });
    sample1.addEventListener("click", () => queryInput.value = "ما هو الفصل 1؟");
    sample2.addEventListener("click", () => queryInput.value = "ما الفرق بين أحكام قانون المالية وأحكام المرسوم عدد 15 لسنة 2022؟");
  </script>
</body>
</html>
"""


@router.get("/ui", response_class=HTMLResponse)
async def ui_page() -> HTMLResponse:
    return HTMLResponse(content=UI_HTML)

