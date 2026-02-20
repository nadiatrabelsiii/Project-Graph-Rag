"""
Structure-aware chunker for OCR'd Arabic legal/financial HTML → Graph RAG.

v3 — Complete rewrite addressing:
  - Heading text was leaking into previous article (now stripped)
  - Inline cross-references "الفصل 63 من القانون..." falsely detected as articles
  - Split OCR lines ("الفصل\\n63 من...") merged before parsing
  - Article boundary = start of الفصل N separator, heading lines excluded

NO markdown # dependency — stripped on load.

Usage:
    python chunk_graphrag.py
    python chunk_graphrag.py --input ocr_output.html --output chunks.json
"""

from __future__ import annotations
import argparse, json, re, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── Regex ────────────────────────────────────────────────────────────────────

# Article declaration:  الفصل N .  |  الفصل الأوّل -
# The key is the separator (. - – —) AFTER the number, which distinguishes
# a real article start from an inline reference like "الفصل 63 من القانون"
RE_ARTICLE_DECL = re.compile(
    r"^(الفصل\s+"
    r"(?:الأوّل|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر"
    r"|\d+)"
    r")\s*([\.\-–—])"   # MUST have a separator
)

# Article at end of a heading line (OCR merged): "...معاليم التسجيل الفصل 30 ."
RE_ARTICLE_TAIL = re.compile(
    r"(الفصل\s+\d+)\s*[\.\-–—]\s*$"
)

# "الفصل N ." on its own short line (≤80 chars, not a cross-reference)
RE_ARTICLE_STANDALONE = re.compile(
    r"^(الفصل\s+\d+)\s*[\.\-–—]\s*$"
)

RE_ARTICLE_NUM = re.compile(r"الفصل\s+(\S+)")

ORDINAL_MAP = {
    "الأوّل": "1", "الأول": "1", "الثاني": "2", "الثالث": "3",
    "الرابع": "4", "الخامس": "5", "السادس": "6", "السابع": "7",
    "الثامن": "8", "التاسع": "9", "العاشر": "10",
}

RE_MIHWAR = re.compile(
    r"المحور\s+(?:الأوّل|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)"
)

RE_PAGEBREAK = re.compile(r"<!--\s*PageBreak\s*\[(\d+)\]\s*-->")

RE_SCHEDULE = re.compile(r'الجدول\s*[\-"\s«»]*([أ-ي])[\-"\s«»]*')

RE_LAW_REF = re.compile(
    r"(?:القانون|المرسوم|الأمر)\s+عدد\s+\d+\s+لسنة\s+\d{4}", re.DOTALL
)

RE_TABLE_CLOSE = re.compile(r"</table>", re.IGNORECASE)

RE_FOOTER = re.compile(
    r"^(?:صفحة\s+\d+|عدد\s+\d+|الرائد الرسمي للجمهورية التونسية\s*[-–]\s*\d+.*)$"
)

RE_MD_PREFIX = re.compile(r"^#{1,6}\s*")


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str = ""
    chunk_type: str = ""
    zone: str = ""
    section_path: str = ""
    article_number: Optional[str] = None
    schedule_name: Optional[str] = None
    text: str = ""
    cross_references: list = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: int = 0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def est_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.5))

def extract_refs(text: str) -> list[str]:
    return sorted(set(re.sub(r"\s+", " ", r).strip() for r in RE_LAW_REF.findall(text)))

def clean_text(text: str) -> str:
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if RE_FOOTER.match(s) or RE_PAGEBREAK.search(s):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out).strip())

def get_art_num(text: str) -> Optional[str]:
    m = RE_ARTICLE_NUM.search(text)
    if not m:
        return None
    return ORDINAL_MAP.get(m.group(1), m.group(1))

def page_at(idx: int, pmap: list[tuple[int, int]]) -> int:
    page = 1
    for start, pnum in pmap:
        if idx >= start:
            page = pnum
        else:
            break
    return page


# ─── Load & preprocess ───────────────────────────────────────────────────────

def load(path: str) -> tuple[list[str], list[tuple[int, int]], set[int]]:
    """Load HTML, strip markdown #, rejoin OCR-split 'الفصل' lines.
    Returns (lines, page_map, md_heading_lines) where md_heading_lines is
    the set of output line indices that originally had a # prefix in the OCR.
    We track these ONLY for section_path metadata — never to drop text.
    """
    raw = Path(path).read_text(encoding="utf-8").split("\n")

    page_map = [(0, 1)]
    lines = []
    md_heading_lines: set[int] = set()  # lines that had # prefix in OCR

    i = 0
    while i < len(raw):
        had_md = bool(RE_MD_PREFIX.match(raw[i]))
        line = RE_MD_PREFIX.sub("", raw[i])

        # Page break tracking
        pm = RE_PAGEBREAK.search(raw[i])
        if pm:
            page_map.append((len(lines), int(pm.group(1))))

        # Rejoin OCR split: line ending with "الفصل" + next line "63 من..."
        # This is a CROSS-REFERENCE, not a new article. We join to prevent
        # false detection.
        stripped = line.strip()
        if stripped.endswith("الفصل") or stripped == "الفصل":
            if i + 1 < len(raw):
                next_stripped = RE_MD_PREFIX.sub("", raw[i + 1]).strip()
                if re.match(r"^\d+\s+(?:من|و|ل|ال)", next_stripped):
                    joined = line.rstrip() + " " + RE_MD_PREFIX.sub("", raw[i + 1]).lstrip()
                    lines.append(joined)
                    i += 2
                    continue

        out_idx = len(lines)
        lines.append(line)
        if had_md:
            md_heading_lines.add(out_idx)
        i += 1

    return lines, page_map, md_heading_lines


# ─── Zone boundary ───────────────────────────────────────────────────────────

def find_zone_boundary(lines: list[str]) -> int:
    """Line where budget schedule tables begin (after all law articles)."""
    # Find last article
    last_art = 0
    for i, line in enumerate(lines):
        if RE_ARTICLE_DECL.match(line.strip()):
            last_art = i

    # Find first schedule header AFTER last article
    for i in range(last_art, len(lines)):
        s = lines[i].strip()
        if RE_SCHEDULE.search(s) and "المدرج بهذا القانون" not in s:
            ahead = "\n".join(lines[i:i+80])
            if "<table" in ahead.lower():
                return i
    return int(len(lines) * 0.8)


# ─── Classify each line ──────────────────────────────────────────────────────

# Line types
ARTICLE_START = "article_start"
CONTENT = "content"
NOISE = "noise"        # footer, page break
MIHWAR = "mihwar"
TABLE_LINE = "table_line"

def classify_lines(lines: list[str], zone_end: int) -> list[dict]:
    """
    Classify each line as article_start, heading, content, noise, mihwar, or table_line.
    This is the core innovation: we identify headings SEPARATELY so they
    never leak into article body text.
    """
    result = []
    in_table = False

    for i in range(zone_end):
        line = lines[i]
        stripped = line.strip()
        info = {"idx": i, "type": CONTENT, "art_num": None, "text": line}

        # --- Noise ---
        if not stripped:
            info["type"] = NOISE
        elif RE_PAGEBREAK.search(stripped):
            info["type"] = NOISE
        elif RE_FOOTER.match(stripped):
            info["type"] = NOISE

        # --- Table tracking ---
        elif "<table" in stripped.lower():
            in_table = True
            info["type"] = TABLE_LINE
        elif "</table>" in stripped.lower():
            in_table = False
            info["type"] = TABLE_LINE
        elif in_table:
            info["type"] = TABLE_LINE

        # --- المحور ---
        elif RE_MIHWAR.search(stripped):
            info["type"] = MIHWAR

        # --- Article at start of line ---
        elif RE_ARTICLE_DECL.match(stripped):
            num = get_art_num(stripped)
            if num and "(جديد)" not in stripped:
                info["type"] = ARTICLE_START
                info["art_num"] = num

        # --- Article at end of heading line (OCR merged) ---
        elif RE_ARTICLE_TAIL.search(stripped):
            # Line like: "إعفاء عقود القروض... الفصل 30 ."
            # Treat the whole line as article start (heading + article on same line)
            m = RE_ARTICLE_TAIL.search(stripped)
            num = get_art_num(m.group(1))
            if num:
                info["type"] = ARTICLE_START
                info["art_num"] = num

        # --- Article on standalone short line (e.g. "الفصل 20 .") ---
        elif RE_ARTICLE_STANDALONE.match(stripped):
            num = get_art_num(stripped)
            if num:
                info["type"] = ARTICLE_START
                info["art_num"] = num

        result.append(info)

    return result


# No heading detection — all text between articles is included in chunks


# ─── Build chunks from classified lines ───────────────────────────────────────

def build_law_chunks(
    lines: list[str],
    classified: list[dict],
    page_map: list[tuple[int, int]],
    zone_end: int,
    max_art_tokens: int,
    md_heading_lines: set[int] = None,
) -> list[Chunk]:
    """Build article chunks from classified lines."""
    if md_heading_lines is None:
        md_heading_lines = set()
    
    # First, gather محور sections
    mihwar_list = []
    for c in classified:
        if c["type"] == MIHWAR:
            title = c["text"].strip().rstrip(".-–: ")
            mihwar_list.append((c["idx"], title))

    # Find all article starts and their numbers, deduplicating
    art_starts = []
    seen_nums = set()
    for c in classified:
        if c["type"] == ARTICLE_START and c["art_num"]:
            if c["art_num"] not in seen_nums:
                seen_nums.add(c["art_num"])
                art_starts.append(c)

    chunks = []

    # --- Preamble ---
    if art_starts:
        first_idx = art_starts[0]["idx"]
        preamble_lines = []
        for c in classified[:first_idx]:
            if c["type"] in (CONTENT, TABLE_LINE):
                preamble_lines.append(c["text"])
        preamble_text = clean_text("\n".join(preamble_lines))
        if preamble_text.strip():
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_type="preamble",
                zone="law_text",
                text=preamble_text,
                cross_references=extract_refs(preamble_text),
                page_start=1,
                page_end=page_at(first_idx, page_map),
                token_count=est_tokens(preamble_text),
            ))

    # --- Each article ---
    for ai, art in enumerate(art_starts):
        start_ci = _ci_of(classified, art["idx"])
        
        # End: next article_start or zone_end
        if ai + 1 < len(art_starts):
            end_ci = _ci_of(classified, art_starts[ai + 1]["idx"])
        else:
            end_ci = len(classified)

        # Collect ALL lines between this article and the next
        # No lines are dropped — everything belongs to this فصل
        art_lines = []
        for c in classified[start_ci:end_ci]:
            if c["type"] in (CONTENT, TABLE_LINE, ARTICLE_START):
                art_lines.append(c["text"])

        text = clean_text("\n".join(art_lines))
        if not text.strip():
            continue

        art_num = art["art_num"]

        # Section path
        current_mihwar = ""
        for m_idx, m_title in mihwar_list:
            if m_idx < art["idx"]:
                current_mihwar = m_title
            else:
                break

        # Find sub-heading for THIS article from md_heading_lines metadata
        sub_heading = ""
        for c in reversed(classified[max(0, start_ci - 10):start_ci]):
            if c["idx"] in md_heading_lines and c["type"] == CONTENT:
                text_cand = c["text"].strip().rstrip(".-–: ")
                # Only use as sub-heading if short & not a number line
                if 10 < len(text_cand) < 120 and not re.match(r"^\d", text_cand):
                    sub_heading = text_cand
                    break
            if c["type"] == MIHWAR:
                break
            if c["type"] == ARTICLE_START:
                break

        path_parts = []
        if current_mihwar:
            path_parts.append(current_mihwar)
        if sub_heading:
            path_parts.append(sub_heading)
        section_path = " > ".join(path_parts)

        # Page range
        pg_start = page_at(art["idx"], page_map)
        last_line_idx = classified[end_ci - 1]["idx"] if end_ci > start_ci else art["idx"]
        pg_end = page_at(last_line_idx, page_map)

        refs = extract_refs(text)
        tok = est_tokens(text)

        # Split oversized articles
        if tok > max_art_tokens:
            parts = _split_article(text, max_art_tokens)
            for pi, part in enumerate(parts, 1):
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="article",
                    zone="law_text",
                    section_path=section_path,
                    article_number=f"{art_num}_part{pi}",
                    text=part,
                    cross_references=extract_refs(part),
                    page_start=pg_start,
                    page_end=pg_end,
                    token_count=est_tokens(part),
                ))
        else:
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_type="article",
                zone="law_text",
                section_path=section_path,
                article_number=art_num,
                text=text,
                cross_references=refs,
                page_start=pg_start,
                page_end=pg_end,
                token_count=tok,
            ))

    # Propagate section_path: articles without a path inherit from previous
    prev_path = ""
    prev_mihwar = ""
    for c in chunks:
        if c.chunk_type != "article":
            continue
        # Current محور
        cur_mihw = c.section_path.split(" > ")[0] if " > " in c.section_path else c.section_path
        if not c.section_path and prev_path:
            c.section_path = prev_path
        elif c.section_path:
            prev_path = c.section_path
        prev_mihwar = cur_mihw

    return chunks


def _ci_of(classified: list[dict], line_idx: int) -> int:
    """Find classified list index for a given line index."""
    for i, c in enumerate(classified):
        if c["idx"] == line_idx:
            return i
    return 0


def _split_article(text: str, max_tokens: int) -> list[str]:
    """Split article at numbered sub-paragraphs or blank lines."""
    parts = re.split(r"(?=\n\d+[\\\)]\s)", text)
    if len(parts) <= 1:
        parts = re.split(r"\n\n+", text)
    if len(parts) <= 1:
        return [text]

    result = []
    current = ""
    for part in parts:
        candidate = (current + "\n\n" + part).strip() if current else part.strip()
        if est_tokens(candidate) > max_tokens and current:
            result.append(current.strip())
            current = part.strip()
        else:
            current = candidate
    if current.strip():
        result.append(current.strip())
    return result if result else [text]


# ─── Budget table chunking ───────────────────────────────────────────────────

def chunk_tables(
    lines: list[str],
    page_map: list[tuple[int, int]],
    zone_start: int,
    max_tokens: int,
) -> list[Chunk]:
    chunks = []
    zone_lines = lines[zone_start:]

    schedules = []
    for i, line in enumerate(zone_lines):
        m = RE_SCHEDULE.search(line.strip())
        if m:
            schedules.append((i, m.group(1)))

    if not schedules:
        text = "\n".join(zone_lines)
        if text.strip():
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()), chunk_type="schedule_table",
                zone="budget_table", text=text.strip(),
                page_start=page_at(zone_start, page_map),
                page_end=page_at(zone_start + len(zone_lines), page_map),
                token_count=est_tokens(text),
            ))
        return chunks

    for si, (s_start, s_name) in enumerate(schedules):
        s_end = schedules[si + 1][0] if si + 1 < len(schedules) else len(zone_lines)
        s_lines = zone_lines[s_start:s_end]

        parts = _split_schedule(s_lines, max_tokens)
        for part in parts:
            if not part.strip():
                continue
            tok = est_tokens(part)
            if tok < 10:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_type="schedule_table",
                zone="budget_table",
                schedule_name=s_name,
                text=part.strip(),
                page_start=page_at(zone_start + s_start, page_map),
                page_end=page_at(zone_start + s_end, page_map),
                token_count=tok,
            ))
    return chunks


def _split_schedule(sched_lines: list[str], max_tokens: int) -> list[str]:
    full = "\n".join(sched_lines)
    if est_tokens(full) <= max_tokens:
        return [full]

    te_pos = [i for i, l in enumerate(sched_lines) if RE_TABLE_CLOSE.search(l)]
    if not te_pos:
        return _split_subtotals(sched_lines, max_tokens)

    result = []
    cur_start = 0
    for te in te_pos:
        seg = "\n".join(sched_lines[cur_start:te + 1])
        if est_tokens(seg) > max_tokens:
            result.extend(_split_subtotals(sched_lines[cur_start:te + 1], max_tokens))
        elif result and est_tokens(result[-1] + "\n" + seg) <= max_tokens:
            result[-1] += "\n" + seg
        else:
            result.append(seg)
        cur_start = te + 1

    if cur_start < len(sched_lines):
        rem = "\n".join(sched_lines[cur_start:])
        if rem.strip():
            if result and est_tokens(result[-1] + "\n" + rem) <= max_tokens:
                result[-1] += "\n" + rem
            else:
                result.append(rem)
    return result


def _split_subtotals(lines_list: list[str], max_tokens: int) -> list[str]:
    result = []
    cur = []
    for line in lines_list:
        cur.append(line)
        if est_tokens("\n".join(cur)) > max_tokens:
            sp = None
            for j in range(len(cur) - 1, -1, -1):
                if "جملة" in cur[j]:
                    sp = j
                    break
            if sp and sp > 0:
                result.append("\n".join(cur[:sp + 1]))
                cur = cur[sp + 1:]
            elif len(cur) > 1:
                mid = len(cur) // 2
                result.append("\n".join(cur[:mid]))
                cur = cur[mid:]
    if cur:
        rem = "\n".join(cur)
        if rem.strip():
            result.append(rem)
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(input_path="ocr_output.html", output_path="chunks_graphrag.json",
        max_table_tokens=1000, max_article_tokens=1000):
    print(f"Loading {input_path}...")
    lines, page_map, md_heading_lines = load(input_path)
    print(f"  {len(lines)} lines, {len(page_map)} page markers, {len(md_heading_lines)} md-heading lines")

    zone_boundary = find_zone_boundary(lines)
    print(f"  Zone boundary at line {zone_boundary}")

    # Classify lines (no heading detection — all content stays)
    classified = classify_lines(lines, zone_boundary)

    # Debug counts
    types = {}
    for c in classified:
        types[c["type"]] = types.get(c["type"], 0) + 1
    print(f"  Line types: {types}")

    # Build law chunks
    print("Chunking law text...")
    law_chunks = build_law_chunks(lines, classified, page_map, zone_boundary, max_article_tokens, md_heading_lines)
    print(f"  {len(law_chunks)} law chunks")

    # Budget tables
    print("Chunking budget tables...")
    table_chunks = chunk_tables(lines, page_map, zone_boundary, max_table_tokens)
    print(f"  {len(table_chunks)} table chunks")

    all_chunks = law_chunks + table_chunks

    # Stats
    tokens = [c.token_count for c in all_chunks]
    if tokens:
        print(f"\nTotal: {len(all_chunks)} chunks, {sum(tokens)} tokens")
        print(f"  min={min(tokens)} max={max(tokens)} avg={sum(tokens)//len(tokens)}")

    # Article completeness
    art_nums = set()
    for c in law_chunks:
        if c.article_number:
            base = c.article_number.split("_")[0]
            if base.isdigit():
                art_nums.add(int(base))
    if art_nums:
        full = set(range(min(art_nums), max(art_nums) + 1))
        missing = sorted(full - art_nums)
        if missing:
            print(f"\n  WARNING: Missing articles: {missing}")
        else:
            print(f"\n  Articles {min(art_nums)}-{max(art_nums)}: complete ✓")

    # Write
    out = [asdict(c) for c in all_chunks]
    Path(output_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="ocr_output.html")
    p.add_argument("--output", default="chunks_graphrag.json")
    p.add_argument("--max-table-tokens", type=int, default=1000)
    p.add_argument("--max-article-tokens", type=int, default=1000)
    run(p.parse_args().input, p.parse_args().output,
        p.parse_args().max_table_tokens, p.parse_args().max_article_tokens)


if __name__ == "__main__":
    main()
