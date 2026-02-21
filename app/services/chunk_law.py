"""
Structure-aware chunker for OCR'd Arabic legal/financial HTML → Graph RAG.

v4 — Multi-file support + document year detection.

Addresses:
  - Heading text was leaking into previous article (now stripped)
  - Inline cross-references "الفصل 63 من القانون..." falsely detected as articles
  - Split OCR lines ("الفصل\n63 من...") merged before parsing
  - Article boundary = start of الفصل N separator, heading lines excluded

Supports multiple OCR HTML files in a directory — each file is a separate
document.  The document title and year are auto-detected from the header
(e.g.  قانون عدد 17 لسنة 2025 مؤرخ في 12 ديسمبر 2025 يتعلق بقانون المالية لسنة 2026).

NO markdown # dependency — stripped on load.

Usage:
    python chunk_graphrag.py
    python chunk_graphrag.py --input-dir OCR_Law --output chunks_graphrag.json
    python chunk_graphrag.py --input ocr_output.html --output chunks.json
"""

from __future__ import annotations
import argparse, glob, json, re, uuid
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

# Year detection:  قانون المالية لسنة 2026   or   لسنة 2025
RE_LAW_YEAR_FINANCE = re.compile(r"قانون\s+المالية\s+لسنة\s+(\d{4})")
RE_LAW_YEAR_GENERIC = re.compile(r"لسنة\s+(\d{4})")

# Document title:  قانون عدد ... لسنة ... يتعلق ...
RE_LAW_TITLE = re.compile(r"(قانون\s+عدد\s+\d+\s+لسنة\s+\d{4}[^\n]*)")


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str = ""
    chunk_type: str = ""
    document_title: str = ""
    document_year: Optional[str] = None
    source_file: str = ""
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

def detect_law_title_and_year(lines: list[str], limit: int = 30) -> tuple[str, Optional[str]]:
    """Scan the first `limit` lines for a law title and the target year.

    The *year* we want is the fiscal / application year, e.g. for
    'قانون عدد 17 لسنة 2025 يتعلق بقانون المالية لسنة 2026' → year = '2026'.
    """
    doc_title = ""
    doc_year: Optional[str] = None

    text_block = "\n".join(lines[:limit])

    # Title: first occurrence of  قانون عدد ...
    tm = RE_LAW_TITLE.search(text_block)
    if tm:
        doc_title = re.sub(r"\s+", " ", tm.group(1)).strip().rstrip(".()١")

    # Year: prefer "قانون المالية لسنة YYYY" (fiscal year), else last لسنة YYYY
    fm = RE_LAW_YEAR_FINANCE.search(text_block)
    if fm:
        doc_year = fm.group(1)
    else:
        all_years = RE_LAW_YEAR_GENERIC.findall(text_block)
        if all_years:
            doc_year = all_years[-1]  # last year found is typically the target

    return doc_title, doc_year


def load(path: str) -> tuple[list[str], list[tuple[int, int]], set[int], str, Optional[str]]:
    """Load HTML, strip markdown #, rejoin OCR-split 'الفصل' lines.
    Returns (lines, page_map, md_heading_lines, doc_title, doc_year) where
    md_heading_lines is the set of output line indices that originally had
    a # prefix in the OCR.
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

    doc_title, doc_year = detect_law_title_and_year(lines)

    return lines, page_map, md_heading_lines, doc_title, doc_year


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
    doc_title: str = "",
    doc_year: Optional[str] = None,
    source_file: str = "",
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

    # Identify أحكام الميزانية section boundaries
    ahkam_start = None
    mihwar1_idx = None
    for i, c in enumerate(classified):
        if ahkam_start is None and "أحكام الميزانية" in c["text"]:
            ahkam_start = i
        if mihwar1_idx is None and re.search(r"المحور\s+الأو[ّ]?ل", c["text"]):
            mihwar1_idx = i
    # No special chunking: treat articles inside أحكام الميزانية as normal articles

    # --- Each article ---
    for ai, art in enumerate(art_starts):
        start_ci = _ci_of(classified, art["idx"])
        # End: next article_start or zone_end
        if ai + 1 < len(art_starts):
            end_ci = _ci_of(classified, art_starts[ai + 1]["idx"])
        else:
            end_ci = len(classified)

        # Section path: أحكام الميزانية if inside its boundaries, otherwise المحور/subheading
        section_path = ""
        if ahkam_start is not None and mihwar1_idx is not None and start_ci >= ahkam_start and start_ci < mihwar1_idx:
            section_path = "أحكام الميزانية"
        else:
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

        # Collect ALL lines between the last heading/subheading and article start, and all lines up to the next article
        art_lines = []
        # Merge multi-line headlines/subheadlines before الفصل, attach to section_path, not chunk text
        art_lines = []
        # Find the first ARTICLE_START line in this chunk
        article_found = False
        for c in classified[start_ci:end_ci]:
            if not article_found:
                if c["type"] == ARTICLE_START:
                    article_found = True
                    art_lines.append(c["text"])
                # else: skip lines before الفصل (they are used for section_path/subheading only)
            else:
                if c["type"] in (CONTENT, TABLE_LINE, ARTICLE_START):
                    art_lines.append(c["text"])

        text = clean_text("\n".join(art_lines))
        if not text.strip():
            continue

        art_num = art["art_num"]

        # Page range
        pg_start = page_at(art["idx"], page_map)
        last_line_idx = classified[end_ci - 1]["idx"] if end_ci > start_ci else art["idx"]
        pg_end = page_at(last_line_idx, page_map)

        refs = extract_refs(text)
        tok = est_tokens(text)

        # Always treat as one article chunk, no subchunking by numbers
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            chunk_type="article",
            document_title=doc_title,
            document_year=doc_year,
            source_file=source_file,
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
        if c.chunk_type not in ("article", "article_subchunk"):
            continue
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
    doc_title: str = "",
    doc_year: Optional[str] = None,
    source_file: str = "",
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
                document_title=doc_title,
                document_year=doc_year,
                source_file=source_file,
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
                document_title=doc_title,
                document_year=doc_year,
                source_file=source_file,
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

def _collect_input_files(input_path: str | None, input_dir: str | None) -> list[str]:
    """Resolve input HTML files from --input or --input-dir."""
    files: list[str] = []
    if input_dir:
        pattern = str(Path(input_dir) / "*.html")
        files = sorted(glob.glob(pattern))
    elif input_path:
        if "*" in input_path:
            files = sorted(glob.glob(input_path))
        else:
            files = [input_path]
    return [f for f in files if Path(f).is_file()]


def process_single_file(input_path: str, max_table_tokens: int,
                        max_article_tokens: int) -> list[Chunk]:
    """Process one OCR HTML file and return its chunks."""
    source_file = Path(input_path).name
    print(f"\n{'='*60}")
    print(f"Processing: {source_file}")
    print(f"{'='*60}")

    lines, page_map, md_heading_lines, doc_title, doc_year = load(input_path)
    print(f"  {len(lines)} lines, {len(page_map)} page markers")
    print(f"  Title: {doc_title[:80] or '(not detected)'}")
    print(f"  Year:  {doc_year or '(not detected)'}")

    zone_boundary = find_zone_boundary(lines)
    print(f"  Zone boundary at line {zone_boundary}")

    classified = classify_lines(lines, zone_boundary)

    types = {}
    for c in classified:
        types[c["type"]] = types.get(c["type"], 0) + 1
    print(f"  Line types: {types}")

    # Build law chunks
    print("  Chunking law text...")
    law_chunks = build_law_chunks(
        lines, classified, page_map, zone_boundary, max_article_tokens,
        md_heading_lines, doc_title=doc_title, doc_year=doc_year,
        source_file=source_file,
    )
    print(f"  {len(law_chunks)} law chunks")

    # Budget tables
    print("  Chunking budget tables...")
    table_chunks = chunk_tables(
        lines, page_map, zone_boundary, max_table_tokens,
        doc_title=doc_title, doc_year=doc_year, source_file=source_file,
    )
    print(f"  {len(table_chunks)} table chunks")

    all_chunks = law_chunks + table_chunks

    # Stats
    tokens = [c.token_count for c in all_chunks]
    if tokens:
        print(f"  Total: {len(all_chunks)} chunks, {sum(tokens)} tokens")
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
            print(f"  WARNING: Missing articles: {missing}")
        else:
            print(f"  Articles {min(art_nums)}-{max(art_nums)}: complete ✓")

    return all_chunks


def run(input_path="ocr_output.html", output_path="chunks_graphrag.json",
        max_table_tokens=1000, max_article_tokens=1000, input_dir=None):
    files = _collect_input_files(input_path, input_dir)
    if not files:
        print(f"ERROR: No HTML files found (input={input_path}, input_dir={input_dir})")
        return

    print(f"Found {len(files)} input file(s): {[Path(f).name for f in files]}")

    all_chunks: list[Chunk] = []
    for fpath in files:
        chunks = process_single_file(fpath, max_table_tokens, max_article_tokens)
        all_chunks.extend(chunks)

    # Overall stats
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_chunks)} chunks from {len(files)} file(s)")
    tokens = [c.token_count for c in all_chunks]
    if tokens:
        print(f"  Total tokens: {sum(tokens)}")
        print(f"  min={min(tokens)} max={max(tokens)} avg={sum(tokens)//len(tokens)}")

    # Write
    out = [asdict(c) for c in all_chunks]
    Path(output_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None,
                   help="Single HTML file or glob pattern (e.g. 'OCR_Law/*.html')")
    p.add_argument("--input-dir", default=None,
                   help="Directory containing OCR HTML files (e.g. 'OCR_Law')")
    p.add_argument("--output", default="chunks_graphrag.json")
    p.add_argument("--max-table-tokens", type=int, default=1000)
    p.add_argument("--max-article-tokens", type=int, default=1000)
    args = p.parse_args()

    # Default: look for OCR_Law directory, fall back to single file
    input_path = args.input
    input_dir = args.input_dir
    if not input_path and not input_dir:
        if Path("OCR_Law").is_dir():
            input_dir = "OCR_Law"
        else:
            input_path = "ocr_output.html"

    run(input_path=input_path, output_path=args.output,
        max_table_tokens=args.max_table_tokens,
        max_article_tokens=args.max_article_tokens,
        input_dir=input_dir)


if __name__ == "__main__":
    main()
