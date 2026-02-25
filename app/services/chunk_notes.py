"""
Chunker for مذكرة عامة (general notes / circulars) — Arabic OCR'd HTML → Graph RAG.

This handles documents structured by numbered sections (1., 2., 3...),
sub-sections (أ, ب), and Roman numerals (I, II, III) rather than فصل articles.

Supports multiple OCR HTML files in a directory — each file is a separate
document.  The document year is auto-detected from the title line
(e.g.  مذكرة عامة عدد 02 لسنة 2026).

Approach:
  1. Parse document into sections using line-by-line scanning
  2. Detect section boundaries from numbered patterns and standalone short lines
  3. Each section + its body text = one chunk
  4. Build hierarchical section_path (parent > child)
  5. Strip noise (footers, page numbers, page breaks)
  6. NO text dropped — all content between boundaries goes into the chunk

Usage:
    python chunk_notes.py
    python chunk_notes.py --input-dir OCR_Notes --output chunks_notes.json
    python chunk_notes.py --input ocr_output_notes.html --output chunks_notes.json
"""

from __future__ import annotations
import argparse, glob, json, re, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── Regex ────────────────────────────────────────────────────────────────────

RE_MD_PREFIX = re.compile(r"^\s*(#{1,6})\s*")

RE_PAGEBREAK = re.compile(r"<!--\s*PageBreak\s*\[(\d+)\]\s*-->")

RE_FOOTER = re.compile(
    r"^(?:"
    r"صفحة\s+\d+|"
    r"عدد\s+\d+|"
    r"الرائد الرسمي للجمهورية التونسية\s*[-–].*|"
    r"page\s+\d+|"
    r"journal officiel de la republique tunisienne.*|"
    r"n[°oº]\s*\d+|"
    r"\d{1,3}"
    r")$",
    re.IGNORECASE,
)

RE_LAW_REF = re.compile(
    r"(?:القانون|المرسوم|الأمر|الفصل)\s+(?:عدد\s+)?\d+\s+(?:لسنة|من)", re.DOTALL
)
RE_LAW_REF_FR = re.compile(
    r"(?:loi|decret|décret|décret-loi)\s*(?:n[°oº]?\s*[\d\-–—]+)?\s*(?:du|de)?\s*[^,\n]*?(?:19\d{2}|20\d{2})",
    re.IGNORECASE | re.DOTALL,
)
RE_ARTICLE_SRC_REF_AR = re.compile(
    r"(?:الفصل|فصل)\s+(\d+)\s*(?:مكرر|مكرّر|\(جديد\)|\(مكرر\))?\s*"
    r"(?:من|بالفصل)\s+"
    r"(القانون|المرسوم|الأمر)\s+عدد\s+(\d+)\s+لسنة\s+(\d{4})",
    re.DOTALL,
)
RE_ARTICLE_SRC_REF_FR = re.compile(
    r"(?:article|art\.?)\s*(\d+|1er|premi(?:er|ere))\s*"
    r"(?:du|de la|de l['’])\s*"
    r"(loi|decret|décret)\s+n?[°oº]?\s*([\d\-]+)\s*(?:de|/)\s*(\d{4})",
    re.IGNORECASE | re.DOTALL,
)
RE_ARTICLE_SRC_REF_FR_ALT = re.compile(
    r"(?:article|art\.?)\s*(\d+|1er|premi(?:er|ere)).{0,60}?"
    r"(loi|decret|décret)\s+n?[°oº]?\s*([\d]+(?:[-–—]\d+)?)",
    re.IGNORECASE | re.DOTALL,
)

# Year detection from title:  مذكرة عامة عدد 02 لسنة 2026
RE_NOTE_YEAR = re.compile(r"لسنة\s+(\d{4})")
RE_NOTE_YEAR_FR = re.compile(
    r"(?:l['’]ann[eé]e\s+|n[°oº]\s*\d+\s*[/\-]\s*|/)\s*(19\d{2}|20\d{2})",
    re.IGNORECASE,
)

RE_DATE_STAMP_FR = re.compile(
    r"^\d{1,2}\s*(?:JAN|F[ÉE]V|MAR|AVR|MAI|JUIN|JUIL|AOU|AO[ÛU]|SEP|OCT|NOV|D[ÉE]C)\s*\d{4}$",
    re.IGNORECASE,
)
RE_DATE_STAMP_FR_ALT = re.compile(
    r"^\d{1,2}\s+\d{1,2}\s+(?:JAN|F[ÉE]V|MAR|AVR|MAI|JUIN|JUIL|AOU|AO[ÛU]|SEP|OCT|NOV|D[ÉE]C)\s+\d{4}$",
    re.IGNORECASE,
)


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str = ""
    chunk_type: str = ""           # "section", "summary", "preamble", "signature"
    document_title: str = ""
    document_year: Optional[str] = None
    source_file: str = ""
    section_path: str = ""
    section_number: Optional[str] = None
    text: str = ""
    cross_references: list = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: int = 0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def est_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.5))


def extract_refs(text: str) -> list[str]:
    refs: list[str] = []
    for m in RE_ARTICLE_SRC_REF_AR.finditer(text):
        refs.append(f"الفصل {m.group(1)} من {m.group(2)} عدد {m.group(3)} لسنة {m.group(4)}")
    for m in RE_ARTICLE_SRC_REF_FR.finditer(text):
        refs.append(
            f"article {m.group(1)} de {m.group(2)} n {m.group(3)} de {m.group(4)}"
        )
    for m in RE_ARTICLE_SRC_REF_FR_ALT.finditer(text):
        src_num = re.sub(r"[–—]", "-", m.group(3))
        src_year = ""
        year_in_num = re.match(r"^(19\d{2}|20\d{2})-", src_num)
        if year_in_num:
            src_year = year_in_num.group(1)
        if src_year:
            refs.append(f"article {m.group(1)} de {m.group(2)} n {src_num} de {src_year}")

    refs.extend(RE_LAW_REF.findall(text))
    refs.extend(RE_LAW_REF_FR.findall(text))
    return sorted(set(re.sub(r"\s+", " ", r).strip() for r in refs if str(r).strip()))


def clean_text(text: str) -> str:
    """Remove noise lines (footers, page breaks, page numbers) but keep all content."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if RE_FOOTER.match(s) or RE_PAGEBREAK.search(s):
            continue
        if not s:
            out.append("")
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out).strip())


def page_at(idx: int, pmap: list[tuple[int, int]]) -> int:
    page = 1
    for start, pnum in pmap:
        if idx >= start:
            page = pnum
        else:
            break
    return page


# ─── Load & preprocess ───────────────────────────────────────────────────────

@dataclass
class ParsedLine:
    idx: int
    raw: str
    text: str              # with # stripped
    md_level: int          # 0 = no heading, 1-6 = heading level
    is_noise: bool
    page: int


def detect_year(text: str) -> Optional[str]:
    """Extract the year from a note title like  مذكرة عامة عدد 02 لسنة 2026."""
    m = RE_NOTE_YEAR.search(text)
    if m:
        return m.group(1)
    mf = RE_NOTE_YEAR_FR.search(text)
    return mf.group(1) if mf else None


def _looks_like_date_stamp(text: str) -> bool:
    s = (text or "").strip()
    return bool(RE_DATE_STAMP_FR.match(s) or RE_DATE_STAMP_FR_ALT.match(s))


def load_and_parse(path: str) -> tuple[list[ParsedLine], str, Optional[str]]:
    """Load HTML, parse each line, track pages.
    Return (parsed_lines, doc_title, document_year)."""
    raw_lines = Path(path).read_text(encoding="utf-8").split("\n")

    parsed = []
    current_page = 1
    doc_title = ""
    doc_year: Optional[str] = None

    for i, raw in enumerate(raw_lines):
        # Page break
        pm = RE_PAGEBREAK.search(raw)
        if pm:
            current_page = int(pm.group(1))

        # Detect markdown heading level (may have leading whitespace from OCR)
        md_match = RE_MD_PREFIX.match(raw)
        md_level = len(md_match.group(1)) if md_match else 0
        text = RE_MD_PREFIX.sub("", raw) if md_match else raw

        stripped = text.strip()

        # Noise detection
        is_noise = False
        if not stripped:
            is_noise = True
        elif RE_PAGEBREAK.search(stripped):
            is_noise = True
        elif RE_FOOTER.match(stripped):
            is_noise = True

        # Capture document title from first h1 / Arabic note line / French note line.
        if not doc_title:
            if md_level == 1:
                doc_title = stripped
            elif "مذكرة" in stripped and "عامة" in stripped:
                doc_title = stripped
            elif re.search(r"note\s+(commune|generale|g[eé]n[eé]rale)", stripped, re.IGNORECASE):
                doc_title = stripped
        else:
            # Prefer a semantic note title over an OCR date stamp.
            if _looks_like_date_stamp(doc_title):
                if ("مذكرة" in stripped and "عامة" in stripped) or re.search(
                    r"note\s+(commune|generale|g[eé]n[eé]rale)", stripped, re.IGNORECASE
                ):
                    doc_title = stripped

        # Detect year from title or early lines (first 20 lines)
        if doc_year is None and i < 20:
            # Normalise stretched letters:  مـــذكـــرة → مذكرة
            normalised = re.sub(r"ـ+", "", stripped)
            year_candidate = detect_year(normalised)
            if year_candidate:
                doc_year = year_candidate

        parsed.append(ParsedLine(
            idx=i, raw=raw, text=text,
            md_level=md_level, is_noise=is_noise,
            page=current_page,
        ))

    # Normalise title (remove stretching)
    doc_title = re.sub(r"ـ+", "", doc_title)

    return parsed, doc_title, doc_year


# ─── Section detection ────────────────────────────────────────────────────────

@dataclass
class Section:
    start_idx: int         # index in parsed_lines
    heading: str           # section heading text
    level: int             # hierarchy depth (1=top, 2=sub, 3=sub-sub)
    number: str            # section number as string (e.g., "1", "2.أ", "III")


def detect_sections(parsed: list[ParsedLine]) -> list[Section]:
    """
    Detect section boundaries from markdown headings.
    Each line with a markdown # prefix starts a new section.
    Level mapping: h1 → level 0, h2 → level 1, h3 → level 2, h4 → level 3
    """
    sections = []

    for idx, pl in enumerate(parsed):
        if pl.is_noise:
            continue

        heading_text = pl.text.strip().rstrip(".-–: ")
        if not heading_text or len(heading_text) < 3:
            continue
        if RE_DATE_STAMP_FR.match(heading_text) or RE_DATE_STAMP_FR_ALT.match(heading_text):
            continue

        # Primary signal: markdown headings.
        if pl.md_level > 0:
            level = pl.md_level
        else:
            # Fallback for OCR without markdown heading markers.
            # Detect structured headings: I., II., 1-, A., أ- ...
            m_pref = re.match(r"^(?:[IVXLC]{1,8}[\.\-]|[0-9]{1,3}[\.\-\)]|[A-Z][\.\-]|[أ-ي][\.\-])\s+", heading_text)
            m_obj = re.match(r"^(?:objet|resume|résumé|الموضوع)\s*[:\-]", heading_text, re.IGNORECASE)
            if not (m_pref or m_obj):
                continue

            # Avoid false positives on long prose lines.
            if len(heading_text) > 180:
                continue
            # Heuristic depth.
            if m_obj:
                level = 1
            elif re.match(r"^[IVXLC]{1,8}[\.\-]\s+", heading_text):
                level = 2
            elif re.match(r"^[0-9]{1,3}[\.\-\)]\s+", heading_text):
                level = 3
            else:
                level = 4

            # If previous line is also a heading marker, keep only one boundary.
            if idx > 0 and not parsed[idx - 1].is_noise and parsed[idx - 1].md_level == 0:
                prev = parsed[idx - 1].text.strip()
                if re.match(r"^(?:[IVXLC]{1,8}|[0-9]{1,3}|[A-Z]|[أ-ي])[\.\-\)]?$", prev):
                    continue

        # Extract section number if present.
        sec_num = ""
        # Patterns: "1.", "1-", "II.", "III.", "أ-", "ب-"
        m = re.match(r"^(\d+[\.\-\)]?|[IVXLC]+[\.\-]?|[A-Z][\.\-]?|[أ-ي][\.\-])\s*", heading_text, re.IGNORECASE)
        if m:
            sec_num = m.group(1).rstrip(".-) ")

        sections.append(Section(
            start_idx=pl.idx,
            heading=heading_text,
            level=level,
            number=sec_num,
        ))

    return sections


# ─── Build chunks ─────────────────────────────────────────────────────────────

def build_chunks(
    parsed: list[ParsedLine],
    sections: list[Section],
    doc_title: str,
    doc_year: Optional[str],
    source_file: str,
    max_tokens: int,
) -> list[Chunk]:
    """Build one chunk per section. All text between sections goes into the chunk."""
    chunks = []

    if not sections:
        # No sections found — one big chunk
        all_text = clean_text("\n".join(pl.text for pl in parsed if not pl.is_noise))
        if all_text.strip():
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_type="section",
                document_title=doc_title,
                document_year=doc_year,
                source_file=source_file,
                text=all_text,
                cross_references=extract_refs(all_text),
                page_start=1,
                page_end=parsed[-1].page if parsed else 1,
                token_count=est_tokens(all_text),
            ))
        return chunks

    # Build section path hierarchy
    # Track parent headings at each level
    level_stack: dict[int, str] = {}

    for si, sec in enumerate(sections):
        # Determine end: next section start or end of document
        if si + 1 < len(sections):
            end_idx = sections[si + 1].start_idx
        else:
            end_idx = len(parsed)

        # Collect ALL text lines in this section (between boundaries)
        text_lines = []
        pg_start = None
        pg_end = None

        for pl in parsed[sec.start_idx:end_idx]:
            if pl.is_noise:
                continue
            # Strip markdown prefix from text but include everything
            line_text = pl.text
            text_lines.append(line_text)

            if pg_start is None:
                pg_start = pl.page
            pg_end = pl.page

        text = clean_text("\n".join(text_lines))
        if not text.strip():
            continue

        # Build section path
        level_stack[sec.level] = sec.heading
        # Remove any deeper levels from stack
        deeper = [k for k in level_stack if k > sec.level]
        for k in deeper:
            del level_stack[k]

        # Section path = all levels from 1 to current
        path_parts = []
        for lvl in sorted(level_stack.keys()):
            if lvl <= sec.level:
                path_parts.append(level_stack[lvl])
        section_path = " > ".join(path_parts)

        # Determine chunk type
        heading_lower = sec.heading.strip()
        if "ملخص" in heading_lower:
            chunk_type = "summary"
        elif "المدير العام" in heading_lower or "التوقيع" in heading_lower:
            chunk_type = "signature"
        else:
            chunk_type = "section"

        refs = extract_refs(text)
        tok = est_tokens(text)
        if _is_low_value_heading_chunk(text, tok, refs):
            continue

        # Split if oversized
        if tok > max_tokens:
            parts = _split_section(text, max_tokens)
            for pi, part in enumerate(parts, 1):
                part_refs = extract_refs(part)
                part_tok = est_tokens(part)
                if _is_low_value_heading_chunk(part, part_tok, part_refs):
                    continue
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type=chunk_type,
                    document_title=doc_title,
                    document_year=doc_year,
                    source_file=source_file,
                    section_path=section_path,
                    section_number=f"{sec.number}_part{pi}" if sec.number else f"part{pi}",
                    text=part,
                    cross_references=part_refs,
                    page_start=pg_start or 1,
                    page_end=pg_end or 1,
                    token_count=part_tok,
                ))
        else:
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_type=chunk_type,
                document_title=doc_title,
                document_year=doc_year,
                source_file=source_file,
                section_path=section_path,
                section_number=sec.number or None,
                text=text,
                cross_references=refs,
                page_start=pg_start or 1,
                page_end=pg_end or 1,
                token_count=tok,
            ))

    return chunks


def _split_section(text: str, max_tokens: int) -> list[str]:
    """Split oversized section at paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", text)
    if len(paragraphs) <= 1:
        return _hard_split_words(text, max_tokens)

    result = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para.strip()
        if est_tokens(candidate) > max_tokens and current:
            result.append(current.strip())
            current = para.strip()
        else:
            current = candidate
    if current.strip():
        result.append(current.strip())
    safe: list[str] = []
    for part in (result if result else [text]):
        if est_tokens(part) <= max_tokens:
            safe.append(part)
            continue
        for p in _hard_split_by_lines(part.split("\n"), max_tokens):
            if est_tokens(p) <= max_tokens:
                safe.append(p)
            else:
                safe.extend(_hard_split_words(p, max_tokens))
    return [p for p in safe if p.strip()]


def _is_low_value_heading_chunk(text: str, tok: int, refs: list[str]) -> bool:
    """Drop heading-only micro chunks (e.g., isolated 'RESUME')."""
    if tok > 5 or refs:
        return False
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) != 1:
        return False
    line = lines[0]
    # One short heading line without body is low-value for retrieval.
    if len(line) > 48:
        return False
    if re.match(r"^(?:resume|résumé|ملخص)$", line, re.IGNORECASE):
        return True
    if re.match(r"^(?:[A-Z]|[أ-ي]|[IVXLC]+)[\.\-\)]\s+.+$", line):
        return True
    return True


def _hard_split_by_lines(lines_list: list[str], max_tokens: int) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for line in lines_list:
        candidate = "\n".join(cur + [line])
        if cur and est_tokens(candidate) > max_tokens:
            out.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return [p for p in out if p.strip()]


def _hard_split_words(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    cur: list[str] = []
    for w in words:
        candidate = " ".join(cur + [w])
        if cur and est_tokens(candidate) > max_tokens:
            out.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return [p for p in out if p.strip()]


# ─── Main ─────────────────────────────────────────────────────────────────────

def _collect_input_files(input_path: str, input_dir: str | None) -> list[str]:
    """Resolve input HTML files from --input or --input-dir."""
    files: list[str] = []
    if input_dir:
        pattern = str(Path(input_dir) / "**" / "*.html")
        files = sorted(glob.glob(pattern, recursive=True))
    elif input_path:
        # Single file or glob pattern
        if "*" in input_path:
            files = sorted(glob.glob(input_path, recursive=True))
        else:
            files = [input_path]
    return [f for f in files if Path(f).is_file()]


def process_single_file(input_path: str, max_tokens: int) -> list[Chunk]:
    """Process one OCR HTML file and return its chunks."""
    source_file = Path(input_path).as_posix()
    print(f"\n{'='*60}")
    print(f"Processing: {source_file}")
    print(f"{'='*60}")

    parsed, doc_title, doc_year = load_and_parse(input_path)
    if not doc_year:
        path_years = re.findall(r"(19\d{2}|20\d{2})", source_file)
        if path_years:
            doc_year = path_years[-1]
    print(f"  {len(parsed)} lines, title: {doc_title[:80]}")
    print(f"  Detected year: {doc_year or '(not found)'}")

    non_noise = sum(1 for pl in parsed if not pl.is_noise)
    print(f"  {non_noise} content lines")

    # Detect sections
    sections = detect_sections(parsed)
    print(f"  {len(sections)} sections detected:")
    for s in sections:
        print(f"    L{s.level} [{s.number or '-':>5s}] {s.heading[:70]}")

    # Build chunks
    print("\n  Chunking...")
    chunks = build_chunks(parsed, sections, doc_title, doc_year, source_file, max_tokens)
    print(f"  {len(chunks)} chunks")

    # Stats
    tokens = [c.token_count for c in chunks]
    if tokens:
        print(f"  Total tokens: {sum(tokens)}")
        print(f"  min={min(tokens)} max={max(tokens)} avg={sum(tokens)//len(tokens)}")

    # Verify no text dropped
    all_content = "\n".join(pl.text for pl in parsed if not pl.is_noise)
    all_chunks_text = "\n".join(c.text for c in chunks)
    raw_chars = len(clean_text(all_content))
    chunk_chars = len(all_chunks_text)
    ratio = chunk_chars / raw_chars * 100 if raw_chars > 0 else 0
    print(f"  Text retention: {chunk_chars}/{raw_chars} chars ({ratio:.1f}%)")

    return chunks


def run(input_path="ocr_output_notes.html", output_path="chunks_notes.json",
        max_tokens=900, input_dir=None):
    files = _collect_input_files(input_path, input_dir)
    if not files:
        print(f"ERROR: No HTML files found (input={input_path}, input_dir={input_dir})")
        return

    print(f"Found {len(files)} input file(s): {[Path(f).name for f in files]}")

    all_chunks: list[Chunk] = []
    for fpath in files:
        chunks = process_single_file(fpath, max_tokens)
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
                   help="Single HTML file or glob pattern (e.g. 'OCR_Notes/*.html')")
    p.add_argument("--input-dir", default=None,
                   help="Directory containing OCR HTML files (e.g. 'OCR_Notes')")
    p.add_argument("--output", default="chunks_notes.json")
    p.add_argument("--max-tokens", type=int, default=900)
    args = p.parse_args()

    # Default: look for OCR_Notes directory, fall back to single file
    input_path = args.input
    input_dir = args.input_dir
    if not input_path and not input_dir:
        if Path("OCR_Notes").is_dir():
            input_dir = "OCR_Notes"
        else:
            input_path = "ocr_output_notes.html"

    run(input_path=input_path, output_path=args.output,
        max_tokens=args.max_tokens, input_dir=input_dir)


if __name__ == "__main__":
    main()
