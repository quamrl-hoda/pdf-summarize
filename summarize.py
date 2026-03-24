"""
PDF Summarizer CLI
------------------
Usage:
    python summarize.py path/to/file.pdf
    python summarize.py path/to/file.pdf --mode brief
    python summarize.py path/to/file.pdf --mode detailed
    python summarize.py path/to/file.pdf --mode bullets
    python summarize.py path/to/file.pdf --chunk-size 800 --model gpt-4o-mini
"""

import argparse
import sys
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass

import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# load env
load_dotenv()

DEFAULT_CHUNK_SIZE   = 1000   # words per chunk
DEFAULT_CHUNK_OVERLAP = 100   # word overlap between chunks
DEFAULT_MODEL        = "gpt-4o-mini"
MAX_CHUNK_SUMMARY_TOKENS = 400
MAX_FINAL_SUMMARY_TOKENS = 1200

SUMMARY_PROMPTS = {
    "brief": (
        "You are an expert summarizer. Given a portion of a document, "
        "write a concise 2-3 sentence summary capturing the main point."
    ),
    "detailed": (
        "You are an expert summarizer. Given a portion of a document, "
        "write a thorough summary that covers all key ideas, arguments, and findings."
    ),
    "bullets": (
        "You are an expert summarizer. Given a portion of a document, "
        "extract the key points as a concise bullet list (max 5 bullets)."
    ),
}

FINAL_PROMPTS = {
    "brief": (
        "You are an expert summarizer. Below are summaries of individual sections "
        "of a document. Synthesise them into a single, coherent paragraph summary "
        "of the entire document. Be concise."
    ),
    "detailed": (
        "You are an expert summarizer. Below are summaries of individual sections "
        "of a document. Synthesise them into a detailed, well-structured summary "
        "covering all major themes, arguments, and conclusions."
    ),
    "bullets": (
        "You are an expert summarizer. Below are bullet-point summaries of individual "
        "sections of a document. Merge and deduplicate them into a final bullet-point "
        "list of the most important takeaways from the entire document (max 10 bullets)."
    ),
}


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class PageText:
    page_num: int
    text: str

@dataclass
class Chunk:
    index: int
    text: str
    word_count: int
    page_start: int
    page_end: int


# ─────────────────────────────────────────────
# Step 1 – Extract text from PDF
# ─────────────────────────────────────────────

def extract_text(pdf_path: Path) -> list[PageText]:
    """Extract text from each page using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"  → {total} page(s) detected")
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            # Normalise whitespace
            cleaned = re.sub(r"\s+", " ", raw).strip()
            if cleaned:
                pages.append(PageText(page_num=i, text=cleaned))
    return pages


# ─────────────────────────────────────────────
# Step 2 – Chunk the text
# ─────────────────────────────────────────────

def chunk_pages(
    pages: list[PageText],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Word-level sliding window chunker.
    Preserves page-number metadata per chunk.
    """
    # Flatten all pages into (word, page_num) pairs
    word_source: list[tuple[str, int]] = []
    for p in pages:
        for word in p.text.split():
            word_source.append((word, p.page_num))

    if not word_source:
        return []

    chunks = []
    start = 0
    idx = 0

    while start < len(word_source):
        end = min(start + chunk_size, len(word_source))
        slice_ = word_source[start:end]

        text = " ".join(w for w, _ in slice_)
        page_start = slice_[0][1]
        page_end   = slice_[-1][1]

        chunks.append(Chunk(
            index=idx,
            text=text,
            word_count=len(slice_),
            page_start=page_start,
            page_end=page_end,
        ))

        idx += 1
        # Move forward by (chunk_size - overlap) words
        start += max(1, chunk_size - overlap)

    return chunks


# ─────────────────────────────────────────────
# Step 3 – Map: summarise each chunk
# ─────────────────────────────────────────────

def summarise_chunk(
    client: OpenAI,
    chunk: Chunk,
    mode: str,
    model: str,
) -> str:
    """Call the LLM to summarise a single chunk."""
    system_prompt = SUMMARY_PROMPTS[mode]
    user_message  = f"Document section (pages {chunk.page_start}–{chunk.page_end}):\n\n{chunk.text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=MAX_CHUNK_SUMMARY_TOKENS,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def map_summaries(
    client: OpenAI,
    chunks: list[Chunk],
    mode: str,
    model: str,
) -> list[str]:
    """Summarise all chunks (map step)."""
    summaries = []
    for chunk in chunks:
        label = f"Chunk {chunk.index + 1}/{len(chunks)} (p{chunk.page_start}–p{chunk.page_end}, {chunk.word_count} words)"
        print(f"  → Summarising {label} ...", end=" ", flush=True)
        t0 = time.time()
        summary = summarise_chunk(client, chunk, mode, model)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        summaries.append(summary)
    return summaries


# ─────────────────────────────────────────────
# Step 4 – Reduce: merge chunk summaries
# ─────────────────────────────────────────────

def reduce_summaries(
    client: OpenAI,
    summaries: list[str],
    mode: str,
    model: str,
) -> str:
    """Combine all chunk summaries into one final summary (reduce step)."""
    combined = "\n\n---\n\n".join(
        f"[Section {i + 1}]\n{s}" for i, s in enumerate(summaries)
    )
    system_prompt = FINAL_PROMPTS[mode]
    user_message  = f"Section summaries:\n\n{combined}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=MAX_FINAL_SUMMARY_TOKENS,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Step 5 – Print formatted output
# ─────────────────────────────────────────────

DIVIDER = "─" * 60

def print_report(
    pdf_path: Path,
    pages: list[PageText],
    chunks: list[Chunk],
    final_summary: str,
    mode: str,
    model: str,
    elapsed: float,
) -> None:
    total_words = sum(c.word_count for c in chunks)
    print(f"\n{DIVIDER}")
    print("  PDF SUMMARY REPORT")
    print(DIVIDER)
    print(f"  File   : {pdf_path.name}")
    print(f"  Pages  : {len(pages)}")
    print(f"  Words  : ~{total_words:,}")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Mode   : {mode}")
    print(f"  Model  : {model}")
    print(f"  Time   : {elapsed:.1f}s")
    print(DIVIDER)
    print()
    print(final_summary)
    print(f"\n{DIVIDER}\n")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise a PDF using a map-reduce LLM pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--mode",
        choices=["brief", "detailed", "bullets"],
        default="brief",
        help="Summary style (default: brief)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="WORDS",
        help=f"Words per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        metavar="WORDS",
        help=f"Overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate PDF path ──
    if not args.pdf.exists():
        print(f"[ERROR] File not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)
    if args.pdf.suffix.lower() != ".pdf":
        print(f"[ERROR] Not a PDF file: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    # ── Validate API key ──
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("        Run: export OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    t_start = time.time()

    # ── Step 1: Extract ──
    print(f"\n[1/4] Extracting text from '{args.pdf.name}' ...")
    pages = extract_text(args.pdf)
    if not pages:
        print("[ERROR] No extractable text found. The PDF may be scanned/image-only.", file=sys.stderr)
        print("        Tip: add --ocr support with pytesseract for scanned PDFs.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Chunk ──
    print(f"\n[2/4] Chunking text (chunk_size={args.chunk_size}, overlap={args.overlap}) ...")
    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"  → {len(chunks)} chunk(s) created")

    # ── Step 3: Map ──
    print(f"\n[3/4] Summarising chunks (mode='{args.mode}', model='{args.model}') ...")
    chunk_summaries = map_summaries(client, chunks, args.mode, args.model)

    # ── Step 4: Reduce ──
    print("\n[4/4] Merging chunk summaries into final summary ...")
    if len(chunk_summaries) == 1:
        # No need to reduce a single chunk
        final_summary = chunk_summaries[0]
    else:
        final_summary = reduce_summaries(client, chunk_summaries, args.mode, args.model)

    elapsed = time.time() - t_start

    # ── Print report ──
    print_report(args.pdf, pages, chunks, final_summary, args.mode, args.model, elapsed)


if __name__ == "__main__":
    main()