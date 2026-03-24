"""
Unit tests for the PDF summarizer pipeline.
Run with: pytest tests/ -v
"""

import pytest
from summarize import (
    PageText,
    Chunk,
    chunk_pages,
)


# ─────────────────────────────────────────────
# chunk_pages tests
# ─────────────────────────────────────────────

def make_page(page_num: int, word_count: int) -> PageText:
    """Helper: create a PageText with `word_count` dummy words."""
    text = " ".join(f"word{i}" for i in range(word_count))
    return PageText(page_num=page_num, text=text)


def test_chunk_pages_single_page_fits_in_one_chunk():
    pages = [make_page(1, 50)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].word_count == 50


def test_chunk_pages_creates_multiple_chunks():
    pages = [make_page(1, 500)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=0)
    assert len(chunks) == 5
    for c in chunks:
        assert c.word_count == 100


def test_chunk_pages_overlap_produces_extra_chunks():
    # 200 words, chunk 100, overlap 50 → stride=50 → 4 chunks
    # starts: 0, 50, 100, 150  (150+100=250 > 200, so last chunk is partial)
    pages = [make_page(1, 200)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=50)
    assert len(chunks) == 4


def test_chunk_pages_preserves_page_numbers():
    pages = [make_page(1, 100), make_page(2, 100)]
    chunks = chunk_pages(pages, chunk_size=150, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].page_start == 1
    assert chunks[1].page_end == 2


def test_chunk_pages_empty_input():
    assert chunk_pages([]) == []


def test_chunk_pages_empty_page_text():
    pages = [PageText(page_num=1, text="")]
    chunks = chunk_pages(pages, chunk_size=100, overlap=0)
    assert chunks == []


def test_chunk_pages_indices_are_sequential():
    pages = [make_page(1, 300)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=0)
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_pages_last_chunk_smaller():
    pages = [make_page(1, 250)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=0)
    # 3 chunks: 100 + 100 + 50
    assert len(chunks) == 3
    assert chunks[-1].word_count == 50


def test_chunk_pages_text_reconstructs_words():
    pages = [make_page(1, 10)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=0)
    words = chunks[0].text.split()
    assert words == [f"word{i}" for i in range(10)]