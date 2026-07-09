"""Tests for NER aggregation helpers."""

from backend.ner_engine import aggregate_keywords, chunk_text


def test_aggregate_keywords_merges_duplicates() -> None:
    raw = [
        {"term": "Tokyo", "type": "city", "freq": 1},
        {"term": "tokyo", "type": "city", "freq": 1},
        {"term": "Japan", "type": "country", "freq": 1},
    ]
    result = aggregate_keywords(raw)
    city = next(r for r in result if r["type"] == "city")
    assert city["freq"] == 2
    assert len(result) == 2


def test_chunk_text_splits_long_document() -> None:
    long_text = "Word " * 800
    chunks = chunk_text(long_text, max_chars=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
