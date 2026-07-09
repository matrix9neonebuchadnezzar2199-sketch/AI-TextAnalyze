"""Tests for NER aggregation helpers."""

from backend.ner_engine import aggregate_keywords


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
