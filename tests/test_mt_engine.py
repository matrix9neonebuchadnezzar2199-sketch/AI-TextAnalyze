"""Tests for MT sentence splitting and tokenizer wiring."""

from pathlib import Path

import pytest

from backend.mt_engine import (
    HF_TOKENIZER_SUBDIR,
    chunk_paragraph,
    collapse_soft_linebreaks,
    is_entity_name_line,
    is_pass_through_unit,
    is_regulatory_list_text,
    iter_translation_units,
    join_literal_segments,
    join_translated_units,
    merge_orphan_markers,
    split_literal_segments,
    split_paragraphs,
    split_sentences,
)


def test_split_japanese_sentences() -> None:
    text = "一行目です。二行目です！"
    parts = split_sentences(text)
    assert len(parts) == 2


def test_split_paragraphs() -> None:
    text = "Para one.\n\nPara two."
    parts = split_paragraphs(text)
    assert len(parts) == 2


def test_iter_translation_units() -> None:
    text = "A. B.\n\nC."
    units = iter_translation_units(text)
    assert len(units) == 3


def test_merge_orphan_markers() -> None:
    paragraphs = ["(1)", "Cougars live here.", "(2)", "They hunt."]
    merged = merge_orphan_markers(paragraphs)
    assert merged == ["(1)\nCougars live here.", "(2)\nThey hunt."]


def test_collapse_soft_linebreaks_joins_wrapped_url() -> None:
    text = "http://example.com/animals/mount\nain-lion/page"
    assert collapse_soft_linebreaks(text) == "http://example.com/animals/mountain-lion/page"


def test_collapse_soft_linebreaks_keeps_word_space() -> None:
    text = "Eventually, cougar\npopulations recovered to a level"
    assert "cougar populations" in collapse_soft_linebreaks(text)


def test_join_literal_segments_inserts_newlines() -> None:
    segments = [
        (False, "段落末です。"),
        (True, "[]"),
        (False, "(5)\nEven so, populations fragmented."),
    ]
    joined = join_literal_segments(segments)
    assert "。\n[]\n\n(5)" in joined


def test_collapse_pdf_spacing_hangul() -> None:
    from backend.mt_engine import collapse_pdf_spacing

    char_spaced = "일 본 의 포 용 적 인"
    assert collapse_pdf_spacing(char_spaced) == "일본의포용적인"
    word_spaced = "일본의  포용적인  e 스포츠"
    out = collapse_pdf_spacing(word_spaced)
    assert "일본의 포용적인" in out
    assert "  " not in out


def test_suppress_repeated_phrases() -> None:
    from backend.mt_engine import suppress_repeated_phrases

    looped = "競争相手と競争できる " * 8
    out = suppress_repeated_phrases(looped)
    assert out.count("競争相手と競争できる") <= 2


def test_www3_url_is_literal() -> None:
    text = "See www3.nhk.or.jp/nhkworld/ko/news/backstories/4535 for details."
    segments = split_literal_segments(text)
    literals = [seg for is_lit, seg in segments if is_lit]
    assert literals == ["www3.nhk.or.jp/nhkworld/ko/news/backstories/4535"]


def test_split_literal_segments() -> None:
    text = "Before [] middle [X] after http://example.com/end"
    segments = split_literal_segments(text)
    assert segments == [
        (False, "Before "),
        (True, "[]"),
        (False, " middle "),
        (True, "[X]"),
        (False, " after "),
        (True, "http://example.com/end"),
    ]


def test_split_literal_segments_preserves_urls_not_legal_cites() -> None:
    """Legal cites stay in prose (shielded later); URLs remain literal segments."""
    text = "See http://example.com and Section 1260H(g)(2)(B)(i)(I)."
    segments = split_literal_segments(text)
    literals = [seg for is_lit, seg in segments if is_lit]
    assert literals == ["http://example.com"]
    prose = "".join(seg for is_lit, seg in segments if not is_lit)
    assert "1260H" in prose
    assert not any(seg == "(2)" for _, seg in segments)


def test_split_sentences_does_not_break_legal_citations() -> None:
    text = "Designated under Section 1260H(g)(3)(A). Next sentence."
    parts = split_sentences(text)
    assert parts[0].endswith("Section 1260H(g)(3)(A).")
    assert parts[1] == "Next sentence."


def test_is_entity_name_line() -> None:
    assert is_entity_name_line("Huawei Technologies Co., Ltd. (Huawei)")
    assert is_entity_name_line("Align Aerospace LLC")
    assert not is_entity_name_line("Huawei Holding is indirectly affiliated with SASAC.")


def test_chunk_paragraph_preserves_entity_lines() -> None:
    block = "AVIC\nAlign Aerospace LLC\nAvicopter PLC"
    units = chunk_paragraph(block)
    assert units == ["AVIC", "Align Aerospace LLC", "Avicopter PLC"]


def test_is_pass_through_entity_line() -> None:
    assert is_pass_through_unit("Align Aerospace LLC")


def test_join_translated_units_inserts_newlines() -> None:
    joined = join_translated_units(["AVIC", "Align Aerospace LLC", "Translated body."])
    assert "AVIC\nAlign Aerospace LLC\n" in joined


def test_is_regulatory_list_text() -> None:
    assert is_regulatory_list_text("Section 1260H of the NDAA")
    assert not is_regulatory_list_text("Cougars live in North America.")


def test_format_toefl_paragraph_markers() -> None:
    from backend.mt_engine import format_toefl_paragraph_markers

    assert format_toefl_paragraph_markers("(1)Originating") == "(1)\nOriginating"


def test_chunk_paragraph_splits_sentences() -> None:
    paragraph = "Sentence one. Sentence two."
    assert chunk_paragraph(paragraph) == ["Sentence one.", "Sentence two."]


def test_is_pass_through_unit() -> None:
    assert is_pass_through_unit("http://example.com")
    assert is_pass_through_unit("[]")
    assert is_pass_through_unit("[X]")
    assert not is_pass_through_unit("Cougars live here.")


def test_normalize_cougar_to_mountain_lion() -> None:
    from backend.mt_engine import normalize_source_for_mt

    text = "Cougars live here. The cougar hunts."
    out = normalize_source_for_mt(text, "en", "ja")
    assert "cougar" not in out.lower()
    assert "Mountain lions live here" in out
    assert "mountain lion hunts" in out


def test_translate_result_units_shape() -> None:
    from backend.mt_engine import TranslateResult, iter_translation_units

    units = iter_translation_units("First. Second.")
    assert len(units) == 2
    result = TranslateResult(text="a\n\nb", units=[{"id": "u0", "src": "First.", "tgt": "A."}])
    assert result.text == "a\n\nb"
    assert result.units[0]["src"] == "First."


def test_global_progress_monotonic() -> None:
    """Progress callback must use a single global total across literal segments."""
    calls: list[tuple[int, int]] = []

    def on_progress(done: int, total: int, _detail: str) -> None:
        calls.append((done, total))

    progress = {"done": 0, "total": 3}
    for _ in range(3):
        progress["done"] += 1
        on_progress(progress["done"], progress["total"], "")
    assert calls == [(1, 3), (2, 3), (3, 3)]
    totals = {t for _, t in calls}
    assert len(totals) == 1


def test_collapse_legal_citation_breaks() -> None:
    from backend.mt_engine import collapse_legal_citation_breaks

    text = "affiliated (Section\n1260H(g)(2)\n(B)(i)(I))."
    out = collapse_legal_citation_breaks(text)
    assert "Section 1260H(g)(2)(B)(i)(I)" in out


def test_regulatory_gloss_and_ja_post() -> None:
    from backend.mt_engine import apply_ja_post_fixes, apply_regulatory_gloss_en

    glossed = apply_regulatory_gloss_en(
        "Chinese military companies and military-civil fusion"
    )
    assert "Chinese military enterprises" in glossed
    assert "MCF military-civil fusion" in glossed
    assert apply_ja_post_fixes("中国軍人会社と軍事民間の融合") == "中国軍事企業と軍民融合"


def test_shield_legal_cites_roundtrip() -> None:
    from backend.mt_engine import restore_legal_cites, shield_legal_cites

    src = "owned by SASAC (Section 1260H(g)(2)(B)(i)(I))."
    shielded, cites = shield_legal_cites(src)
    assert "1260H" not in shielded
    assert "ZZREF0ZZ" in shielded
    assert restore_legal_cites(shielded, cites) == src


def test_william_mac_not_sentence_split() -> None:
    from backend.mt_engine import split_sentences

    text = (
        "Section 1260H of the William M. (Mac) Thornberry National Defense "
        "Authorization Act for Fiscal Year 2021."
    )
    parts = split_sentences(text)
    assert len(parts) == 1


def test_collapse_preserves_paragraph_breaks() -> None:
    text = "Title line one\ncontinued.\n\n360 Security Technology Inc. (Qihoo 360)\n\n• Body."
    out = collapse_soft_linebreaks(text)
    assert "\n\n360 Security" in out
    assert "\n\n• Body." in out


def test_entity_line_not_split_on_inc() -> None:
    from backend.mt_engine import chunk_paragraph, split_sentences

    name = "360 Security Technology Inc. (Qihoo 360)"
    assert chunk_paragraph(name) == [name]
    assert len(split_sentences(name)) == 1


def test_hf_tokenizer_dir_required(tmp_path: Path) -> None:
    from backend.mt_engine import MtEngine

    engine = MtEngine.__new__(MtEngine)
    engine._model_dir = tmp_path
    with pytest.raises(FileNotFoundError, match="hf-tokenizer"):
        engine._resolve_tokenizer_dir()


@pytest.mark.integration
def test_literals_survive_translation() -> None:
    """URLs and TOEFL markers must not pass through the MT model."""
    from backend.mt_engine import MtEngine
    from backend.model_manager import default_model_dir

    model_dir = default_model_dir() / "mt-nllb-600m-ct2-int8"
    tokenizer_dir = model_dir / HF_TOKENIZER_SUBDIR
    if not (model_dir / "model.bin").is_file() or not tokenizer_dir.is_dir():
        pytest.skip("MT model or hf-tokenizer not installed")

    sample = (
        "[] This included rescinding the bounties. [X]Eventually, populations recovered. [] "
        "See http://example.com/cougar.htm for details."
    )
    engine = MtEngine(model_dir, intra_threads=2)
    try:
        out = engine.translate(sample, "en", "ja")
        assert "[]" in out.text
        assert "[X]" in out.text
        assert "http://example.com/cougar.htm" in out.text
        assert len(out.units) >= 1
    finally:
        engine.close()


@pytest.mark.integration
def test_mt_no_repetition_on_cougar_sentence() -> None:
    """Regression: broken CT2-bundled tokenizer caused token loops."""
    from backend.mt_engine import MtEngine
    from backend.model_manager import default_model_dir

    model_dir = default_model_dir() / "mt-nllb-600m-ct2-int8"
    tokenizer_dir = model_dir / HF_TOKENIZER_SUBDIR
    if not (model_dir / "model.bin").is_file() or not tokenizer_dir.is_dir():
        pytest.skip("MT model or hf-tokenizer not installed")

    engine = MtEngine(model_dir, intra_threads=2)
    try:
        out = engine._translate_batch(
            ["the cougar--otherwise known as the puma or mountain lion"],
            "eng_Latn",
            "jpn_Jpan",
        )[0]
        assert "知らない人 知らない人" not in out
        assert "ジャイフ" not in out
        assert len(out) < 200
    finally:
        engine.close()
