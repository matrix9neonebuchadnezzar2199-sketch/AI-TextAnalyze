#!/usr/bin/env python3
"""Extract + translate fixtures under 翻訳テスト記事 for iterative QA."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.model_manager import ModelManager  # noqa: E402
from backend.pdf_reader import read_text_file  # noqa: E402

ARTICLE_DIR = ROOT / "翻訳テスト記事"
EXTRACTED = ROOT / "honnyakutesuto" / "_extracted" / "articles"
OUT = ROOT / "honnyakutesuto" / "_results" / "articles"

# (slug, src_lang, filename substring to match)
FIXTURES = [
    ("ru_tass_telegram", "ru", "ロシア語"),
    ("zh_people_ai", "zh", "中国語"),
    ("en_mainichi_disaster", "en", "英語"),
    ("ko_hani_hormuz", "ko", "韓国語"),
]


def find_pdf(marker: str) -> Path:
    for path in ARTICLE_DIR.glob("*.pdf"):
        if marker in path.name:
            return path
    raise FileNotFoundError(f"No PDF matching {marker!r} in {ARTICLE_DIR}")


def extract_all() -> dict[str, Path]:
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for slug, _lang, marker in FIXTURES:
        pdf = find_pdf(marker)
        text = read_text_file(pdf)
        dest = EXTRACTED / f"{slug}.txt"
        dest.write_text(text, encoding="utf-8")
        mapping[slug] = dest
        print(f"extracted {slug}: {len(text)} chars from {pdf.name}", flush=True)
    return mapping


def heuristic_score(src: str, tgt: str, lang: str) -> dict:
    """Cheap offline quality signals (not BLEU). Lower bad_* is better."""
    src_lines = [ln for ln in src.splitlines() if ln.strip()]
    tgt_lines = [ln for ln in tgt.splitlines() if ln.strip()]
    # 原文がほぼそのまま残っている割合（CJK以外の長い英単語連続など）
    latin_runs = re.findall(r"[A-Za-z]{8,}", tgt)
    # 明らかな崩壊パターン
    bad_patterns = [
        (r"(.)\1{6,}", "char_repeat"),
        (r"(?:[\u3040-\u30ff] ){8,}", "spaced_kana"),
        (r"(?:[\u4e00-\u9fff] ){8,}", "spaced_cjk"),
        (r"について。\s*について", "ni_tsuite_loop"),
        (r"\(。", "broken_paren"),
        (r"。\s*。", "double_period"),
    ]
    bad_hits: dict[str, int] = {}
    for pat, name in bad_patterns:
        bad_hits[name] = len(re.findall(pat, tgt))

    # URL / Telegram リンク保持
    src_urls = set(re.findall(r"https?://\S+|t\.me/\S+", src, flags=re.I))
    kept_urls = sum(1 for u in src_urls if u in tgt)

    # ハングル／キリル／簡体字の大量残留（訳し漏れの粗い指標）
    residual = {
        "hangul": len(re.findall(r"[\uac00-\ud7a3]", tgt)),
        "cyrillic": len(re.findall(r"[\u0400-\u04ff]", tgt)),
        "han": len(re.findall(r"[\u4e00-\u9fff]", tgt)),
    }
    # JA 訳では漢字は正常なので han は zh 以外では減点しない
    residual_penalty = 0
    if lang == "ko":
        residual_penalty = residual["hangul"]
    elif lang == "ru":
        residual_penalty = residual["cyrillic"]
    elif lang == "zh":
        # 訳文に簡体字が極端に多い＝未翻訳ブロック疑い（閾値は相対）
        residual_penalty = max(0, residual["han"] - len(re.findall(r"[\u3040-\u30ff]", tgt)))

    ratio = (len(tgt) / max(len(src), 1))
    return {
        "src_chars": len(src),
        "tgt_chars": len(tgt),
        "src_lines": len(src_lines),
        "tgt_lines": len(tgt_lines),
        "latin_long_tokens": len(latin_runs),
        "url_src": len(src_urls),
        "url_kept": kept_urls,
        "bad_hits": bad_hits,
        "bad_total": sum(bad_hits.values()),
        "residual_penalty": residual_penalty,
        "len_ratio": round(ratio, 3),
    }


def translate_one(
    slug: str,
    lang: str,
    src_path: Path,
    *,
    mt: str,
    tag: str,
    max_chars: int,
) -> dict:
    text = src_path.read_text(encoding="utf-8")
    if max_chars > 0:
        text = text[:max_chars]
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}_src.txt").write_text(text, encoding="utf-8")

    mgr = ModelManager()
    mgr.set_selected_mt(mt)
    engine = mgr.load_mt()
    t0 = time.time()
    result = engine.translate(text, lang, "ja")
    elapsed = time.time() - t0
    mgr.unload_all()

    ja_path = out_dir / f"{slug}_ja.txt"
    ja_path.write_text(result.text, encoding="utf-8")
    units = "\n---\n".join(
        f"[{u['id']}]\nSRC: {u['src']}\nTGT: {u['tgt']}" for u in result.units
    )
    (out_dir / f"{slug}_units.txt").write_text(units, encoding="utf-8")
    score = heuristic_score(text, result.text, lang)
    score.update(
        {
            "slug": slug,
            "lang": lang,
            "mt": mt,
            "tag": tag,
            "elapsed_s": round(elapsed, 1),
            "units": len(result.units),
        }
    )
    (out_dir / f"{slug}_score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{tag}] {slug} {lang}->ja chars={len(text)}->{len(result.text)} "
        f"units={len(result.units)} {elapsed:.1f}s bad={score['bad_total']} "
        f"residual={score['residual_penalty']}",
        flush=True,
    )
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--mt", default="nllb-1.3b", choices=["nllb-600m", "nllb-1.3b"])
    parser.add_argument("--tag", default="r0")
    parser.add_argument("--max-chars", type=int, default=0, help="0=full")
    parser.add_argument("--only", choices=["ru", "zh", "en", "ko"], default=None)
    args = parser.parse_args()

    mapping = extract_all()
    if args.extract_only:
        return 0

    scores = []
    for slug, lang, _marker in FIXTURES:
        if args.only and lang != args.only:
            continue
        scores.append(
            translate_one(
                slug,
                lang,
                mapping[slug],
                mt=args.mt,
                tag=args.tag,
                max_chars=args.max_chars,
            )
        )

    summary = OUT / args.tag / "summary.json"
    summary.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
