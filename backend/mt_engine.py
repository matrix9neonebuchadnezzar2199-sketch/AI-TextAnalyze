"""NLLB CTranslate2 machine translation (CPU only)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TranslateResult:
    """Aligned translation output for UI highlighting."""

    text: str
    units: list[dict[str, Any]] = field(default_factory=list)

NLLB_LANG_CODES: dict[str, str] = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "ko": "kor_Hang",
}

# CT2 同梱 tokenizer.json は NLLB 用ではなく誤トークン化するため、HF 正規 tokenizer を別途配置する
NLLB_HF_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"
HF_TOKENIZER_SUBDIR = "hf-tokenizer"

ProgressCallback = Callable[[int, int, str], None]

# CT2 はバッチ推論の方が逐次より大幅に速い
TRANSLATE_BATCH_SIZE = 16
MAX_PARAGRAPH_CHARS = 1000
MAX_SENTENCE_CHARS = 900
# 段落丸ごと翻訳は長文で出力が途中打ち切りになるため、常に文単位を優先
MAX_WHOLE_PARAGRAPH_CHARS = 0

_URL_RE = re.compile(
    r"https?://[^\s\]\)<>]+|(?:www\d*\.)[^\s\]\)<>]+",
    re.IGNORECASE,
)
_ORPHAN_MARKER_RE = re.compile(r"^\(\d+\)$")
_TOEFL_MARKER_RE = re.compile(r"^\[\s*\]$|^\[X\]$", re.IGNORECASE)
_TOEFL_PARA_ONLY_RE = re.compile(r"^\(\d{1,2}\)$")
_PARA_NUM_START_RE = re.compile(r"^\(\d+\)")
# 1260H 引用: Section 1260H(g)(2)(B)(i)(I) および "and (g)(3)(B)(iv)" 継続
_LEGAL_CITE_INLINE_RE = re.compile(
    r"(?:Sections?\s+)?1260H(?:\([^)]*\))+(?:\s+and\s+(?:\([^)]*\))+)*"
    r"|10\s+U\.S\.C\.\s*§\s*[^,\n;]+"
    r"|Public Law \d+-\d+",
    re.IGNORECASE,
)
_ENTITY_SUFFIX_RE = re.compile(
    r"\b(Inc\.|Ltd\.|Limited|Corporation|Corp\.|LLC|PLC|Co\.|Group)\b",
    re.IGNORECASE,
)
# URL / Telegram / スキーム無しドメイン / ページ番号は翻訳しない
_LITERAL_SPLIT_RE = re.compile(
    r"("
    r"https?://[^\s\]\)<>]+"
    r"|(?:www\d*\.)[^\s\]\)<>]+"
    r"|t\.me(?:/[^\s\]\)<>]*)?"
    r"|\b(?:[\w-]+\.)+(?:com\.cn|co\.jp|co\.kr|com|org|net|info|io|cn|jp|ru)"
    r"(?:/[^\s\]\)<>]*)?"
    r"|\[\s*\]|\[X\]|\[END\]|-{5,}"
    r"|\b\d{1,2}/\d{1,2}\b"
    r")",
    re.IGNORECASE,
)
# EN→JA 法令文書向け用語（翻訳前に英語側を明確化）
_REGULATORY_GLOSS_EN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bChinese military companies\b", re.I), "Chinese military enterprises"),
    (re.compile(r"\bChinese military company\b", re.I), "Chinese military enterprise"),
    (re.compile(r"\bmilitary[-\s]civil fusion\b", re.I), "MCF military-civil fusion"),
    (re.compile(r"\bDeputy Secretary of Defense\b"), "US Deputy Defense Secretary"),
    (
        re.compile(r"\bMinistry of State Security\b(?!\s*\(MSS\))"),
        "China Ministry of State Security (MSS)",
    ),
    (
        re.compile(r"\bMinistry of Industry and Information Technology\b(?!\s*\(MIIT\))"),
        "China Ministry of Industry and Information Technology (MIIT)",
    ),
    (
        re.compile(r"\bNational Defense Authorization Act\b"),
        "National Defense Authorization Act (NDAA)",
    ),
    (re.compile(r"\bdefense industrial base\b", re.I), "defense industry base"),
    (re.compile(r"\bLittle Giant\b"), "Little Giant designated SME"),
]
# EN ニュース／一般記事向け（固有名・地形の定訳誘導）
_NEWS_GLOSS_EN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bShigeru Ishiba\b"), "Ishiba Shigeru"),
    (re.compile(r"\bSanae Takaichi\b"), "Takaichi Sanae"),
    (re.compile(r"\bDigital Agency\b"), "Japan Digital Agency"),
    (re.compile(r"\bNankai Trough\b"), "Nankai Trough megathrust zone"),
    (re.compile(r"\bChishima Trench\b"), "Chishima Kuril Trench"),
    (re.compile(r"\bJapan Trench\b"), "Japan Trench deep-sea trench"),
    (re.compile(r"\bHouse of Councillors\b"), "Japanese House of Councillors upper house"),
    (re.compile(r"\bdisaster management agency\b", re.I), "Disaster Management Agency"),
    (re.compile(r"\bdisaster management minister\b", re.I), "Minister for Disaster Management"),
]
# KO→JA: 誤訳が多い語は「日本語に近い英語定訳」へ寄せ、漏れは JA 後処理で拾う
_NEWS_GLOSS_KO: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"호르무즈"), "Hormuz"),
    (re.compile(r"자처한"), "who proclaimed himself as"),
    (re.compile(r"통행료"), "passage toll"),
    (re.compile(r"\b내라\b"), "must pay"),
    (re.compile(r"홍해"), "the Red Sea"),
    (re.compile(r"후티"), "Houthi rebels"),
    (re.compile(r"양해각서"), "memorandum of understanding (MOU)"),
    (re.compile(r"해상봉쇄"), "naval blockade"),
    (re.compile(r"일방\s*선언"), "unilateral announcement"),
    (re.compile(r"수호자"), "guardian"),
]
# ZH→JA: 定訳が崩れやすい政治／AI用語
_NEWS_GLOSS_ZH: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"全球南方"), "Global South"),
    (re.compile(r"人工智能治理"), "AI governance"),
    (re.compile(r"全球人工智能治理"), "global AI governance"),
    (re.compile(r"主旨讲话"), "keynote speech"),
    (re.compile(r"高级别会议"), "high-level meeting"),
    (re.compile(r"世界人工智能大会"), "World Artificial Intelligence Conference"),
]
# RU→JA: 固有名・略称の安定化
_NEWS_GLOSS_RU: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ТАСС-ДОСЬЕ"), "TASS dossier"),
    (re.compile(r"\bТАСС\b"), "TASS"),
    (re.compile(r"Ормузск(?:ий|ого|ом|ий)?\s+пролив", re.I), "Strait of Hormuz"),
    (
        re.compile(r"Корпуса стражей исламской революции"),
        "IRGC Islamic Revolutionary Guard Corps",
    ),
    (re.compile(r"\bКСИР\b"), "IRGC"),
]
# 出力側の既知誤訳を後処理で矯正
_JA_POST_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"中国軍人(?:会社|企業)"), "中国軍事企業"),
    (re.compile(r"中国軍企社"), "中国軍事企業"),
    (re.compile(r"軍事民間(?:の)?融合"), "軍民融合"),
    (re.compile(r"MCF\s*軍民融合"), "軍民融合"),
    (re.compile(r"副国防総理"), "国防副長官"),
    (re.compile(r"国家安全保障省"), "国家安全部"),
    (re.compile(r"産業情報技術省"), "工業情報化部"),
    (re.compile(r"国防許可法"), "国防授権法"),
    (re.compile(r"関連事項について"), ""),
    (re.compile(r"オートル・テクノロジー"), "Autel Technology"),
    (re.compile(r"オートルロボティクス|オートルロボット工学"), "Autel Robotics"),
    (re.compile(r"小さな巨人指定中小企業"), "Little Giant"),
    (re.compile(r"\)\s*について。"), ")。"),
    (re.compile(r"\(。"), "（"),
    (re.compile(r"。\s*。"), "。"),
    (re.compile(r"\.\s*。"), "。"),
    (re.compile(r"\(Autel Technology\)\s*は"), "は"),
    # ニュース記事で観測された系統誤訳
    (re.compile(r"石原(?:市長)?(?:元)?首相"), "石破茂元首相"),
    (re.compile(r"石原市長"), "石破茂"),
    (re.compile(r"石原一郎"), "石破茂"),
    (re.compile(r"サナエ(?:・タカイチ)?首相|サナエ・タカイチ"), "高市早苗首相"),
    (re.compile(r"南井谷|南カイのメガトラストゾーン|南カイ・メガトラスト[^。、]*"), "南海トラフ"),
    (re.compile(r"チシマ[・･\s]*キュリル[沟溝]?|チシマ海峡"), "千島海溝"),
    (re.compile(r"デジタル機関"), "デジタル庁"),
    (re.compile(r"トキオ"), "東京"),
    (re.compile(r"\bAgency\b"), "機関"),
    (re.compile(r"洪荒|赤海"), "紅海"),
    (re.compile(r"自粛した"), "自称した"),
    (re.compile(r"通行料\s*20%\s*下降"), "通行料20%を支払え"),
    (re.compile(r"20%\s*減免"), "20%課金"),
    (re.compile(r"タース(?:・ドシエ)?"), "TASS"),
    (re.compile(r"ホーティド|ホーティ|フーシー"), "フーティ"),
    (re.compile(r"ホルミューズ|ホルマス"), "ホルムズ"),
    (re.compile(r"人工智能"), "人工知能"),
    (re.compile(r"治理"), "ガバナンス"),
    (re.compile(r"人々が。\s*コムです"), ""),
    (re.compile(r"政治について。"), ""),
    (re.compile(r"t形式です\.me"), "t.me"),
    (re.compile(r"t\s*形式です\s*\.\s*me"), "t.me"),
    (re.compile(r"unilateral declaration"), "一方的な宣言"),
    (re.compile(r"unilateral announcement"), "一方的な宣言"),
    (re.compile(r"トランジット[・･\s]*タール"), "通行料"),
    (re.compile(r"トランジット[・･\s]*トール"), "通行料"),
    (re.compile(r"self-proclaimed"), "自称"),
    (re.compile(r"who proclaimed himself as"), "と自称する"),
    (re.compile(r"the Red Sea"), "紅海"),
    (re.compile(r"ホーシ"), "フーティ"),
    (re.compile(r"として自らを宣言した"), "と自称した"),
]
_COUAGR_RE = re.compile(r"\b[Cc]ougars?\b")
_PUMA_RE = re.compile(r"\b[Pp]umas?\b")
_MOOSE_RE = re.compile(r"\bmoose\b", re.IGNORECASE)
_HANGUL_SPACED_RE = re.compile(r"(?:[\uac00-\ud7a3] ){3,}[\uac00-\ud7a3]")
_CJK_SPACED_RE = re.compile(r"(?:[\u4e00-\u9fff] ){3,}[\u4e00-\u9fff]")
_REPEATED_PHRASE_RE = re.compile(r"(.{4,}?)(?:\s*\1){2,}")


def _looks_like_url_continuation(prev: str, nxt: str) -> bool:
    """True when a newline is likely inside a URL/path rather than prose."""
    window = f"{prev[-48:]}{nxt[:48]}"
    if re.search(r"https?://|www\d*\.", window, re.IGNORECASE):
        return True
    # Telegram 短縮ドメイン t.\nme
    if re.search(r"(?i)t\.\s*$", prev) and re.match(r"(?i)me\b", nxt or ""):
        return True
    if "/" in prev[-40:] and re.match(r"[a-zA-Z0-9._/?=&%#:@-]", nxt or ""):
        return True
    return False


def collapse_soft_linebreaks(text: str) -> str:
    """Join PDF line-wraps: URLs without space, Latin prose with space."""
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if len(lines) <= 1:
        return normalized

    out: list[str] = []
    buf = lines[0]
    for line in lines[1:]:
        if not line:
            out.append(buf)
            out.append("")  # 空行＝段落区切りを保持
            buf = ""
            continue
        if not buf:
            buf = line
            continue
        if _looks_like_url_continuation(buf, line):
            buf = f"{buf}{line}"
        elif re.search(r"(?i)t\.\s*$", buf) and re.match(r"(?i)me\b", line):
            buf = f"{buf}{line}"
        elif re.search(r"(?i)sections?\s*$", buf) and re.match(r"(?i)1260H", line):
            buf = f"{buf} {line}"
        elif re.search(r"(?i)1260H\s*$", buf) and line.startswith("("):
            buf = f"{buf}{line}"
        elif re.search(r"\([A-Za-z0-9]+\)$", buf) and re.match(r"^\([A-Za-z0-9]+\)", line):
            # 1260H(g)\n(2)(B) のような引用途中改行
            buf = f"{buf}{line}"
        elif re.search(r"\b[A-Z]\.\s*$", buf) and re.match(r"^\([A-Za-z]", line):
            # William M.\n(Mac)
            buf = f"{buf} {line}"
        elif is_entity_name_line(buf.strip()) and is_entity_name_line(line.strip()):
            # 企業名ロスターは行を結合しない
            out.append(buf)
            buf = line
        elif buf.endswith("-") and re.match(r"[A-Za-z]", line):
            buf = f"{buf[:-1]}{line}"
        elif (
            not re.search(r"[.!?。．！？:]$", buf.rstrip())
            and re.search(r"[A-Za-z0-9)\]\"']$", buf.rstrip())
            and re.match(r"[A-Za-z(\"']", line)
        ):
            # PDF 折り返し（文末記号なし）のみスペース結合
            buf = f"{buf} {line}"
        elif re.search(r"[\u0400-\u04FF\u4e00-\u9fff\uac00-\ud7a3]$", buf) and re.match(
            r"[\u0400-\u04FF\u4e00-\u9fff\uac00-\ud7a3]", line
        ):
            buf = f"{buf}{line}"
        else:
            out.append(buf)
            buf = line
    if buf != "" or (lines and lines[-1] == ""):
        out.append(buf)
    # 末尾の余剰空行を1つに正規化
    while len(out) > 1 and out[-1] == "" and out[-2] == "":
        out.pop()
    return "\n".join(out)


def collapse_pdf_spacing(text: str) -> str:
    """Collapse PDF char-spaced CJK/Hangul runs and squeeze excess blanks."""
    text = collapse_soft_linebreaks(text)
    # 「일 본 의 포 용」のような1文字トークン連続のみ潰す（語間空白は残す）
    text = _HANGUL_SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    text = _CJK_SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    # 引用符まわりの PDF 空白
    text = re.sub(r"([「『“‘\"])\s+", r"\1", text)
    text = re.sub(r"\s+([」』”’\"])", r"\1", text)
    text = re.sub(r"\s+([,，.。!！?？])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text


def apply_gloss(text: str, rules: list[tuple[re.Pattern[str], str]]) -> str:
    """Apply ordered glossary substitutions."""
    for pattern, repl in rules:
        text = pattern.sub(repl, text)
    return text


def apply_regulatory_gloss_en(text: str) -> str:
    """Rewrite EN phrases that NLLB systematically mistranslates in 1260H docs."""
    return apply_gloss(text, _REGULATORY_GLOSS_EN)


def apply_news_gloss_en(text: str) -> str:
    """Rewrite EN news terms that NLLB systematically mistranslates."""
    return apply_gloss(text, _NEWS_GLOSS_EN)


def collapse_legal_citation_breaks(text: str) -> str:
    """Join PDF wraps inside Section 1260H(...) citations after soft-linebreak pass."""
    text = re.sub(r"(?i)(sections?)\s*\n\s*(1260H)", r"\1 \2", text)
    text = re.sub(r"(?i)(1260H)\s*\n\s*(\()", r"\1\2", text)
    text = re.sub(r"(\([A-Za-z0-9]+\))\s*\n\s*(\([A-Za-z0-9]+\))", r"\1\2", text)
    text = re.sub(r"\b([A-Z])\.\s*\n\s*(\([A-Za-z])", r"\1. \2", text)
    return text


def apply_ja_post_fixes(text: str) -> str:
    """Correct known JA mistranslations after decoding."""
    for pattern, repl in _JA_POST_FIXES:
        text = pattern.sub(repl, text)
    return text


def shield_legal_cites(text: str) -> tuple[str, list[str]]:
    """Replace legal citations with ASCII placeholders for MT, return (text, cites)."""
    cites: list[str] = []

    def _shield(match: re.Match[str]) -> str:
        cites.append(match.group(0))
        return f"ZZREF{len(cites) - 1}ZZ"

    return _LEGAL_CITE_INLINE_RE.sub(_shield, text), cites


def restore_legal_cites(text: str, cites: list[str]) -> str:
    """Restore placeholders produced by ``shield_legal_cites``."""
    for idx, cite in enumerate(cites):
        token = f"ZZREF{idx}ZZ"
        text = text.replace(token, cite)
        # MT が Z/空白を欠落・挿入する場合の保険
        text = re.sub(rf"(?i)z{{1,3}}\s*ref\s*{idx}\s*z{{1,3}}", cite, text)
    return text


def suppress_repeated_phrases(text: str) -> str:
    """Truncate pathological MT loops like 'A A A A' or long phrase repeats."""
    if not text:
        return text
    cleaned = text
    for _ in range(5):
        nxt = _REPEATED_PHRASE_RE.sub(r"\1", cleaned)
        if nxt == cleaned:
            break
        cleaned = nxt
    # token-level runaway: same short token 4+ times
    cleaned = re.sub(r"(\S+)(?:\s+\1){3,}", r"\1", cleaned)
    return cleaned


def clamp_decoding_length(source_chars: int) -> int:
    """Bound CT2 output length to reduce loop room on short units."""
    return max(96, min(768, source_chars * 2 + 64))


def is_regulatory_list_text(text: str) -> bool:
    """Heuristic: NDAA 1260H entity lists and similar regulatory rosters."""
    return "1260H" in text or "U.S.C." in text


def is_entity_name_line(line: str) -> bool:
    """Latin-only entity / subsidiary lines should not be machine-translated."""
    stripped = line.strip()
    if not stripped or stripped.startswith("•") or stripped.startswith("*"):
        return False
    # 文が2つ以上ある行は企業名ではない
    if re.search(r"\.\s+[A-Z]", stripped):
        return False
    lower = stripped.lower()
    if any(
        v in lower
        for v in (
            " is ",
            " are ",
            " has ",
            " was ",
            " have ",
            " shall ",
            " because ",
            " directly ",
            " indirectly ",
        )
    ):
        return False
    if _ENTITY_SUFFIX_RE.search(stripped):
        return True
    if re.search(r"\b(formerly|also known as)\b", lower):
        return True
    if re.search(r"\b[a-z]{3,}\b", stripped):
        allowed_lower = {"of", "and", "for", "the", "van", "de", "di", "formerly"}
        lowers = re.findall(r"\b[a-z]{3,}\b", stripped)
        if any(w not in allowed_lower for w in lowers):
            return False
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.&()'\-/]*", stripped) and len(stripped) < 160:
        return True
    return False


def format_toefl_paragraph_markers(text: str) -> str:
    """Ensure (1)/(2) markers sit on their own line before translation."""
    return re.sub(r"\((\d{1,2})\)(?=[^\n])", r"(\1)\n", text)


def _is_url_like_literal(content: str) -> bool:
    s = content.strip()
    if not s:
        return False
    if _URL_RE.match(s):
        return True
    if re.match(r"(?i)t\.me\b", s):
        return True
    if re.match(
        r"(?i)[\w.-]+\.(?:com\.cn|co\.jp|co\.kr|com|org|net|info|io|cn|jp|ru)\b",
        s,
    ):
        return True
    return False


def join_literal_segments(segments: list[tuple[bool, str]]) -> str:
    """Join translated segments with readable spacing around TOEFL markers."""
    out: list[str] = []
    for idx, (is_literal, content) in enumerate(segments):
        if idx > 0:
            prev_literal, prev_content = segments[idx - 1]
            if is_literal and content.strip() in ("[]", "[X]"):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
            elif is_literal and _TOEFL_PARA_ONLY_RE.fullmatch(content.strip()):
                if out and not out[-1].endswith("\n\n"):
                    out.append("\n\n")
            elif is_literal and re.fullmatch(r"-{5,}", content.strip()):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
            elif not is_literal and prev_literal and re.fullmatch(
                r"-{5,}", prev_content.strip()
            ):
                if out and not out[-1].endswith("\n\n"):
                    out.append("\n\n")
            elif is_literal and _is_url_like_literal(content):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
            elif not is_literal and prev_literal and _is_url_like_literal(prev_content):
                if out and not out[-1].endswith(("\n", " ")):
                    out.append("\n")
            elif not is_literal and _PARA_NUM_START_RE.match(content.lstrip()):
                if prev_literal and prev_content.strip() in ("[]", "[X]"):
                    if out and not out[-1].endswith("\n\n"):
                        out.append("\n\n" if not out[-1].endswith("\n") else "\n")
            elif not is_literal and prev_literal and _TOEFL_PARA_ONLY_RE.fullmatch(
                prev_content.strip()
            ):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
        out.append(content)
        if is_literal and content.strip() in ("[]", "[X]") and not content.endswith("\n"):
            out.append("\n")
        if is_literal and content.strip().lower() == "[end]" and not content.endswith("\n"):
            out.append("\n")
        if is_literal and re.fullmatch(r"-{5,}", content.strip()) and not content.endswith("\n"):
            out.append("\n\n")
        if is_literal and _is_url_like_literal(content) and not content.endswith("\n"):
            out.append("\n")
    return "".join(out)


def normalize_source_for_mt(text: str, src: str, tgt: str) -> str:
    """Apply lightweight source fixes for known NLLB failure modes."""
    text = collapse_pdf_spacing(text)
    if tgt != "ja":
        return text

    if src == "ko":
        return apply_gloss(text, _NEWS_GLOSS_KO)
    if src == "zh":
        return apply_gloss(text, _NEWS_GLOSS_ZH)
    if src == "ru":
        return apply_gloss(text, _NEWS_GLOSS_RU)

    # EN→JA 固有の誤訳回避
    if src in ("en", "auto"):

        def _cougar_repl(match: re.Match[str]) -> str:
            word = match.group(0)
            plural = word.lower().endswith("s")
            if word[0].isupper():
                return "Mountain lions" if plural else "Mountain lion"
            return "mountain lions" if plural else "mountain lion"

        text = _COUAGR_RE.sub(_cougar_repl, text)
        text = _PUMA_RE.sub(_cougar_repl, text)
        text = _MOOSE_RE.sub("North American moose", text)
        text = apply_news_gloss_en(text)
        if is_regulatory_list_text(text):
            text = collapse_legal_citation_breaks(text)
            text = apply_regulatory_gloss_en(text)
            return text
        text = format_toefl_paragraph_markers(text)
        text = re.sub(r"\bparagraph\s+(\d{1,2})\b", r"item \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\bin\s+paragraph\s+(\d{1,2})\b", r"in item \1", text, flags=re.IGNORECASE)
        text = re.sub(r"(?m)^(\d{1,2})\.\s+", r"Question \1: ", text)
        text = re.sub(r"\bThe word\b", "The term", text)
        text = re.sub(r"\bThe phrase\b", "The expression", text)
    return text


def split_literal_segments(text: str) -> list[tuple[bool, str]]:
    """Split text into literal (URL/TOEFL marker) and translatable segments."""
    text = collapse_pdf_spacing(text)
    segments: list[tuple[bool, str]] = []
    pos = 0
    for match in _LITERAL_SPLIT_RE.finditer(text):
        if match.start() > pos:
            segments.append((False, text[pos : match.start()]))
        segments.append((True, match.group(0)))
        pos = match.end()
    if pos < len(text):
        segments.append((False, text[pos:]))
    if not segments:
        segments.append((False, text))
    return segments


def merge_orphan_markers(paragraphs: list[str]) -> list[str]:
    """Attach standalone (1)/(2) markers to the following paragraph."""
    merged: list[str] = []
    idx = 0
    while idx < len(paragraphs):
        current = paragraphs[idx].strip()
        if _ORPHAN_MARKER_RE.fullmatch(current) and idx + 1 < len(paragraphs):
            merged.append(f"{current}\n{paragraphs[idx + 1].strip()}")
            idx += 2
            continue
        merged.append(current)
        idx += 1
    return merged


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines while keeping paragraph structure."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    paragraphs = parts if parts else [normalized]
    return merge_orphan_markers(paragraphs)


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences (Latin + CJK punctuation)."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    # 法令引用 1260H(g)(2)(B) 内のピリオドで誤分割しない
    shielded = paragraph
    cite_hits: list[str] = []

    def _shield(match: re.Match[str]) -> str:
        cite_hits.append(match.group(0))
        return f"⟦C{len(cite_hits) - 1}⟧"

    shielded = _LEGAL_CITE_INLINE_RE.sub(_shield, shielded)
    # William M. (Mac) のようなミドルイニシャルで文分割しない
    shielded = re.sub(r"\b([A-Z])\.\s+(?=\()", r"\1⟦DOT⟧ ", shielded)
    # Inc. / Ltd. / U.S. 等の略語ピリオドで文分割しない
    shielded = re.sub(
        r"\b(Inc|Ltd|Corp|Co|LLC|PLC|Mr|Mrs|Ms|Dr|Jr|Sr|U\.S|Pub)\.",
        lambda m: m.group(0).replace(".", "⟦DOT⟧"),
        shielded,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"(?<=[。．！？!?\.])\s*", shielded)
    sentences: list[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        p = p.replace("⟦DOT⟧", ".")
        for idx, cite in enumerate(cite_hits):
            p = p.replace(f"⟦C{idx}⟧", cite)
        sentences.append(p)
    return sentences if sentences else [paragraph]


def split_quiz_options(unit: str) -> list[str]:
    """Split long multiple-choice blocks so one bad option does not loop the batch."""
    if len(unit) < 280 or not re.search(r"\([A-D]\)", unit):
        return [unit]
    parts = re.split(r"(?<=\))\s*(?=\([A-D]\))", unit)
    return [part.strip() for part in parts if part.strip()]


def chunk_paragraph(paragraph: str) -> list[str]:
    """Split into lines/sentences; long blocks are further chunked by character length."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    if (
        MAX_WHOLE_PARAGRAPH_CHARS > 0
        and len(paragraph) <= MAX_WHOLE_PARAGRAPH_CHARS
    ):
        return [paragraph]
    # 企業名1行はそのまま（Inc. ピリオドで割らない）
    if "\n" not in paragraph and is_entity_name_line(paragraph):
        return [paragraph]

    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    if len(lines) > 1:
        units: list[str] = []
        for line in lines:
            if is_entity_name_line(line):
                units.append(line)
                continue
            if line.startswith("•"):
                units.extend(_chunk_single_block(line))
                continue
            units.extend(_chunk_single_block(line))
        return units if units else [paragraph]

    return _chunk_single_block(paragraph)


def _chunk_single_block(block: str) -> list[str]:
    units: list[str] = []
    for sentence in split_sentences(block):
        if len(sentence) <= MAX_SENTENCE_CHARS:
            units.extend(split_quiz_options(sentence))
            continue
        for i in range(0, len(sentence), MAX_SENTENCE_CHARS):
            piece = sentence[i : i + MAX_SENTENCE_CHARS]
            units.extend(split_quiz_options(piece))
    return units if units else [block]


def is_pass_through_unit(unit: str) -> bool:
    """Units that should not be sent to the MT model."""
    stripped = unit.strip()
    if not stripped:
        return True
    if _URL_RE.fullmatch(stripped):
        return True
    if _TOEFL_MARKER_RE.fullmatch(stripped):
        return True
    if _TOEFL_PARA_ONLY_RE.fullmatch(stripped):
        return True
    if stripped.lower() == "[end]":
        return True
    if re.fullmatch(r"-{5,}", stripped):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}", stripped):
        return True
    if is_entity_name_line(stripped):
        return True
    return False


def join_translated_units(units: list[str]) -> str:
    """Join translated units, preserving line breaks for roster-style lines."""
    if not units:
        return ""
    parts: list[str] = []
    for unit in units:
        parts.append(unit)
        stripped = unit.strip()
        if not stripped:
            continue
        if (
            is_entity_name_line(stripped)
            or stripped.startswith("•")
            or _LEGAL_CITE_INLINE_RE.fullmatch(stripped)
        ) and not unit.endswith("\n"):
            parts.append("\n")
    return "".join(parts)


def iter_translation_units(text: str) -> list[tuple[int, str]]:
    """Flatten paragraphs into translation units with paragraph index."""
    units: list[tuple[int, str]] = []
    paragraphs = split_paragraphs(text)
    for para_idx, paragraph in enumerate(paragraphs):
        for chunk in chunk_paragraph(paragraph):
            units.append((para_idx, chunk))
    return units if units else [(0, text.strip())]


class MtEngine:
    """NLLB CT2 engine — load → translate → close."""

    def __init__(self, model_dir: Path, *, intra_threads: int = 2) -> None:
        self._model_dir = Path(model_dir)
        self._translator: Any = None
        self._tokenizer: Any = None
        self._load_model(intra_threads)

    def _resolve_tokenizer_dir(self) -> Path:
        """Return local HF tokenizer path (required for correct NLLB tokenization)."""
        local = self._model_dir / HF_TOKENIZER_SUBDIR
        if local.is_dir() and (local / "tokenizer_config.json").is_file():
            return local
        raise FileNotFoundError(
            f"NLLB HF tokenizer not found at {local}. "
            "Run: python scripts/download-models.py"
        )

    def _load_hf_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer_dir = self._resolve_tokenizer_dir()
        try:
            return AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                fix_mistral_regex=True,
            )
        except TypeError:
            return AutoTokenizer.from_pretrained(str(tokenizer_dir))

    def _load_model(self, intra_threads: int) -> None:
        import ctranslate2

        self._translator = ctranslate2.Translator(
            str(self._model_dir),
            device="cpu",
            inter_threads=1,
            intra_threads=intra_threads,
        )
        self._tokenizer = self._load_hf_tokenizer()
        logger.info(
            "MT CT2 loaded from %s (tokenizer: %s)",
            self._model_dir,
            self._resolve_tokenizer_dir(),
        )

    def close(self) -> None:
        self._translator = None
        self._tokenizer = None

    def translate(
        self,
        text: str,
        src: str,
        tgt: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> TranslateResult:
        """Translate text in batches, preserving paragraphs and unit alignment."""
        if self._translator is None or self._tokenizer is None:
            raise RuntimeError("MT engine is not loaded")

        src_code = NLLB_LANG_CODES.get(src)
        tgt_code = NLLB_LANG_CODES.get(tgt)
        if tgt_code is None:
            raise ValueError(f"Unsupported target language: {tgt}")
        if src != "auto" and src_code is None:
            raise ValueError(f"Unsupported source language: {src}")

        segments = split_literal_segments(text)
        progress_total = 0
        for is_literal, content in segments:
            if is_literal or not content.strip():
                continue
            progress_total += len(
                iter_translation_units(normalize_source_for_mt(content, src, tgt))
            )
        progress = {"done": 0, "total": max(progress_total, 1)}

        built: list[tuple[bool, str]] = []
        alignment: list[dict[str, Any]] = []
        unit_seq = 0

        for is_literal, content in segments:
            if is_literal:
                built.append((True, content))
                if content:
                    alignment.append(
                        {"id": f"u{unit_seq}", "src": content, "tgt": content}
                    )
                    unit_seq += 1
                continue
            if not content.strip():
                built.append((False, content))
                continue

            body_text, pairs = self._translate_body_with_pairs(
                normalize_source_for_mt(content, src, tgt),
                src_code,
                tgt_code,
                progress=progress,
                on_progress=on_progress,
            )
            for src_unit, tgt_unit in pairs:
                alignment.append(
                    {
                        "id": f"u{unit_seq}",
                        "src": src_unit,
                        "tgt": tgt_unit,
                    }
                )
                unit_seq += 1
            built.append((False, body_text))

        return TranslateResult(
            text=join_literal_segments(built),
            units=alignment,
        )

    def _translate_body_with_pairs(
        self,
        text: str,
        src_code: str | None,
        tgt_code: str,
        *,
        progress: dict[str, int],
        on_progress: ProgressCallback | None = None,
    ) -> tuple[str, list[tuple[str, str]]]:
        """Translate a text block and return (joined text, src/tgt unit pairs)."""
        units = iter_translation_units(text)
        paragraph_count = max((idx for idx, _ in units), default=0) + 1
        translated_by_para: dict[int, list[str]] = {i: [] for i in range(paragraph_count)}
        pairs: list[tuple[str, str]] = []

        batch_indices: list[int] = []
        batch_sentences: list[str] = []

        def _report(detail: str) -> None:
            if on_progress:
                on_progress(progress["done"], progress["total"], detail)

        def _flush_batch() -> None:
            nonlocal batch_indices, batch_sentences
            if not batch_sentences:
                return
            shielded_batch: list[str] = []
            cite_lists: list[list[str]] = []
            for sentence in batch_sentences:
                shielded, cites = shield_legal_cites(sentence)
                shielded_batch.append(shielded)
                cite_lists.append(cites)
            pieces = self._translate_batch(shielded_batch, src_code, tgt_code)
            for unit_idx, piece, cites in zip(batch_indices, pieces, cite_lists):
                para_idx, original = units[unit_idx]
                tgt_piece = piece if piece else original
                tgt_piece = restore_legal_cites(tgt_piece, cites)
                if tgt_code == "jpn_Jpan":
                    tgt_piece = apply_ja_post_fixes(tgt_piece)
                translated_by_para[para_idx].append(tgt_piece)
                pairs.append((original, tgt_piece))
                progress["done"] += 1
                _report(tgt_piece[:60] if tgt_piece else original[:60])
            batch_indices = []
            batch_sentences = []

        for unit_idx, (para_idx, sentence) in enumerate(units):
            if is_pass_through_unit(sentence):
                tgt_piece = sentence.strip()
                translated_by_para[para_idx].append(tgt_piece)
                pairs.append((sentence, tgt_piece))
                progress["done"] += 1
                _report(tgt_piece[:60] if tgt_piece else sentence[:60])
                continue

            batch_indices.append(unit_idx)
            batch_sentences.append(sentence)
            if len(batch_sentences) >= TRANSLATE_BATCH_SIZE:
                _flush_batch()

        _flush_batch()

        paragraphs = [
            join_translated_units(translated_by_para[i]).strip()
            for i in range(paragraph_count)
            if translated_by_para[i]
        ]
        return "\n\n".join(paragraphs), pairs

    def _translate_batch(
        self,
        sentences: list[str],
        src_code: str | None,
        tgt_code: str,
    ) -> list[str]:
        """Translate multiple sentences in one CT2 batch call."""
        assert self._translator is not None
        assert self._tokenizer is not None

        if not sentences:
            return []

        tokenizer = self._tokenizer
        if src_code:
            tokenizer.src_lang = src_code

        sources = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence))
            for sentence in sentences
        ]
        prefixes = [[tgt_code]] * len(sources)
        # 短い文は出力長を抑え、反復 n-gram を禁止してループを防ぐ
        max_len = max(clamp_decoding_length(len(s)) for s in sentences)
        results = self._translator.translate_batch(
            sources,
            target_prefix=prefixes,
            beam_size=2,
            max_batch_size=max(TRANSLATE_BATCH_SIZE, len(sources)),
            max_decoding_length=max_len,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        decoded: list[str] = []
        for sentence, result in zip(sentences, results):
            if not result.hypotheses:
                decoded.append("")
                continue
            tokens = list(result.hypotheses[0])
            if tokens and tokens[0] == tgt_code:
                tokens = tokens[1:]
            text = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(tokens),
                skip_special_tokens=True,
            ).strip()
            text = suppress_repeated_phrases(text)
            # NLLB が稀に出す無意味な「ほら」を除去
            text = re.sub(r"(?:^|[。．\s])ほら(?:[。．\s]|$)", " ", text).strip()
            text = re.sub(r"\s{2,}", " ", text)
            # ほぼ英語のまま残った出力には日本語句点を付けない
            latin_ratio = (
                sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
                / max(len(text), 1)
            )
            if (
                tgt_code == "jpn_Jpan"
                and text
                and latin_ratio < 0.45
                and text[-1] not in "。．！？!?」』）)]"
            ):
                text += "。"
            decoded.append(text)
        return decoded
