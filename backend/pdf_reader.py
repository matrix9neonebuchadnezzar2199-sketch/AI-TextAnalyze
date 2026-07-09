"""PDF and plain-text file reading."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


def read_text_file(path: str | Path) -> str:
    """Read a .txt file as UTF-8 with BOM fallback.

    Args:
        path: File path.

    Returns:
        File contents.

    Raises:
        FileNotFoundError: Path does not exist.
        ValueError: Unsupported extension.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf")

    if suffix == ".txt":
        return _read_txt(file_path)
    return _read_pdf(file_path)


def _read_txt(file_path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Could not decode text file")


def _read_pdf(file_path: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages).strip()
    except Exception as exc:
        logger.exception("PDF parse failed: %s", file_path)
        raise RuntimeError(f"PDF parse failed: {file_path.name}") from exc
