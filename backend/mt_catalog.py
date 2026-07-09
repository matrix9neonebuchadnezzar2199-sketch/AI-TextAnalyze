"""Known NLLB CTranslate2 MT variants and install metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MtModelSpec:
    """One installable MT model under ``model/<folder>/``."""

    id: str
    folder: str
    label: str
    hf_tokenizer_repo: str
    download_repo: str


MT_CATALOG: tuple[MtModelSpec, ...] = (
    MtModelSpec(
        id="nllb-600m",
        folder="mt-nllb-600m-ct2-int8",
        label="NLLB distilled 600M",
        hf_tokenizer_repo="facebook/nllb-200-distilled-600M",
        download_repo="Tushe/nllb-200-600M-ct2-int8",
    ),
    MtModelSpec(
        id="nllb-1.3b",
        folder="mt-nllb-1.3b-ct2-int8",
        label="NLLB distilled 1.3B",
        hf_tokenizer_repo="facebook/nllb-200-distilled-1.3B",
        download_repo="OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
    ),
)

DEFAULT_MT_MODEL_ID = "nllb-600m"


def get_mt_spec(model_id: str) -> MtModelSpec:
    """Return catalog entry for ``model_id``.

    Raises:
        ValueError: Unknown model id.
    """
    for spec in MT_CATALOG:
        if spec.id == model_id:
            return spec
    raise ValueError(f"Unknown MT model id: {model_id}")
