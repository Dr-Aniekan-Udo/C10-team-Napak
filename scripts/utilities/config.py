"""Typed application configuration loaded from environment values."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_DEFAULTS = {
    "AGRO_RAG_SESSION_DIR": ".local/sessions",
    "AGRO_RAG_RANKER": "cross_encoder",
    "AGRO_RAG_RANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "AGRO_RAG_TOP_K": "5",
}
_SUPPORTED_RANKERS = {"cross_encoder", "lightgbm", "none"}
_NULLABLE_VALUES = {"", "null", "none"}


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required and must be non-empty")
    return value


@dataclass(frozen=True)
class AppConfig:
    api_base_url: str
    api_key: str | None
    model: str
    data_dir: Path
    session_dir: Path
    ranker: str
    rank_model: str
    top_k: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AppConfig":
        """Build config with explicit values taking precedence at every layer."""
        dotenv_path = Path(__file__).resolve().parents[2] / ".env"
        values = _DEFAULTS | _load_dotenv(dotenv_path) | dict(os.environ)
        if environ is not None:
            values.update(environ)

        api_key_value = values.get("AGRO_RAG_API_KEY", "").strip()
        api_key = None if api_key_value.casefold() in _NULLABLE_VALUES else api_key_value
        ranker = values.get("AGRO_RAG_RANKER", _DEFAULTS["AGRO_RAG_RANKER"]).strip()
        if ranker not in _SUPPORTED_RANKERS:
            raise ValueError(f"AGRO_RAG_RANKER must be one of {sorted(_SUPPORTED_RANKERS)}")
        try:
            top_k = int(values.get("AGRO_RAG_TOP_K", _DEFAULTS["AGRO_RAG_TOP_K"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("AGRO_RAG_TOP_K must be a positive integer") from exc
        if top_k < 1:
            raise ValueError("AGRO_RAG_TOP_K must be a positive integer")

        return cls(
            api_base_url=_required(values, "AGRO_RAG_API_BASE_URL"),
            api_key=api_key,
            model=_required(values, "AGRO_RAG_MODEL"),
            data_dir=Path(_required(values, "AGRO_RAG_DATA_DIR")).expanduser(),
            session_dir=Path(values.get("AGRO_RAG_SESSION_DIR", _DEFAULTS["AGRO_RAG_SESSION_DIR"])).expanduser(),
            ranker=ranker,
            rank_model=_required(values, "AGRO_RAG_RANK_MODEL"),
            top_k=top_k,
        )
