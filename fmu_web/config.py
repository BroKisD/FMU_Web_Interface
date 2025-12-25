from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    static_dir: Path
    upload_dir: Path
    max_upload_age_hours: int
    host: str
    port: int
    debug: bool


def _get_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parents[1]
    static_dir = Path(os.getenv("FMU_STATIC_DIR", base_dir / "static"))
    upload_dir = Path(os.getenv("FMU_UPLOAD_DIR", base_dir / "uploads"))
    max_upload_age_hours = int(os.getenv("FMU_MAX_UPLOAD_AGE_HOURS", "24"))
    host = os.getenv("FMU_HOST", "0.0.0.0")
    port = int(os.getenv("FMU_PORT", "8000"))
    debug = _get_bool(os.getenv("FMU_DEBUG"), True)
    return AppConfig(
        base_dir=base_dir,
        static_dir=static_dir,
        upload_dir=upload_dir,
        max_upload_age_hours=max_upload_age_hours,
        host=host,
        port=port,
        debug=debug,
    )
