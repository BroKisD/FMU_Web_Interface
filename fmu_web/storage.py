from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4


@dataclass
class SessionState:
    fmu_token: Optional[str] = None
    fmu_name: Optional[str] = None
    fmu_bytes: Optional[bytes] = None
    input_token: Optional[str] = None
    input_name: Optional[str] = None
    input_bytes: Optional[bytes] = None
    result_blobs: Dict[str, Tuple[str, bytes]] = field(default_factory=dict)


class SessionStore:
    def __init__(self) -> None:
        self._state = SessionState()

    @property
    def fmu_token(self) -> Optional[str]:
        return self._state.fmu_token

    @property
    def fmu_name(self) -> Optional[str]:
        return self._state.fmu_name

    @property
    def fmu_bytes(self) -> Optional[bytes]:
        return self._state.fmu_bytes

    @property
    def input_token(self) -> Optional[str]:
        return self._state.input_token

    @property
    def input_name(self) -> Optional[str]:
        return self._state.input_name

    @property
    def input_bytes(self) -> Optional[bytes]:
        return self._state.input_bytes

    @property
    def result_tokens(self) -> List[str]:
        return list(self._state.result_blobs.keys())

    def set_fmu(self, name: str, data: bytes) -> str:
        token = _new_token()
        self._state.fmu_token = token
        self._state.fmu_name = name
        self._state.fmu_bytes = data
        self._state.input_token = None
        self._state.input_name = None
        self._state.input_bytes = None
        self._state.result_blobs = {}
        return token

    def set_input(self, name: str, data: bytes) -> str:
        token = _new_token()
        self._state.input_token = token
        self._state.input_name = name
        self._state.input_bytes = data
        return token

    def add_result(self, name: str, data: bytes) -> str:
        token = _new_token()
        self._state.result_blobs[token] = (name, data)
        return token

    def get_result(self, token: str) -> Optional[Tuple[str, bytes]]:
        return self._state.result_blobs.get(token)

    def clear(self, upload_dir: Path) -> Dict[str, List[str]]:
        removed: List[str] = []
        if self._state.fmu_token:
            removed.append(self._state.fmu_token)
        if self._state.input_token:
            removed.append(self._state.input_token)
        removed.extend(self._state.result_blobs.keys())
        self._state = SessionState()
        return {"removed": removed, "errors": []}


def cleanup_old_files(upload_dir: Path, max_age_hours: int) -> int:
    if not upload_dir.exists():
        return 0

    now = datetime.now()
    cutoff = now - timedelta(hours=max_age_hours)
    removed_count = 0

    for file_path in upload_dir.iterdir():
        if not file_path.is_file():
            continue
        try:
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        except OSError:
            continue
        if file_mtime < cutoff:
            try:
                file_path.unlink()
                removed_count += 1
            except OSError:
                continue

    return removed_count


def _new_token() -> str:
    return f"mem:{uuid4().hex}"
