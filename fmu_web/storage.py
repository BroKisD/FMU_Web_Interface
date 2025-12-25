from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SessionState:
    fmu_path: Optional[str] = None
    input_file: Optional[str] = None
    result_files: List[str] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._state = SessionState()

    @property
    def fmu_path(self) -> Optional[str]:
        return self._state.fmu_path

    @property
    def input_file(self) -> Optional[str]:
        return self._state.input_file

    @property
    def result_files(self) -> List[str]:
        return list(self._state.result_files)

    def set_fmu_path(self, path: str) -> None:
        self._state.fmu_path = path

    def set_input_file(self, path: str) -> None:
        self._state.input_file = path

    def add_result_file(self, path: str) -> None:
        self._state.result_files.append(path)

    def clear(self, upload_dir: Path) -> Dict[str, List[str]]:
        files_to_remove: List[str] = []
        if self._state.fmu_path:
            files_to_remove.append(self._state.fmu_path)
        if self._state.input_file:
            files_to_remove.append(self._state.input_file)
        files_to_remove.extend(self._state.result_files)

        removed: List[str] = []
        errors: List[str] = []
        for path in files_to_remove:
            if not _is_within_dir(path, upload_dir):
                continue
            try:
                file_path = Path(path)
                if file_path.exists():
                    file_path.unlink()
                    removed.append(path)
            except Exception as exc:
                errors.append(f"Failed to remove {path}: {exc}")

        self._state = SessionState()
        return {"removed": removed, "errors": errors}


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


def _is_within_dir(path: str, upload_dir: Path) -> bool:
    try:
        resolved = Path(path).resolve()
        base = upload_dir.resolve()
        return base == resolved or base in resolved.parents
    except Exception:
        return False
