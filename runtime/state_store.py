from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

_LAST_CLEANUP_BY_DIR: dict[str, float] = {}


def cleanup_orphan_temp_files(directory: str | Path, max_age_s: float = 600.0, prefixes: tuple[str, ...] = ("tmp", ".dashboard_state.json.tmp.")) -> int:
    directory = Path(directory)
    if not directory.exists():
        return 0
    now = time.time()
    removed = 0
    for child in directory.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        try:
            age = now - child.stat().st_mtime
        except FileNotFoundError:
            continue
        if age < max_age_s:
            continue
        try:
            child.unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    return removed


def atomic_write_json(path: str | Path, payload: dict, cleanup_every_s: float = 600.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    cleanup_key = str(path.parent.resolve())
    last_cleanup = _LAST_CLEANUP_BY_DIR.get(cleanup_key, 0.0)
    if cleanup_every_s > 0 and (now - last_cleanup) >= cleanup_every_s:
        cleanup_orphan_temp_files(path.parent, max_age_s=max(cleanup_every_s, 60.0))
        _LAST_CLEANUP_BY_DIR[cleanup_key] = now
    tmp_path = None
    try:
        with NamedTemporaryFile('w', delete=False, dir=str(path.parent), prefix=f'.{path.name}.tmp.', encoding='utf-8') as tmp:
            json.dump(payload, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def append_csv_row(path: str | Path, fieldnames: Iterable[str], row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
