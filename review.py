"""Review dashboard: approve or reject rendered shorts from a phone.

Layout under LOCAL_OUTPUT_DIR (written by the local pipeline):
    <video name>/short_01.mp4 (+ .json, .txt)   pending
    <video name>/approved/short_01.mp4           approved
    <video name>/rejected/short_01.mp4           rejected

A short's status is the folder it sits in, so there is no state file.
"""
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STATUSES = ("pending", "approved", "rejected")
SIDECARS = (".mp4", ".json", ".txt")
_NAME = re.compile(r"short_\d+(-\d+)?")


def _folder(group_dir: Path, status: str) -> Path:
    return group_dir if status == "pending" else group_dir / status


def _meta(mp4: Path) -> Dict:
    try:
        return json.loads(mp4.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def list_groups(root: Path) -> List[Dict]:
    """One entry per source-video folder that holds shorts, newest first."""
    groups = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        shorts = [
            {"name": mp4.stem, "status": s, "rel": mp4.relative_to(d).as_posix(), "meta": _meta(mp4)}
            for s in STATUSES
            for mp4 in sorted(_folder(d, s).glob("short_*.mp4"))
        ]
        if shorts:
            mtime = max((d / s["rel"]).stat().st_mtime for s in shorts)
            groups.append({"name": d.name, "mtime": mtime, "shorts": shorts})
    return sorted(groups, key=lambda g: g["mtime"], reverse=True)


def move_short(root: Path, group: str, name: str, to: str) -> Path:
    """Move a short's mp4 and sidecars into the folder for status `to`; never overwrites."""
    root = Path(root)
    if group in ("", ".", "..") or "/" in group or "\\" in group or not (root / group).is_dir():
        raise ValueError(f"bad group: {group!r}")
    if not _NAME.fullmatch(name):
        raise ValueError(f"bad short name: {name!r}")
    if to not in STATUSES:
        raise ValueError(f"bad status: {to!r}")
    group_dir = root / group
    src_dir = next((_folder(group_dir, s) for s in STATUSES if (_folder(group_dir, s) / f"{name}.mp4").is_file()), None)
    if src_dir is None:
        raise FileNotFoundError(f"{group}/{name}")
    dest_dir = _folder(group_dir, to)
    if src_dir == dest_dir:
        return src_dir / f"{name}.mp4"
    dest_dir.mkdir(exist_ok=True)
    base, stem, n = re.sub(r"-\d+$", "", name), name, 2
    while any((dest_dir / f"{stem}{ext}").exists() for ext in SIDECARS):
        stem, n = f"{base}-{n}", n + 1
    for ext in SIDECARS:
        if (src_dir / f"{name}{ext}").exists():
            shutil.move(str(src_dir / f"{name}{ext}"), str(dest_dir / f"{stem}{ext}"))
    return dest_dir / f"{stem}.mp4"


def parse_range(header: str, size: int) -> Optional[Tuple[int, int]]:
    """Inclusive (start, end) for a single `bytes=` range, or None if invalid/unsatisfiable."""
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
    if not m or not (m.group(1) or m.group(2)):
        return None
    if not m.group(1):  # suffix range: last N bytes
        n = int(m.group(2))
        return (max(size - n, 0), size - 1) if n and size else None
    start = int(m.group(1))
    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    return (start, end) if start <= end and start < size else None
