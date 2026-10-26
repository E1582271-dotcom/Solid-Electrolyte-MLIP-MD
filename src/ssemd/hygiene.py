"""Repository checks: no CJK text, no private identifiers, file-size limits.

Run as ``ssemd check-repo`` (also part of the test suite and of CI).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .paths import REPO

MAX_FILE_BYTES = 5_000_000
MAX_TOTAL_BYTES = 60_000_000
TEXT_SUFFIXES = {".py", ".md", ".csv", ".json", ".txt", ".toml", ".cfg", ".yml", ".yaml", ".sh", ".pbs", ".svg",
                 ".cif", ".cff", ".example", ".gz", ""}
BINARY_SUFFIXES = {".png", ".pdf", ".gz"}

# CJK scripts and full-width forms, as code-point ranges (kept ASCII so this file passes its own check).
CJK_RANGES = ((0x2E80, 0x2FFF), (0x3000, 0x303F), (0x3040, 0x30FF), (0x3100, 0x312F), (0x3200, 0x32FF),
              (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))
CJK = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in CJK_RANGES) + "]")
# Patterns assembled from fragments so that this file does not match itself.
PRIVATE = {
    "student number": re.compile(r"(?<![0-9A-Za-z])[eE]\d{7}(?![0-9A-Za-z])"),
    "e-mail address": re.compile(r"[A-Za-z0-9._%+-]+@(?!users\.noreply)[A-Za-z0-9.-]+\.(?:edu|com|org|sg)\b"),
    "home directory": re.compile("/" + "Users/[A-Za-z]"),
    "cluster scratch path": re.compile("/" + "scratch/users"),
    "cluster home path": re.compile("/" + "home/users/"),
    "cluster host name": re.compile(r"(?:asp2a|aspire2a)[-.][A-Za-z0-9]|\." + r"nscc\.sg"),
    "PBS job id": re.compile(r"\b\d{7,8}\.pbs\d{3}\b"),
}


def tracked_files() -> list[Path]:
    """Files git would publish (tracked plus untracked, not ignored); all files if git is unavailable."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard"],
                             check=True, capture_output=True, text=True).stdout
        return sorted(REPO / line for line in out.splitlines() if line and (REPO / line).is_file())
    except (OSError, subprocess.CalledProcessError):
        return sorted(path for path in REPO.rglob("*") if path.is_file() and ".git" not in path.parts)


def _text(path: Path) -> str | None:
    if path.suffix == ".gz":
        import gzip

        return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    if path.suffix in (".png", ".pdf"):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def check(files: list[Path] | None = None) -> list[str]:
    files = tracked_files() if files is None else files
    problems = []
    total = 0
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        size = path.stat().st_size
        total += size
        if size > MAX_FILE_BYTES:
            problems.append(f"{rel}: {size} bytes > {MAX_FILE_BYTES}")
        if path.suffix not in TEXT_SUFFIXES | BINARY_SUFFIXES and path.name != "Makefile" and path.name != "LICENSE":
            problems.append(f"{rel}: unexpected file type")
        text = _text(path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if CJK.search(line):
                problems.append(f"{rel}:{line_no}: CJK text")
            for name, pattern in PRIVATE.items():
                if pattern.search(line):
                    problems.append(f"{rel}:{line_no}: {name}")
    if total > MAX_TOTAL_BYTES:
        problems.append(f"repository total {total} bytes > {MAX_TOTAL_BYTES}")
    return problems
