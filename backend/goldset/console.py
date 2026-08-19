"""Terminal and JSONL helpers shared by the three gold-set CLIs.

`verify.py`, `screen.py`, and `audit.py` all read single keypresses, clear the
screen, and append one row at a time to a log they can be resumed from. Three
copies of that would drift; one copy is the whole reason this module exists.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def read_key() -> str:
    """Read a single keypress without waiting for Enter."""
    try:
        import msvcrt  # Windows
    except ImportError:
        pass
    else:
        return msvcrt.getch().decode("utf-8", errors="replace").lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, or return [] if it does not exist yet."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row and flush it to disk before returning.

    Explicitly flushed and fsynced: the whole point is that closing the
    terminal at question 80 costs nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a JSONL file from scratch, replacing anything already there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
