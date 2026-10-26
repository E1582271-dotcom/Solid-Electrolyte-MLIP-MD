"""Repository checks: no CJK text, no private identifiers, size limits."""
from __future__ import annotations

from ssemd import hygiene


def test_repository_is_clean():
    assert hygiene.check() == []


def test_checks_detect_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(hygiene, "REPO", tmp_path)
    bad = tmp_path / "notes.md"
    bad.write_text("contact " + "e" + "1234567\n" + chr(0x4E2D) + chr(0x6587) + "\n", encoding="utf-8")
    problems = hygiene.check([bad])
    assert any("student number" in p for p in problems)
    assert any("CJK" in p for p in problems)
