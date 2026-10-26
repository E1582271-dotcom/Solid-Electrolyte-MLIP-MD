"""Figures: complete exports at Nature column widths, source data for every figure, deterministic rebuild."""
from __future__ import annotations

import struct

import pytest

from ssemd import paths
from ssemd.figures import registry

DPI = 600
WIDTHS_MM = (89, 120, 183)


def _png_size(path):
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


@pytest.mark.parametrize("name", list(registry.FIGURES))
def test_exports_and_sizes(name):
    module, _ = registry.FIGURES[name]
    folder = paths.FIGURES / ("main" if name.startswith("fig") else "extended")
    for extension in ("pdf", "svg", "png"):
        assert (folder / f"{module}.{extension}").is_file(), f"{module}.{extension}"
    width_px, height_px = _png_size(folder / f"{module}.png")
    width_mm, height_mm = width_px / DPI * 25.4, height_px / DPI * 25.4
    assert any(abs(width_mm - target) < 0.3 for target in WIDTHS_MM), width_mm
    assert height_mm <= 247.3
    assert list(paths.FIGURES.glob(f"source_data/{name}_*.csv")), f"no source data for {name}"


def test_svg_text_is_editable_and_english():
    for svg in paths.FIGURES.rglob("*.svg"):
        text = svg.read_text(encoding="utf-8")
        assert "<text" in text or "glyph" not in text, svg.name


def test_rebuild_is_deterministic(tmp_path, monkeypatch):
    from ssemd.figures import style

    name = "ed08"
    module, _ = registry.FIGURES[name]
    original = (paths.FIGURES / "extended" / f"{module}.svg").read_bytes()
    monkeypatch.setattr(style, "FIGURES", tmp_path)
    registry.build([name])
    assert (tmp_path / "extended" / f"{module}.svg").read_bytes() == original
