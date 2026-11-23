"""Pruebas unitarias de Fase 6 (especificaciones autoresearch)."""

from __future__ import annotations

from shared.tune import AUTORESEARCH_SPECS, format_autoresearch_block


def test_autoresearch_specs_complete_for_all_workshops() -> None:
    for w in ("w2", "w3", "w4", "w5"):
        spec = AUTORESEARCH_SPECS[w]
        assert "test_invariants.py" in spec["guard"]
        assert "--fast-verify" in spec["verify"]
        block = format_autoresearch_block(w)
        assert "Goal:" in block and "Scope:" in block and "Verify:" in block and "Guard:" in block
