"""
Composite score helpers for ISP evaluation.
"""

import json
from pathlib import Path
from typing import Any


def _infer_composite_mode(cfg: dict[str, Any], formula: dict[str, Any]) -> str:
    """Recover a descriptive normalization mode from old/new JSON schemas."""
    explicit_mode = cfg.get("mode") or formula.get("mode")
    if explicit_mode:
        return str(explicit_mode)

    composite_norm = cfg.get("composite_normalization", {})
    if isinstance(composite_norm, dict) and composite_norm.get("mode"):
        return str(composite_norm["mode"])

    theoretical_ranges = cfg.get("metric_theoretical_range", {})
    if isinstance(theoretical_ranges, dict) and theoretical_ranges:
        return "theoretical_range_scaled"

    baseline_ranges = cfg.get("baseline_minmax", {})
    if isinstance(baseline_ranges, dict) and baseline_ranges:
        return "baseline_minmax_clamped"

    return "legacy_untyped"


def load_composite_config(path: str | Path) -> dict[str, Any]:
    """Load the frozen ``(a, b)`` weights and baseline ranges from JSON."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    formula = cfg.get("user_formula", {})
    return {
        "source": str(path),
        "ranges": cfg.get("baseline_minmax", {}),
        "a": float(formula.get("a", 1.0)),
        "b": float(formula.get("b", 1.0)),
        "mode": _infer_composite_mode(cfg, formula),
    }


def compute_composite_terms(
    vif: float, nrqm: float, unique: float, cfg: dict[str, Any]
) -> dict[str, float]:
    """Return the three contributions to the composite score.

    Keys:
        ``vif_term``     -- VIF (no coefficient).
        ``a_nrqm_term``  -- ``a * nrqm``.
        ``b_unique_term``-- ``b * unique``.
    """
    a = float(cfg["a"])
    b = float(cfg["b"])
    return {
        "vif_term": float(vif),
        "a_nrqm_term": a * float(nrqm),
        "b_unique_term": b * float(unique),
    }


# Legacy alias kept so external code that still imports the old name does
# not break. ``compute_composite_terms`` is the preferred entry point.
compute_normalized_terms = compute_composite_terms


def compute_composite(vif: float, nrqm: float, unique: float, cfg: dict[str, Any]) -> float:
    """Return ``vif + a * nrqm + b * unique`` for the supplied config."""
    terms = compute_composite_terms(vif, nrqm, unique, cfg)
    return float(terms["vif_term"] + terms["a_nrqm_term"] + terms["b_unique_term"])
