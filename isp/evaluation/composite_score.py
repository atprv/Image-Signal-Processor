"""Composite score normalization for E2E ISP evaluation."""

import json
from pathlib import Path
from typing import Any


def load_composite_config(path: str | Path) -> dict[str, Any]:
    """Load frozen metric ranges and weights for the validation composite."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    ranges = cfg.get("baseline_minmax")
    if not isinstance(ranges, dict):
        raise ValueError(f"{path} missing 'baseline_minmax' metric ranges")

    formula = cfg.get("user_formula", {})
    return {
        "source": str(path),
        "ranges": ranges,
        "a": float(formula.get("a", 1.0)),
        "b": float(formula.get("b", 1.0)),
        "mode": cfg.get("normalization", {}).get("mode", "baseline_minmax_clamped"),
    }


def normalize_metric(
    value: float, metric_name: str, cfg: dict[str, Any], clamp: bool = True
) -> float:
    """Normalize one raw metric value to a baseline min-max scale."""
    ranges = cfg["ranges"]
    if metric_name not in ranges:
        raise KeyError(f"Metric '{metric_name}' not found in composite ranges")

    lo = float(ranges[metric_name]["min"])
    hi = float(ranges[metric_name]["max"])
    if hi <= lo:
        raise ValueError(f"Invalid range for {metric_name}: min={lo}, max={hi}")

    norm = (float(value) - lo) / (hi - lo)
    if clamp:
        norm = max(0.0, min(1.0, norm))
    return float(norm)


def compute_normalized_terms(
    vif: float, nrqm: float, unique: float, cfg: dict[str, Any]
) -> dict[str, float]:
    """Return normalized per-metric terms used by the composite score."""
    vif_norm = normalize_metric(vif, "vif", cfg)
    nrqm_norm = normalize_metric(nrqm, "nrqm", cfg)
    unique_norm = normalize_metric(unique, "unique", cfg)
    return {
        "vif_norm": vif_norm,
        "nrqm_norm": nrqm_norm,
        "unique_norm": unique_norm,
        "a_nrqm_norm": float(cfg["a"]) * nrqm_norm,
        "b_unique_norm": float(cfg["b"]) * unique_norm,
    }


def compute_composite(vif: float, nrqm: float, unique: float, cfg: dict[str, Any]) -> float:
    """Composite = VIF_norm + a*NRQM_norm + b*UNIQUE_norm."""
    terms = compute_normalized_terms(vif, nrqm, unique, cfg)
    return float(terms["vif_norm"] + terms["a_nrqm_norm"] + terms["b_unique_norm"])
