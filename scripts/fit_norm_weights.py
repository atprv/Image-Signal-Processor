"""
Fit and freeze the ``(a, b)`` coefficients for the composite quality score.
"""

import argparse
import json
from pathlib import Path

from isp.evaluation.baseline_io import load_baseline_metrics_txt

ROOT = Path(__file__).resolve().parents[1]


NRQM_RANGE_MAX = 10.0
UNIQUE_RANGE_MAX = 3.0
COMPOSITE_MODE = "theoretical_range_scaled"


def round_sig(x: float, sig: int = 6) -> float:
    """Round to ``sig`` significant figures for stable JSON output."""
    if x == 0.0:
        return 0.0
    import math

    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/baselines/norm_weights.json")
    parser.add_argument("--baseline-metrics", default="artifacts/baselines/baseline_metrics.txt")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["day", "night", "tunnel"],
        help="Which baseline scenes to include in the fit.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_metrics)
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path
    baseline_per_scene = load_baseline_metrics_txt(baseline_path)

    scenes = {s: baseline_per_scene[s] for s in args.scenes}
    n = len(scenes)

    vif_vals = [m["vif"] for m in scenes.values()]
    nrqm_vals = [m["nrqm"] for m in scenes.values()]
    unique_vals = [m["unique"] for m in scenes.values()]

    vif_mean = sum(vif_vals) / n
    nrqm_mean = sum(nrqm_vals) / n
    unique_mean = sum(unique_vals) / n

    a = 1.0 / NRQM_RANGE_MAX
    b = 1.0 / UNIQUE_RANGE_MAX

    ranges = {
        "vif": {"min": min(vif_vals), "max": max(vif_vals)},
        "nrqm": {"min": min(nrqm_vals), "max": max(nrqm_vals)},
        "unique": {"min": min(unique_vals), "max": max(unique_vals)},
    }

    per_scene_composite = {}
    for name, m in scenes.items():
        vif_term = m["vif"]
        a_nrqm_term = a * m["nrqm"]
        b_unique_term = b * m["unique"]
        per_scene_composite[name] = {
            "vif_term": round_sig(vif_term, 4),
            "a_nrqm_term": round_sig(a_nrqm_term, 4),
            "b_unique_term": round_sig(b_unique_term, 4),
            "composite": round_sig(vif_term + a_nrqm_term + b_unique_term, 4),
        }

    payload = {
        "description": (
            "Composite score: composite = vif + a * nrqm + b * unique. "
            "The coefficients are chosen from the theoretical ranges of "
            "the pyiqa metrics: a = 1 / NRQM_RANGE_MAX = 1/10 and "
            "b = 1 / UNIQUE_RANGE_MAX = 1/3. With these, a * nrqm lies in "
            "[0, 1] over the full NRQM range [0, 10], and b * unique lies "
            "in [-1, 1] over the full UNIQUE range [-3, 3], giving a "
            "symmetric penalty for degraded UNIQUE while keeping the "
            "positive half mapped into [0, 1]. VIF enters without a "
            "coefficient because it is already bounded in [0, 1]. The "
            "composite has no upper cap: a model that exceeds the "
            "theoretical metric maxima legitimately scores above 1.0 in "
            "the corresponding term."
        ),
        "source": str(baseline_path.relative_to(ROOT)),
        "scenes_used": args.scenes,
        "baseline_means": {
            "vif": round_sig(vif_mean, 6),
            "nrqm": round_sig(nrqm_mean, 6),
            "unique": round_sig(unique_mean, 6),
        },
        "baseline_minmax": {
            "vif": {
                "min": round_sig(ranges["vif"]["min"], 6),
                "max": round_sig(ranges["vif"]["max"], 6),
            },
            "nrqm": {
                "min": round_sig(ranges["nrqm"]["min"], 6),
                "max": round_sig(ranges["nrqm"]["max"], 6),
            },
            "unique": {
                "min": round_sig(ranges["unique"]["min"], 6),
                "max": round_sig(ranges["unique"]["max"], 6),
            },
        },
        "metric_theoretical_range": {
            "nrqm": {"min": 0.0, "max": NRQM_RANGE_MAX},
            "unique": {"min": -UNIQUE_RANGE_MAX, "max": UNIQUE_RANGE_MAX},
        },
        "mode": COMPOSITE_MODE,
        "user_formula": {
            "form": "composite = vif + a * nrqm + b * unique",
            "a": round_sig(a, 6),
            "b": round_sig(b, 6),
            "mode": COMPOSITE_MODE,
            "a_choice": f"1 / {NRQM_RANGE_MAX} (NRQM theoretical max)",
            "b_choice": f"1 / {UNIQUE_RANGE_MAX} (UNIQUE theoretical max)",
        },
        "per_scene_composite": per_scene_composite,
        "baseline_raw": {
            name: {"vif": m["vif"], "nrqm": m["nrqm"], "unique": m["unique"]}
            for name, m in scenes.items()
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Fit baseline: scenes={args.scenes}  n={n}")
    print(f"  Source      = {baseline_path}")
    print(f"  VIF mean    = {vif_mean:.4f}")
    print(f"  NRQM mean   = {nrqm_mean:.4f}")
    print(f"  UNIQUE mean = {unique_mean:.4f}")
    print()
    print("Theoretical metric ranges:")
    print(f"  NRQM   in [0, {NRQM_RANGE_MAX}]")
    print(f"  UNIQUE in [-{UNIQUE_RANGE_MAX}, +{UNIQUE_RANGE_MAX}]")
    print()
    print(f"Chosen weights: a = 1 / NRQM_RANGE_MAX   = {a:.6f}")
    print(f"                b = 1 / UNIQUE_RANGE_MAX = {b:.6f}")
    print(f"Composite form: vif + {a:.4f} * nrqm + {b:.4f} * unique")
    print()
    print("Per-scene composite at baseline:")
    for name, comp in per_scene_composite.items():
        print(
            f"  {name:6s}: vif={comp['vif_term']:.3f} "
            f"+ a*nrqm={comp['a_nrqm_term']:.3f} "
            f"+ b*unique={comp['b_unique_term']:.3f}  "
            f"= {comp['composite']:.3f}"
        )
    print()
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
