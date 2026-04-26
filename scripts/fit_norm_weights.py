"""
Fit and freeze the (a, b) coefficients for the composite quality score.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


BASELINE_PER_SCENE = {
    "day": {
        "vif": 0.702358,
        "nrqm": 5.227967,
        "unique": 0.124453,
        "isp_params": {
            "ltm_a": 0.5,
            "ltm_detail_gain": 30,
            "ltm_detail_threshold": 0.35,
            "hist_target_mean": 0.445,
            "hist_target_std": 0.162,
            "post_denoise_radius": 4,
            "post_denoise_eps": 0.001,
            "raw_y_full_blend": 0.4,
            "sharp_amount": 0.3,
            "saturation": 1.2,
        },
    },
    "night": {
        "vif": 0.519090,
        "nrqm": 7.075381,
        "unique": 0.135025,
        "isp_params": {
            "ltm_a": 0.3,
            "ltm_detail_gain": 8,
            "ltm_detail_threshold": 0.4,
            "sharp_amount": 0.8,
        },
    },
    "tunnel": {
        "vif": 0.693219,
        "nrqm": 6.870870,
        "unique": 0.076290,
        "isp_params": {},
    },
}


def round_sig(x: float, sig: int = 3) -> float:
    """Round to ``sig`` significant figures for stable JSON output."""
    if x == 0.0:
        return 0.0
    import math

    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/baselines/norm_weights.json")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["day", "night", "tunnel"],
        help=(
            "Which baseline scenes to include in the fit. "
            "For a val-focused fit, pass --scenes day night."
        ),
    )
    args = parser.parse_args()

    scenes = {s: BASELINE_PER_SCENE[s] for s in args.scenes}
    n = len(scenes)

    vif_vals = [m["vif"] for m in scenes.values()]
    nrqm_vals = [m["nrqm"] for m in scenes.values()]
    unique_vals = [m["unique"] for m in scenes.values()]

    vif_mean = sum(vif_vals) / n
    nrqm_mean = sum(nrqm_vals) / n
    unique_mean = sum(unique_vals) / n

    a = 1.0
    b = 1.0

    ranges = {
        "vif": {"min": min(vif_vals), "max": max(vif_vals)},
        "nrqm": {"min": min(nrqm_vals), "max": max(nrqm_vals)},
        "unique": {"min": min(unique_vals), "max": max(unique_vals)},
    }

    def norm(metric: str, value: float) -> float:
        lo = ranges[metric]["min"]
        hi = ranges[metric]["max"]
        value = (value - lo) / (hi - lo)
        return max(0.0, min(1.0, value))

    per_scene_composite = {}
    for name, m in scenes.items():
        term_vif = norm("vif", m["vif"])
        term_nrqm = a * norm("nrqm", m["nrqm"])
        term_unique = b * norm("unique", m["unique"])
        per_scene_composite[name] = {
            "vif_norm": round_sig(term_vif, 4),
            "a_nrqm_norm": round_sig(term_nrqm, 4),
            "b_unique_norm": round_sig(term_unique, 4),
            "composite": round_sig(term_vif + term_nrqm + term_unique, 4),
        }

    payload = {
        "description": (
            "Composite score: VIF_norm + a*NRQM_norm + b*UNIQUE_norm, where "
            "each raw metric is baseline min-max normalized and clamped to "
            "[0, 1]. Raw pyiqa UNIQUE may be negative, so UNIQUE_norm is the "
            "score term used for checkpoint selection and Optuna."
        ),
        "source": "artifacts/baselines/baseline_metrics.txt",
        "scenes_used": args.scenes,
        "baseline_means": {
            "vif": round_sig(vif_mean, 6),
            "nrqm": round_sig(nrqm_mean, 6),
            "unique": round_sig(unique_mean, 6),
        },
        "normalization": {
            "mode": "baseline_minmax_clamped",
            "formula": "m_norm = clamp((m - min_baseline) / (max_baseline - min_baseline), 0, 1)",
            "weights": {"vif": 1.0, "nrqm": a, "unique": b},
            "note": "raw UNIQUE can be negative; use unique_norm in the composite",
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
        "user_formula": {
            "form": "composite = VIF_norm + a * NRQM_norm + b * UNIQUE_norm",
            "a": round_sig(a, 4),
            "b": round_sig(b, 4),
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
    print(f"  VIF mean    = {vif_mean:.4f}")
    print(f"  NRQM mean   = {nrqm_mean:.4f}")
    print(f"  UNIQUE mean = {unique_mean:.4f}")
    print()
    print(f"Chosen normalized weights: a = {a:.4f}   b = {b:.4f}")
    print(f"Composite form: VIF_norm + {a:.3f}*NRQM_norm + {b:.3f}*UNIQUE_norm")
    print()
    print("Per-scene composite at baseline:")
    for name, comp in per_scene_composite.items():
        print(
            f"  {name:6s}: VIF_norm={comp['vif_norm']:.3f} "
            f"+ a*NRQM_norm={comp['a_nrqm_norm']:.3f} "
            f"+ b*UNIQUE_norm={comp['b_unique_norm']:.3f}  "
            f"= {comp['composite']:.3f}"
        )
    print()
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
