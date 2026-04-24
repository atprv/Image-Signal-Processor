"""
Recalculate baseline metrics for the current traditional ISP.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from isp.config.config_reader import read_config
from isp.evaluation.evaluation_utils import evaluate
from isp.pipeline.pipeline import ISPPipeline

ISP_PARAMS_DAY = {
    "denoise_eps": 1e-12,
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
}

ISP_PARAMS_NIGHT = {
    "denoise_eps": 1e-12,
    "ltm_a": 0.3,
    "ltm_detail_gain": 8,
    "ltm_detail_threshold": 0.4,
    "sharp_amount": 0.8,
}

ISP_PARAMS_TUNNEL = {}


SCENES = [
    {
        "name": "day",
        "scene_id": 0,
        "raw": "data/day_0.bin",
        "yuv": "data/day_0.yuv",
        "isp_params": ISP_PARAMS_DAY,
    },
    {
        "name": "night",
        "scene_id": 1,
        "raw": "data/night_0.bin",
        "yuv": "data/night_0.yuv",
        "isp_params": ISP_PARAMS_NIGHT,
    },
    {
        "name": "tunnel",
        "scene_id": 2,
        "raw": "data/tunnel_0.bin",
        "yuv": "data/tunnel_0.yuv",
        "isp_params": ISP_PARAMS_TUNNEL,
    },
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = str(ROOT / "data" / "imx623.toml")
    config = read_config(config_path, device=device)

    width = int(config["img"]["width"])
    height = int(config["img"]["height"])
    top, bottom = config["img"]["emb_lines"]
    out_h = height - top - bottom
    print(f"Device: {device} | Resolution: {width}x{height} -> {width}x{out_h}\n")

    all_scene_results = {}
    t0_total = time.time()

    for scene in SCENES:
        name = scene["name"]
        params = scene["isp_params"]
        print(f"{'=' * 60}")
        print(f"Scene: {name}  |  overrides: {params}")
        print(f"{'=' * 60}")

        isp = ISPPipeline(config, device=device, **params)

        eval_items = [
            {
                "scene": name,
                "scene_id": scene["scene_id"],
                "raw_path": str(ROOT / scene["raw"]),
                "yuv_path": str(ROOT / scene["yuv"]),
                "frame_indices": [0],
                "frame_count": 1,
            }
        ]

        t0 = time.time()
        result = evaluate(
            isp=isp,
            model=None,
            eval_items=eval_items,
            config_path=config_path,
            device=device,
            compute_iqa=True,
            max_frames=None,
            verbose=True,
        )
        elapsed = time.time() - t0

        all_scene_results[name] = result
        print(
            f"  L1_Y={result['l1_y']:.6f}  L1_UV={result['l1_uv']:.6f}  "
            f"VIF={result['vif']:.6f}  NRQM={result['nrqm']:.6f}  "
            f"UNIQUE={result['unique']:.6f}  ({elapsed:.1f}s)\n"
        )

        del isp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_total = time.time() - t0_total

    n = len(all_scene_results)
    agg = {
        "l1_y": sum(r["l1_y"] for r in all_scene_results.values()) / n,
        "l1_uv": sum(r["l1_uv"] for r in all_scene_results.values()) / n,
        "vif": sum(r["vif"] for r in all_scene_results.values()) / n,
        "nrqm": sum(r["nrqm"] for r in all_scene_results.values()) / n,
        "unique": sum(r["unique"] for r in all_scene_results.values()) / n,
    }

    print("\n" + "=" * 60)
    print("BASELINE METRICS (per-scene ISP params, even=False)")
    print("=" * 60)
    header = f"{'Metric':<10} {'Aggregate':>10} {'Day':>10} {'Night':>10} {'Tunnel':>10}"
    print(header)
    print("-" * len(header))
    for m in ["l1_y", "l1_uv", "vif", "nrqm", "unique"]:
        row = f"{m:<10} {agg[m]:>10.6f}"
        for name in ["day", "night", "tunnel"]:
            row += f" {all_scene_results[name][m]:>10.6f}"
        print(row)
    print(f"\nTotal time: {elapsed_total:.1f}s")

    out_dir = ROOT / "artifacts" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_metrics.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Baseline metrics (per-scene ISP params, VIF even=False)\n\n")
        f.write("Run info\n")
        f.write(f"- device: {device}\n")
        f.write("- config: data/imx623.toml\n")
        f.write("- frames: 1 per scene (day, night, tunnel)\n")
        f.write("- VIF: even=False\n")
        f.write(f"- elapsed: {elapsed_total:.1f}s\n\n")

        f.write("Aggregate (mean over 3 scenes)\n")
        for m in ["l1_y", "l1_uv", "vif", "nrqm", "unique"]:
            f.write(f"- {m}: {agg[m]:.6f}\n")
        f.write("\n")

        for scene in SCENES:
            name = scene["name"]
            r = all_scene_results[name]
            f.write(f"Per-scene: {name}\n")
            f.write(f"- isp_params: {scene['isp_params']}\n")
            for m in ["l1_y", "l1_uv", "vif", "nrqm", "unique"]:
                f.write(f"- {m}: {r[m]:.6f}\n")
            f.write("\n")

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
