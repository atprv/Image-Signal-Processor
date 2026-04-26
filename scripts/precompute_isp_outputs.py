"""
Precompute ISP outputs for all patches in an HDF5 dataset.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

from isp.config.config_reader import read_config
from isp.evaluation.evaluation_utils import run_isp_frame
from isp.pipeline.pipeline import ISPPipeline

ISP_PARAMS = {
    "day": dict(
        denoise_eps=1e-12,
        ltm_a=0.5,
        ltm_detail_gain=30,
        ltm_detail_threshold=0.35,
        hist_target_mean=0.445,
        hist_target_std=0.162,
        post_denoise_radius=4,
        post_denoise_eps=0.001,
        raw_y_full_blend=0.4,
        sharp_amount=0.3,
        saturation=1.2,
    ),
    "night": dict(
        denoise_eps=1e-12, ltm_a=0.3, ltm_detail_gain=8, ltm_detail_threshold=0.4, sharp_amount=0.8
    ),
    "tunnel": {},
}

SCENE_ID_TO_NAME = {0: "day", 1: "night", 2: "tunnel"}


def parse_args():
    p = argparse.ArgumentParser(description="Precompute ISP outputs for HDF5 patches")
    p.add_argument("--input-h5", required=True, help="Input HDF5 with raw patches")
    p.add_argument("--output-h5", required=True, help="Output HDF5 with ISP results")
    p.add_argument("--config", default="data/imx623.toml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p).resolve()


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    config_path = resolve(args.config)
    config = read_config(str(config_path), device=device)

    input_path = resolve(args.input_h5)
    output_path = resolve(args.output_h5)

    if not input_path.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    isps = {}
    for scene_name, params in ISP_PARAMS.items():
        isp = ISPPipeline(config, device=device, **params)
        for p in isp.parameters():
            p.requires_grad = False
        isp.eval()
        isps[scene_name] = isp

    with h5py.File(input_path, "r") as h5_in:
        n = h5_in["raw"].shape[0]
        patch_size = h5_in["raw"].shape[1]
        uv_patch_size = patch_size // 2

        print(f"Patches: {n}, patch_size: {patch_size}")

        with h5py.File(output_path, "w") as h5_out:
            for key in ["raw", "y_ref", "uv_ref", "scene_id", "frame_idx", "x", "y"]:
                if key in h5_in:
                    h5_in.copy(key, h5_out)

            chunk_rows = max(1, min(64, n))
            h5_out.create_dataset(
                "y_isp",
                shape=(n, 1, patch_size, patch_size),
                dtype=np.float32,
                chunks=(chunk_rows, 1, patch_size, patch_size),
            )
            h5_out.create_dataset(
                "uv_isp",
                shape=(n, 2, uv_patch_size, uv_patch_size),
                dtype=np.float32,
                chunks=(chunk_rows, 2, uv_patch_size, uv_patch_size),
            )

            h5_out.attrs["has_isp_outputs"] = True
            h5_out.attrs["patch_size"] = patch_size

            t0 = time.time()
            batch_write = 64
            y_buf = np.empty((batch_write, 1, patch_size, patch_size), dtype=np.float32)
            uv_buf = np.empty((batch_write, 2, uv_patch_size, uv_patch_size), dtype=np.float32)
            buf_idx = 0
            write_start = 0

            for i in range(n):
                raw_uint16 = h5_in["raw"][i]
                scene_id = int(h5_in["scene_id"][i])
                scene_name = SCENE_ID_TO_NAME.get(scene_id, "tunnel")
                isp = isps[scene_name]

                raw_tensor = torch.from_numpy(raw_uint16.astype(np.float32)).to(device)

                with torch.no_grad():
                    y_isp, uv_isp = run_isp_frame(
                        isp, raw_tensor, width=patch_size, height=patch_size
                    )

                y_buf[buf_idx] = y_isp.cpu().numpy()
                uv_buf[buf_idx] = uv_isp.cpu().numpy()
                buf_idx += 1

                if buf_idx == batch_write or i == n - 1:
                    end = write_start + buf_idx
                    h5_out["y_isp"][write_start:end] = y_buf[:buf_idx]
                    h5_out["uv_isp"][write_start:end] = uv_buf[:buf_idx]
                    write_start = end
                    buf_idx = 0

                if (i + 1) % 500 == 0 or i == n - 1:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (n - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i + 1:6d}/{n}]  {rate:.1f} patches/s  ETA {eta / 60:.1f} min")

    elapsed_total = time.time() - t0
    print(f"\nDone. {n} patches in {elapsed_total:.1f}s ({n / elapsed_total:.1f} patches/s)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
