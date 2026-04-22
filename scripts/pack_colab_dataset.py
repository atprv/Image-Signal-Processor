"""
Pack a compact dataset for Google Colab training.

Keeps only y_isp, uv_isp (float16), y_ref, uv_ref (uint8).
Drops raw patches and metadata to minimize file size.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np


def pack(input_path, output_path, label):
    print(f"Packing {label}: {input_path} -> {output_path}")

    with h5py.File(input_path, "r") as h5_in:
        n = h5_in["y_isp"].shape[0]

        with h5py.File(output_path, "w") as h5_out:
            # ISP outputs as float16
            y_isp = h5_in["y_isp"][:].astype(np.float16)
            uv_isp = h5_in["uv_isp"][:].astype(np.float16)
            h5_out.create_dataset("y_isp", data=y_isp, compression="gzip", compression_opts=1)
            h5_out.create_dataset("uv_isp", data=uv_isp, compression="gzip", compression_opts=1)

            # References as uint8
            h5_out.create_dataset(
                "y_ref", data=h5_in["y_ref"][:], compression="gzip", compression_opts=1
            )
            h5_out.create_dataset(
                "uv_ref", data=h5_in["uv_ref"][:], compression="gzip", compression_opts=1
            )

            h5_out.attrs["n_patches"] = n
            h5_out.attrs["format"] = "colab_compact"

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  {n} patches, {size_mb:.0f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--output-dir", default="dataset/colab")
    args = p.parse_args()

    ds_dir = Path(args.dataset_dir)
    if not ds_dir.is_absolute():
        ds_dir = ROOT / ds_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pack(ds_dir / "train_precomputed.h5", out_dir / "train_compact.h5", "train")
    pack(ds_dir / "val_precomputed.h5", out_dir / "val_compact.h5", "val")

    print("\nDone. Upload these to Google Drive:")
    for f in sorted(out_dir.glob("*.h5")):
        print(f"  {f}  ({f.stat().st_size / (1024 * 1024):.0f} MB)")


if __name__ == "__main__":
    main()
