"""
Extract only the val-split RAW/YUV frames into small "mini" scene files.

Problem: free Drive only has 15 GB but the full scene files are ~19 GB.
Day-13 training reads RAW/YUV scene files only during per-scene validation,
and val uses just 25 specific frames (12 day + 13 night) out of hundreds.
This script copies those frames into tiny scene files and writes a new
splits file that points at them with consecutive frame indices.

Outputs:
  data/day_val_mini.bin, data/day_val_mini.yuv     (~125 MB)
  data/night_val_mini.bin, data/night_val_mini.yuv (~135 MB)
  dataset/splits_mini.json
"""

import argparse
import json
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def raw_frame_size(width: int, height: int) -> int:
    # 16-bit RAW, full height (emb lines included — stripped only at read time)
    return width * height * 2


def yuv_frame_size(width: int, out_height: int) -> int:
    # NV12 over the post-trim height (what NV12VideoReader expects)
    return int(width * out_height * 1.5)


def extract_frames(src_path: Path, dst_path: Path, frame_indices: list, frame_size: int) -> int:
    """Copy selected frames from src to dst. Returns number of frames written."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        for idx in frame_indices:
            src.seek(idx * frame_size)
            buf = src.read(frame_size)
            if len(buf) < frame_size:
                raise RuntimeError(
                    f"Short read at frame {idx} of {src_path}: "
                    f"got {len(buf)} bytes, expected {frame_size}"
                )
            dst.write(buf)
            written += 1
    return written


def normalize_path(p: str) -> str:
    """Windows-style 'data\\x.bin' -> 'data/x.bin' for cross-platform JSON."""
    return p.replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="dataset/splits.json")
    parser.add_argument("--config", default="data/imx623.toml")
    parser.add_argument("--out-splits", default="dataset/splits_mini.json")
    parser.add_argument("--data-dir", default="data", help="Where to write *_val_mini.* files")
    args = parser.parse_args()

    splits_path = ROOT / args.splits
    config_path = ROOT / args.config
    out_splits_path = ROOT / args.out_splits
    data_dir = ROOT / args.data_dir

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    width = cfg["img"]["width"]
    height = cfg["img"]["height"]
    top, bot = cfg["img"]["emb_lines"]
    out_height = height - top - bot

    raw_sz = raw_frame_size(width, height)
    yuv_sz = yuv_frame_size(width, out_height)
    print(
        f"Frame sizes: RAW={raw_sz:,} B ({raw_sz / 1e6:.2f} MB), "
        f"YUV={yuv_sz:,} B ({yuv_sz / 1e6:.2f} MB)"
    )

    val_items = splits["splits"].get("val", [])
    if not val_items:
        print("No val items in splits; nothing to do.")
        sys.exit(1)

    new_val_items = []
    total_bytes = 0

    for item in val_items:
        scene = item["scene"]
        src_raw = ROOT / item["raw_path"].replace("\\", "/")
        src_yuv = ROOT / item["yuv_path"].replace("\\", "/")
        indices = list(item["frame_indices"])
        n = len(indices)

        dst_raw = data_dir / f"{scene}_val_mini.bin"
        dst_yuv = data_dir / f"{scene}_val_mini.yuv"

        print(f"\n[{scene}] {n} frames from {src_raw.name}/{src_yuv.name}")
        print(
            f"  RAW src size: {src_raw.stat().st_size / 1e9:.2f} GB -> "
            f"{dst_raw.name} ({n * raw_sz / 1e6:.1f} MB)"
        )
        n_raw = extract_frames(src_raw, dst_raw, indices, raw_sz)
        n_yuv = extract_frames(src_yuv, dst_yuv, indices, yuv_sz)
        assert n_raw == n == n_yuv

        total_bytes += dst_raw.stat().st_size + dst_yuv.stat().st_size

        new_item = dict(item)
        new_item["raw_path"] = normalize_path(str(dst_raw.relative_to(ROOT)))
        new_item["yuv_path"] = normalize_path(str(dst_yuv.relative_to(ROOT)))
        new_item["frame_indices"] = list(range(n))
        new_item["original_frame_indices"] = indices  # for traceability
        new_val_items.append(new_item)

    # Build new splits: keep train/test sections intact (training uses H5 patches,
    # not scene files; test is final test stage and not packed for Colab).
    new_splits = dict(splits)
    new_splits["splits"] = dict(splits["splits"])
    new_splits["splits"]["val"] = new_val_items

    # Normalize slashes in train/test too so the JSON is cross-platform.
    for split_name in ("train", "test", "test_quick"):
        if split_name in new_splits["splits"]:
            new_splits["splits"][split_name] = [
                {
                    **it,
                    "raw_path": normalize_path(it["raw_path"]),
                    "yuv_path": normalize_path(it["yuv_path"]),
                }
                for it in new_splits["splits"][split_name]
            ]

    out_splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_splits_path, "w", encoding="utf-8") as f:
        json.dump(new_splits, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Total mini scene bytes: {total_bytes / 1e6:.1f} MB")
    print(f"Wrote {out_splits_path}")
    print("\nFor Colab, upload to Drive only:")
    print("  dataset/train_patches.h5")
    print(f"  dataset/{out_splits_path.name}")
    print("  data/imx623.toml")
    for it in new_val_items:
        print(f"  {it['raw_path']}")
        print(f"  {it['yuv_path']}")
    print("  artifacts/checkpoints/cnn_pretrained.pth")


if __name__ == "__main__":
    main()
