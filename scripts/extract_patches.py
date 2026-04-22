import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

try:
    from isp.config.config_reader import read_config
    from isp.io.video_reader import RAWVideoReader
    from isp.io.yuv_reader import NV12VideoReader
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from isp.config.config_reader import read_config
    from isp.io.video_reader import RAWVideoReader
    from isp.io.yuv_reader import NV12VideoReader


SCENE_TO_ID = {"day": 0, "night": 1, "tunnel": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract RAW/Y/UV patches for ISP training")

    parser.add_argument(
        "--data-dir", type=str, default="data", help="Path to directory with RAW/YUV files"
    )
    parser.add_argument(
        "--config", type=str, default="data/imx623.toml", help="Path to camera config (TOML)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset", help="Path to output dataset directory"
    )
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size for RAW and Y")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride")
    parser.add_argument(
        "--debug-samples", type=int, default=3, help="Number of debug patch groups to save"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    return parser.parse_args()


def build_split_spec(data_dir: Path) -> dict[str, list[dict[str, object]]]:
    return {
        "train": [
            {
                "scene": "day",
                "scene_id": scene_name_to_id("day"),
                "raw_path": str(data_dir / "day_0.bin"),
                "yuv_path": str(data_dir / "day_0.yuv"),
                "frame_indices": list(range(0, 264, 6)),
            },
            {
                "scene": "night",
                "scene_id": scene_name_to_id("night"),
                "raw_path": str(data_dir / "night_0.bin"),
                "yuv_path": str(data_dir / "night_0.yuv"),
                "frame_indices": list(range(0, 1008, 20)),
            },
        ],
        "val": [
            {
                "scene": "day",
                "scene_id": scene_name_to_id("day"),
                "raw_path": str(data_dir / "day_0.bin"),
                "yuv_path": str(data_dir / "day_0.yuv"),
                "frame_indices": list(range(264, 331, 6)),
            },
            {
                "scene": "night",
                "scene_id": scene_name_to_id("night"),
                "raw_path": str(data_dir / "night_0.bin"),
                "yuv_path": str(data_dir / "night_0.yuv"),
                "frame_indices": list(range(1008, 1261, 20)),
            },
        ],
        "test": [
            {
                "scene": "tunnel",
                "scene_id": scene_name_to_id("tunnel"),
                "raw_path": str(data_dir / "tunnel_0.bin"),
                "yuv_path": str(data_dir / "tunnel_0.yuv"),
                "frame_indices": list(range(0, 413, 1)),
            },
        ],
        "test_quick": [
            {
                "scene": "tunnel",
                "scene_id": scene_name_to_id("tunnel"),
                "raw_path": str(data_dir / "tunnel_0.bin"),
                "yuv_path": str(data_dir / "tunnel_0.yuv"),
                "frame_indices": list(range(0, 413, 10)),
            },
        ],
    }


def build_patch_grid(
    width: int, height: int, patch_size: int, stride: int
) -> list[tuple[int, int]]:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if patch_size > width or patch_size > height:
        raise ValueError(f"patch_size {patch_size} does not fit into frame {width}x{height}")

    xs = list(range(0, width - patch_size + 1, stride))
    ys = list(range(0, height - patch_size + 1, stride))
    grid = [(x, y) for y in ys for x in xs]

    if not grid:
        raise ValueError("Patch grid is empty")

    for x, y in grid:
        if x % 2 != 0 or y % 2 != 0:
            raise ValueError(f"Patch coordinates must be even for CFA alignment, got x={x}, y={y}")

    return grid


def count_split_patches(split_items: list[dict[str, object]], grid_size: int) -> int:
    total = 0
    for item in split_items:
        total += len(item["frame_indices"]) * grid_size
    return total


def scene_name_to_id(scene_name: str) -> int:
    if scene_name not in SCENE_TO_ID:
        raise KeyError(f"Unknown scene name: {scene_name}")
    return SCENE_TO_ID[scene_name]


def create_h5_file(path: Path, total_patches: int, patch_size: int):
    uv_patch_size = patch_size // 2
    chunk_rows = max(1, min(16, total_patches))

    h5_file = h5py.File(path, "w")
    h5_file.create_dataset(
        "raw",
        shape=(total_patches, patch_size, patch_size),
        dtype=np.uint16,
        chunks=(chunk_rows, patch_size, patch_size),
    )
    h5_file.create_dataset(
        "y_ref",
        shape=(total_patches, patch_size, patch_size),
        dtype=np.uint8,
        chunks=(chunk_rows, patch_size, patch_size),
    )
    h5_file.create_dataset(
        "uv_ref",
        shape=(total_patches, 2, uv_patch_size, uv_patch_size),
        dtype=np.uint8,
        chunks=(chunk_rows, 2, uv_patch_size, uv_patch_size),
    )
    h5_file.create_dataset("scene_id", shape=(total_patches,), dtype=np.uint8, chunks=(chunk_rows,))
    h5_file.create_dataset(
        "frame_idx", shape=(total_patches,), dtype=np.int32, chunks=(chunk_rows,)
    )
    h5_file.create_dataset("x", shape=(total_patches,), dtype=np.int32, chunks=(chunk_rows,))
    h5_file.create_dataset("y", shape=(total_patches,), dtype=np.int32, chunks=(chunk_rows,))

    h5_file.attrs["patch_size"] = patch_size
    h5_file.attrs["uv_patch_size"] = uv_patch_size
    h5_file.attrs["storage"] = "raw:uint16, y_ref:uint8, uv_ref:uint8"

    return h5_file


def extract_uv_patch(
    u_plane: np.ndarray, v_plane: np.ndarray, x: int, y: int, patch_size: int
) -> np.ndarray:
    uv_patch_size = patch_size // 2
    uv_x = x // 2
    uv_y = y // 2

    u_patch = u_plane[uv_y : uv_y + uv_patch_size, uv_x : uv_x + uv_patch_size]
    v_patch = v_plane[uv_y : uv_y + uv_patch_size, uv_x : uv_x + uv_patch_size]

    if u_patch.shape != (uv_patch_size, uv_patch_size) or v_patch.shape != (
        uv_patch_size,
        uv_patch_size,
    ):
        raise ValueError(f"Unexpected UV patch shape at x={x}, y={y}")

    return np.stack([u_patch, v_patch], axis=0)


def write_split(
    split_name: str,
    split_items: list[dict[str, object]],
    h5_path: Path,
    config: dict[str, object],
    patch_size: int,
    patch_grid: list[tuple[int, int]],
    debug_collector: list[dict[str, object]],
    debug_limit: int,
):
    width = config["img"]["width"]
    height = config["img"]["height"]
    top_lines, bottom_lines = config["img"]["emb_lines"]
    out_height = height - top_lines - bottom_lines

    if not patch_grid:
        raise ValueError("patch_grid must not be empty")

    total_patches = count_split_patches(split_items, len(patch_grid))
    print(f"[{split_name}] creating {h5_path} with {total_patches} patches")

    patch_side = patch_size
    h5_file = create_h5_file(h5_path, total_patches, patch_side)

    try:
        write_index = 0
        for item in split_items:
            scene_name = item["scene"]
            scene_id = item["scene_id"]
            raw_path = item["raw_path"]
            yuv_path = item["yuv_path"]
            selected_frames = set(item["frame_indices"])
            processed_frames = 0

            print(f"[{split_name}] scene={scene_name} frames={len(selected_frames)}")

            with (
                RAWVideoReader(raw_path, config, device="cpu") as raw_reader,
                NV12VideoReader(yuv_path, width, out_height, device="cpu") as yuv_reader,
            ):
                frame_pairs = zip(raw_reader, yuv_reader, strict=False)
                for (raw_frame, raw_number), (yuv_frame, yuv_number) in frame_pairs:
                    if raw_number != yuv_number:
                        raise RuntimeError(f"RAW/YUV frame mismatch: {raw_number} vs {yuv_number}")

                    frame_idx = raw_number - 1
                    if frame_idx not in selected_frames:
                        continue

                    processed_frames += 1

                    y_plane, u_plane, v_plane = yuv_frame
                    raw_np = raw_frame.cpu().numpy()
                    y_np = y_plane[0, 0].cpu().numpy()
                    u_np = u_plane[0, 0].cpu().numpy()
                    v_np = v_plane[0, 0].cpu().numpy()

                    per_frame_count = len(patch_grid)
                    raw_batch = np.empty((per_frame_count, patch_side, patch_side), dtype=np.uint16)
                    y_batch = np.empty((per_frame_count, patch_side, patch_side), dtype=np.uint8)
                    uv_batch = np.empty(
                        (per_frame_count, 2, patch_side // 2, patch_side // 2), dtype=np.uint8
                    )
                    scene_batch = np.full((per_frame_count,), scene_id, dtype=np.uint8)
                    frame_batch = np.full((per_frame_count,), frame_idx, dtype=np.int32)
                    x_batch = np.empty((per_frame_count,), dtype=np.int32)
                    y_coord_batch = np.empty((per_frame_count,), dtype=np.int32)

                    for patch_idx, (x, y) in enumerate(patch_grid):
                        raw_patch = raw_np[y : y + patch_side, x : x + patch_side]
                        y_patch = y_np[y : y + patch_side, x : x + patch_side]
                        uv_patch = extract_uv_patch(u_np, v_np, x, y, patch_side)

                        raw_batch[patch_idx] = raw_patch
                        y_batch[patch_idx] = y_patch
                        uv_batch[patch_idx] = uv_patch
                        x_batch[patch_idx] = x
                        y_coord_batch[patch_idx] = y

                        if len(debug_collector) < debug_limit:
                            debug_collector.append(
                                {
                                    "split": split_name,
                                    "scene": scene_name,
                                    "frame_idx": frame_idx,
                                    "x": x,
                                    "y": y,
                                    "raw_patch": raw_patch.copy(),
                                    "y_patch": y_patch.copy(),
                                    "u_patch": uv_patch[0].copy(),
                                    "v_patch": uv_patch[1].copy(),
                                }
                            )

                    end_index = write_index + per_frame_count
                    h5_file["raw"][write_index:end_index] = raw_batch
                    h5_file["y_ref"][write_index:end_index] = y_batch
                    h5_file["uv_ref"][write_index:end_index] = uv_batch
                    h5_file["scene_id"][write_index:end_index] = scene_batch
                    h5_file["frame_idx"][write_index:end_index] = frame_batch
                    h5_file["x"][write_index:end_index] = x_batch
                    h5_file["y"][write_index:end_index] = y_coord_batch
                    write_index = end_index

            if processed_frames != len(selected_frames):
                raise RuntimeError(
                    f"[{split_name}] scene={scene_name} expected {len(selected_frames)} frames, "
                    f"processed {processed_frames}"
                )

            print(f"[{split_name}] scene={scene_name} done, processed_frames={processed_frames}")

        if write_index != total_patches:
            raise RuntimeError(
                f"[{split_name}] expected {total_patches} patches, wrote {write_index}"
            )
    finally:
        h5_file.close()


def save_debug_samples(debug_samples: list[dict[str, object]], output_dir: Path):
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for sample in debug_samples:
        stem = (
            f"{sample['split']}_{sample['scene']}"
            f"_f{sample['frame_idx']:04d}"
            f"_x{sample['x']:04d}"
            f"_y{sample['y']:04d}"
        )

        raw_vis = np.clip(sample["raw_patch"].astype(np.float32), 0.0, 4095.0)
        raw_vis = (raw_vis * (255.0 / 4095.0)).astype(np.uint8)

        Image.fromarray(raw_vis, mode="L").save(debug_dir / f"{stem}_raw.png")
        Image.fromarray(sample["y_patch"], mode="L").save(debug_dir / f"{stem}_y.png")
        Image.fromarray(sample["u_patch"], mode="L").save(debug_dir / f"{stem}_u.png")
        Image.fromarray(sample["v_patch"], mode="L").save(debug_dir / f"{stem}_v.png")


def write_splits_json(
    path: Path, split_spec: dict[str, list[dict[str, object]]], patch_size: int, stride: int
):
    serializable = {
        "patch_size": patch_size,
        "stride": stride,
        "splits": {},
    }

    for split_name, items in split_spec.items():
        serializable["splits"][split_name] = []
        for item in items:
            serializable["splits"][split_name].append(
                {
                    "scene": item["scene"],
                    "scene_id": item["scene_id"],
                    "raw_path": item["raw_path"],
                    "yuv_path": item["yuv_path"],
                    "frame_indices": item["frame_indices"],
                    "frame_count": len(item["frame_indices"]),
                }
            )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def write_test_manifest(path: Path, split_spec: dict[str, list[dict[str, object]]]):
    test_item = split_spec["test"][0]
    quick_item = split_spec["test_quick"][0]
    quick_frames = set(quick_item["frame_indices"])

    fieldnames = ["scene", "frame_idx", "split", "selected_for_quick_eval", "raw_path", "yuv_path"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame_idx in test_item["frame_indices"]:
            writer.writerow(
                {
                    "scene": test_item["scene"],
                    "frame_idx": frame_idx,
                    "split": "test",
                    "selected_for_quick_eval": int(frame_idx in quick_frames),
                    "raw_path": test_item["raw_path"],
                    "yuv_path": test_item["yuv_path"],
                }
            )


def check_output_paths(output_dir: Path, overwrite: bool):
    output_paths = [
        output_dir / "train_patches.h5",
        output_dir / "val_patches.h5",
        output_dir / "splits.json",
        output_dir / "test_manifest.csv",
        output_dir / "debug",
    ]

    if overwrite:
        return

    existing = [path for path in output_paths if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}. Use --overwrite to replace.")


def validate_inputs(split_spec: dict[str, list[dict[str, object]]], config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    for split_items in split_spec.values():
        for item in split_items:
            for key in ["raw_path", "yuv_path"]:
                path = Path(item[key])
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")


def print_split_summary(split_spec: dict[str, list[dict[str, object]]], grid_size: int):
    print("Split summary:")
    for split_name in ["train", "val", "test", "test_quick"]:
        items = split_spec[split_name]
        frame_count = sum(len(item["frame_indices"]) for item in items)
        patch_count = count_split_patches(items, grid_size) if split_name in {"train", "val"} else 0
        print(f"  {split_name:10s} frames={frame_count:4d} patches={patch_count}")


def main():
    args = parse_args()

    if args.patch_size % 2 != 0:
        raise ValueError(f"patch_size must be even, got {args.patch_size}")
    if args.stride % 2 != 0:
        raise ValueError(f"stride must be even, got {args.stride}")

    data_dir = Path(args.data_dir)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    split_spec = build_split_spec(data_dir)
    validate_inputs(split_spec, config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    check_output_paths(output_dir, overwrite=args.overwrite)

    config = read_config(str(config_path), device="cpu")
    width = config["img"]["width"]
    height = config["img"]["height"]
    top_lines, bottom_lines = config["img"]["emb_lines"]
    out_height = height - top_lines - bottom_lines

    patch_grid = build_patch_grid(width, out_height, args.patch_size, args.stride)
    print(f"Frame size after crop: {width}x{out_height}")
    print(f"Patch size: {args.patch_size}, stride: {args.stride}")
    print(f"Patches per frame: {len(patch_grid)}")
    print_split_summary(split_spec, len(patch_grid))

    debug_collector: list[dict[str, object]] = []

    write_split(
        split_name="train",
        split_items=split_spec["train"],
        h5_path=output_dir / "train_patches.h5",
        config=config,
        patch_size=args.patch_size,
        patch_grid=patch_grid,
        debug_collector=debug_collector,
        debug_limit=args.debug_samples,
    )
    write_split(
        split_name="val",
        split_items=split_spec["val"],
        h5_path=output_dir / "val_patches.h5",
        config=config,
        patch_size=args.patch_size,
        patch_grid=patch_grid,
        debug_collector=debug_collector,
        debug_limit=args.debug_samples,
    )

    write_splits_json(output_dir / "splits.json", split_spec, args.patch_size, args.stride)
    write_test_manifest(output_dir / "test_manifest.csv", split_spec)
    save_debug_samples(debug_collector, output_dir)

    print("Extraction complete.")
    print(f"Train patches: {count_split_patches(split_spec['train'], len(patch_grid))}")
    print(f"Val patches:   {count_split_patches(split_spec['val'], len(patch_grid))}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
