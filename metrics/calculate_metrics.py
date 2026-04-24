"""
Image quality metric calculation:

- VIF (Visual Information Fidelity) - comparison of CFA (Bayer) and the Y channel
- NRQM (No-Reference Quality Metric)
- UNIQUE (Unsupervised Image Quality Estimation)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyiqa
import torch
from PIL import Image
from tqdm import tqdm

try:
    from vif import vif_cfa_to_y

    from isp.color.conversions import nv12_uint8_to_rgb_bt709_full
    from isp.config.config_reader import read_config
    from isp.io.video_reader import RAWVideoReader
    from isp.io.yuv_reader import NV12VideoReader
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    METRICS_DIR = Path(__file__).resolve().parent

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(METRICS_DIR) not in sys.path:
        sys.path.insert(0, str(METRICS_DIR))

    from vif import vif_cfa_to_y

    from isp.color.conversions import nv12_uint8_to_rgb_bt709_full
    from isp.config.config_reader import read_config
    from isp.io.video_reader import RAWVideoReader
    from isp.io.yuv_reader import NV12VideoReader


def save_first_frame_visualization(
    y: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    rgb: torch.Tensor,
    output_path: Path,
    verbose: bool = True,
):
    """
    Save a detailed visualization of the first frame with YUV components and RGB.

    Args:
        y: Y channel [1, 1, H, W]
        u: U channel [1, 1, H/2, W/2]
        v: V channel [1, 1, H/2, W/2]
        rgb: RGB image [1, 3, H, W]
        output_path: path for saving the visualization
        verbose: whether to print status information
    """
    try:
        import matplotlib.pyplot as plt

        y_np = y[0, 0].cpu().numpy()
        u_np = u[0, 0].cpu().numpy()
        v_np = v[0, 0].cpu().numpy()
        rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            "First Frame - YUV to RGB Conversion Analysis",
            fontsize=16,
            fontweight="bold",
        )

        axes[0, 0].imshow(y_np, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title(f"Y Channel (Luma)\nRange: [{y_np.min():.0f}, {y_np.max():.0f}]")
        axes[0, 0].axis("off")

        im_u = axes[0, 1].imshow(u_np, cmap="RdBu_r", vmin=0, vmax=255)
        u_dev = np.abs(u_np - 128.0).mean()
        axes[0, 1].set_title(f"U Channel (Cb)\nDeviation from 128: {u_dev:.2f}")
        axes[0, 1].axis("off")
        plt.colorbar(im_u, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im_v = axes[0, 2].imshow(v_np, cmap="RdBu_r", vmin=0, vmax=255)
        v_dev = np.abs(v_np - 128.0).mean()
        axes[0, 2].set_title(f"V Channel (Cr)\nDeviation from 128: {v_dev:.2f}")
        axes[0, 2].axis("off")
        plt.colorbar(im_v, ax=axes[0, 2], fraction=0.046, pad=0.04)

        axes[1, 0].imshow(rgb_np)
        axes[1, 0].set_title("RGB Result")
        axes[1, 0].axis("off")

        axes[1, 1].hist(rgb_np[:, :, 0].flatten(), bins=50, alpha=0.5, color="red", label="R")
        axes[1, 1].hist(
            rgb_np[:, :, 1].flatten(),
            bins=50,
            alpha=0.5,
            color="green",
            label="G",
        )
        axes[1, 1].hist(rgb_np[:, :, 2].flatten(), bins=50, alpha=0.5, color="blue", label="B")
        axes[1, 1].set_title("RGB Histogram")
        axes[1, 1].set_xlabel("Value")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].axis("off")
        r_mean = rgb_np[:, :, 0].mean()
        g_mean = rgb_np[:, :, 1].mean()
        b_mean = rgb_np[:, :, 2].mean()
        channel_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))

        stats_text = f"""Statistics:

RGB Means:
R: {r_mean:.4f}
G: {g_mean:.4f}
B: {b_mean:.4f}

Channel Diff: {channel_diff:.4f}

YUV Deviations:
U: {u_dev:.2f}
V: {v_dev:.2f}
"""
        if channel_diff < 0.01:
            stats_text += "\n WARNING:\nRGB appears grayscale!"
            text_color = "red"
        else:
            stats_text += "\n OK: RGB has color"
            text_color = "green"

        axes[1, 2].text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            verticalalignment="center",
            fontfamily="monospace",
            color=text_color,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        if verbose:
            print(f"Detailed visualization saved to: {output_path}")

        return True

    except ImportError:
        if verbose:
            print("Warning: matplotlib not available, skipping detailed visualization")
        return False
    except Exception as e:
        if verbose:
            print(f"Warning: Error saving visualization: {e}")
        return False


def calculate_metrics(
    bin_path: str,
    yuv_path: str,
    config_path: str,
    output_csv: str | None = None,
    max_frames: int | None = None,
    device: str = "cuda",
    verbose: bool = True,
    save_first_frame: bool = True,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Calculate VIF, NRQM, and UNIQUE for a BIN/YUV pair.

    Args:
        bin_path: path to the RAW video file (.bin)
        yuv_path: path to the YUV file in NV12 format
        config_path: path to the camera configuration file (.toml)
        output_csv: path to the CSV file for saving results
        max_frames: maximum number of frames to process (None = all)
        device: execution device ("cuda" or "cpu")
        verbose: whether to print progress information
        save_first_frame: whether to save the first frame as PNG
        output_dir: directory for saving images
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    if output_dir is None and output_csv is not None:
        output_dir = str(Path(output_csv).parent)
    elif output_dir is None:
        output_dir = "."
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading configuration from {config_path}...")

    config = read_config(config_path, device=device)

    width = config["img"]["width"]
    height = config["img"]["height"]
    top, bottom = config["img"]["emb_lines"]
    out_h = height - top - bottom
    pattern = config["img"].get("bayer_pattern", "RGGB")

    if verbose:
        print("Video parameters:")
        print(f"  Resolution: {width}x{height} -> {width}x{out_h}")
        print(f"  Bayer pattern: {pattern}")
        print(f"  Device: {device}")
        print("\nInitializing pyiqa metrics...")

    nrqm = pyiqa.create_metric("nrqm", device=torch.device(device))
    unique = pyiqa.create_metric("unique", device=torch.device(device))

    if verbose:
        print("  NRQM: initialized")
        print("  UNIQUE: initialized")
        print("\nProcessing frames...")
        print(f"BIN: {bin_path}")
        print(f"YUV: {yuv_path}\n")

    results: dict[str, list] = {"frame": [], "vif": [], "nrqm": [], "unique": []}

    frame_count = 0

    with (
        RAWVideoReader(bin_path, config, device=device) as bin_reader,
        NV12VideoReader(yuv_path, width, out_h, device=device) as yuv_reader,
    ):
        bin_iter = iter(bin_reader)
        yuv_iter = iter(yuv_reader)

        if verbose and max_frames:
            pbar = tqdm(total=max_frames, desc="Processing")
        else:
            pbar = tqdm(desc="Processing", disable=not verbose)

        with torch.no_grad():
            while True:
                try:
                    raw_frame, idx = next(bin_iter)
                    yuv_frame, _ = next(yuv_iter)
                except StopIteration:
                    break

                y, u, v = yuv_frame

                if frame_count == 0 and verbose:
                    u_mean = u.float().mean().item()
                    v_mean = v.float().mean().item()
                    u_std = u.float().std().item()
                    v_std = v.float().std().item()
                    print("\nFirst frame color check:")
                    print(f"  U channel: mean={u_mean:.1f}, std={u_std:.1f}")
                    print(f"  V channel: mean={v_mean:.1f}, std={v_std:.1f}")
                    if abs(u_mean - 128) < 5 and abs(v_mean - 128) < 5 and u_std < 5 and v_std < 5:
                        print("  WARNING: UV channels appear neutral (grayscale)!")
                    else:
                        print("  OK: Color information detected in UV channels")

                cfa = raw_frame.unsqueeze(0).unsqueeze(0)
                cfa_16bit = (cfa.float() * (65535.0 / 4095.0)).clamp(0, 65535).to(torch.int32)

                vif_score = vif_cfa_to_y(cfa=cfa_16bit, y=y, pattern=pattern, even=True)

                rgb = nv12_uint8_to_rgb_bt709_full(y, u, v)

                if frame_count == 0 and save_first_frame:
                    rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()
                    rgb_uint8 = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(rgb_uint8)

                    yuv_name = Path(yuv_path).stem
                    frame_path = output_dir_path / f"{yuv_name}_frame_0000.png"
                    img.save(frame_path)

                    if verbose:
                        print(f"\nFirst frame saved to: {frame_path}")

                    detailed_path = output_dir_path / f"{yuv_name}_frame_0000_analysis.png"
                    save_first_frame_visualization(y, u, v, rgb, detailed_path, verbose)

                    if verbose:
                        r_mean = rgb[0, 0].mean().item()
                        g_mean = rgb[0, 1].mean().item()
                        b_mean = rgb[0, 2].mean().item()
                        channel_diff = max(
                            abs(r_mean - g_mean),
                            abs(g_mean - b_mean),
                            abs(r_mean - b_mean),
                        )
                        print("\nRGB conversion check:")
                        print(f"  R={r_mean:.4f}, G={g_mean:.4f}, B={b_mean:.4f}")
                        print(f"  Channel difference: {channel_diff:.4f}")
                        if channel_diff < 0.01:
                            print("  WARNING: RGB appears grayscale!")
                        else:
                            print("  OK: RGB has color variation")
                        print()

                nrqm_score = nrqm(rgb)
                unique_score = unique(rgb)

                results["frame"].append(idx)
                results["vif"].append(vif_score.item())
                results["nrqm"].append(nrqm_score.item())
                results["unique"].append(unique_score.item())

                frame_count += 1
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "VIF": f"{vif_score.item():.4f}",
                        "NRQM": f"{nrqm_score.item():.4f}",
                        "UNIQUE": f"{unique_score.item():.4f}",
                    }
                )

                if max_frames and frame_count >= max_frames:
                    if verbose:
                        print(f"\nReached max_frames limit ({max_frames})")
                    break

        pbar.close()

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n{'=' * 80}")
        print("Metrics Summary:")
        print(f"{'=' * 80}")
        print(f"Total frames processed: {frame_count}")
        print("\nVIF (Visual Information Fidelity):")
        print(f"  Mean:   {df['vif'].mean():.4f}")
        print(f"  Std:    {df['vif'].std():.4f}")
        print(f"  Min:    {df['vif'].min():.4f}")
        print(f"  Max:    {df['vif'].max():.4f}")
        print("\nNRQM (No-Reference Quality Metric):")
        print(f"  Mean:   {df['nrqm'].mean():.4f}")
        print(f"  Std:    {df['nrqm'].std():.4f}")
        print(f"  Min:    {df['nrqm'].min():.4f}")
        print(f"  Max:    {df['nrqm'].max():.4f}")
        print("\nUNIQUE (Unsupervised Image Quality):")
        print(f"  Mean:   {df['unique'].mean():.4f}")
        print(f"  Std:    {df['unique'].std():.4f}")
        print(f"  Min:    {df['unique'].min():.4f}")
        print(f"  Max:    {df['unique'].max():.4f}")
        print(f"{'=' * 80}")

    if output_csv:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nResults saved to: {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Calculate image quality metrics")

    parser.add_argument("--bin", required=True, help="Path to RAW .bin video")
    parser.add_argument("--yuv", required=True, help="Path to YUV NV12 video")
    parser.add_argument("--config", required=True, help="Path to camera config (.toml)")
    parser.add_argument("--output", default=None, help="Path to output CSV")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()

    for path in [args.bin, args.yuv, args.config]:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    calculate_metrics(
        bin_path=args.bin,
        yuv_path=args.yuv,
        config_path=args.config,
        output_csv=args.output,
        max_frames=args.max_frames,
        device=args.device,
    )


if __name__ == "__main__":
    main()
