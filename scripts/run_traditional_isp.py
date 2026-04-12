import argparse
import time
from pathlib import Path

import torch

from isp.config.config_reader import read_config
from isp.io.video_reader import RAWVideoReader
from isp.io.video_writer import AsyncYUVWriter
from isp.pipeline.pipeline import ISPPipeline


def process_video(
    video_path: str,
    config_path: str,
    output_path: str,
    max_frames: int = None,
    verbose: bool = True,
    device: str = "cuda",
    **isp_params,
):
    """
    Process a RAW video with the ISP pipeline.

    Args:
        video_path: Path to the RAW video file (.bin)
        config_path: Path to the camera config file (.toml)
        output_path: Path to the output YUV file
        max_frames: Maximum number of frames to process (None processes all frames)
        verbose: Whether to print progress information
        device: Compute device ('cuda' or 'cpu')
        **isp_params: Extra ISP pipeline parameters
    """

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if verbose:
        print(f"Loading configuration from {config_path}...")
    config = read_config(config_path, device=device)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        if device == "cuda":
            print(f"Initializing ISP Pipeline on CUDA: {torch.cuda.get_device_name(0)}...")
        else:
            print("Initializing ISP Pipeline on CPU...")
        if isp_params:
            print(f"ISP Parameters: {isp_params}")

    isp = ISPPipeline(config, device=device, **isp_params)

    isp.eval()

    if verbose:
        info = isp.get_pipeline_info()
        print("\nPipeline Info:")
        print(f"  Device: {info['device']}")
        print(f"  Modules: {len(info['modules'])}")
        print(f"\nProcessing video: {video_path}")
        print(f"Output: {output_path}\n")

    frame_count = 0
    total_isp_time = 0.0
    total_write_time = 0.0

    with (
        RAWVideoReader(video_path, config, device=device) as reader,
        AsyncYUVWriter(output_path) as writer,
    ):
        with torch.no_grad():
            if verbose:
                print("Warming up CUDA kernels...")

            warmup_frame, _ = reader.read_frame()
            if warmup_frame is not None:
                _ = isp(warmup_frame)
                if device == "cuda":
                    torch.cuda.synchronize()

            if verbose:
                print("Warmup complete. Starting processing...\n")

            reader.file_handle.seek(0)

            start_time = time.perf_counter()

            for raw_frame, frame_number in reader:
                isp_start = time.perf_counter()

                yuv_frame = isp(raw_frame)

                if device == "cuda":
                    torch.cuda.synchronize()

                isp_time = time.perf_counter() - isp_start
                total_isp_time += isp_time

                write_start = time.perf_counter()

                if yuv_frame.is_cuda:
                    yuv_frame_cpu = yuv_frame.to("cpu", non_blocking=True)
                else:
                    yuv_frame_cpu = yuv_frame

                writer.write(yuv_frame_cpu)

                write_time = time.perf_counter() - write_start
                total_write_time += write_time

                frame_count += 1

                if verbose and frame_count % 10 == 0:
                    instant_fps = 1.0 / isp_time

                    print(
                        f"Frame {frame_number:4d} | "
                        f"ISP: {isp_time * 1000:6.2f}ms | "
                        f"Write: {write_time * 1000:5.2f}ms | "
                        f"Instant FPS: {instant_fps:5.1f} "
                    )

                if max_frames and frame_count >= max_frames:
                    if verbose:
                        print(f"\nReached max_frames limit ({max_frames})")
                    break

    total_elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
    avg_isp_time = total_isp_time / frame_count if frame_count > 0 else 0
    avg_write_time = total_write_time / frame_count if frame_count > 0 else 0
    isp_fps = 1.0 / avg_isp_time if avg_isp_time > 0 else 0.0

    if verbose:
        print(f"\n{'=' * 80}")
        print("Processing complete!")
        print(f"{'=' * 80}")
        print(f"Total frames processed:    {frame_count}")
        print(f"Total time:                {total_elapsed:.2f} s")
        print()
        print(f"Average ISP time:          {avg_isp_time * 1000:.2f} ms/frame")
        print(f"Average write time:        {avg_write_time * 1000:.2f} ms/frame")
        print(f"Average total time:        {(avg_isp_time + avg_write_time) * 1000:.2f} ms/frame")
        print()
        print(f"ISP throughput:            {isp_fps:.2f} FPS (pure processing)")
        print(f"Overall throughput:        {avg_fps:.2f} FPS (with I/O)")
        print()
        print(f"Output saved to: {output_path}")
        print(f"{'=' * 80}")

    return {
        "frames": frame_count,
        "total_time": total_elapsed,
        "avg_fps": avg_fps,
        "avg_isp_time": avg_isp_time,
        "avg_write_time": avg_write_time,
        "isp_fps": isp_fps,
    }


def main() -> int:
    """
    CLI entry point for the ISP pipeline.
    """
    parser = argparse.ArgumentParser(description="ISP Pipeline for RAW video processing")

    parser.add_argument("--video", type=str, required=True, help="Path to RAW video file")
    parser.add_argument("--config", type=str, required=True, help="Path to camera config (TOML)")
    parser.add_argument("--output", type=str, required=True, help="Path to output YUV file")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    parser.add_argument(
        "--awb-max-gain", type=float, default=4.0, help="Max AWB gain (default: 4.0)"
    )
    parser.add_argument(
        "--denoise-radius", type=int, default=2, help="Denoise filter radius (default: 2)"
    )
    parser.add_argument(
        "--denoise-eps", type=float, default=100.0, help="Denoise epsilon (default: 100.0)"
    )
    parser.add_argument(
        "--ltm-a", type=float, default=0.7, help="LTM compression coefficient (default: 0.7)"
    )
    parser.add_argument(
        "--ltm-b", type=float, default=0.0, help="LTM brightness shift (default: 0.0)"
    )
    parser.add_argument(
        "--ltm-radius", type=int, default=8, help="LTM guided filter radius (default: 8)"
    )
    parser.add_argument(
        "--ltm-downsample", type=float, default=0.5, help="Downsample factor for LTM (default: 0.5)"
    )
    parser.add_argument(
        "--ltm-eps",
        type=float,
        default=1e-3,
        help="Epsilon for guided filter in LTM (default: 1e-3)",
    )
    parser.add_argument(
        "--ltm-target-mean",
        type=float,
        default=0.0,
        help="LTM target mean brightness (default: 0.0)",
    )
    parser.add_argument(
        "--ltm-detail-gain", type=float, default=1.0, help="LTM detail gain (default: 1.0)"
    )
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma value (default: 2.2)")
    parser.add_argument(
        "--sharp-amount", type=float, default=0.0, help="Sharpening amount (default: 0.0)"
    )
    parser.add_argument(
        "--sharp-radius", type=float, default=1.0, help="Sharpening Gaussian sigma (default: 1.0)"
    )
    parser.add_argument(
        "--sharp-threshold",
        type=float,
        default=0.01,
        help="Sharpening noise threshold (default: 0.01)",
    )
    parser.add_argument(
        "--awb-lum-mask-low",
        type=float,
        default=0.0,
        help="AWB luminance mask lower bound (default: 0.0)",
    )
    parser.add_argument(
        "--awb-lum-mask-high",
        type=float,
        default=1.0,
        help="AWB luminance mask upper bound (default: 1.0)",
    )
    parser.add_argument(
        "--raw-y-blend",
        type=float,
        default=0.0,
        help="Raw green detail blend into Y channel (default: 0.0)",
    )
    parser.add_argument(
        "--raw-y-blur-radius",
        type=int,
        default=8,
        help="Blur radius for raw Y base/detail split (default: 8)",
    )
    parser.add_argument(
        "--raw-y-full-blend",
        type=float,
        default=0.0,
        help="Full Y blend with raw green (default: 0.0)",
    )
    parser.add_argument(
        "--ltm-detail-threshold",
        type=float,
        default=0.0,
        help="LTM detail noise threshold (default: 0.0)",
    )
    parser.add_argument(
        "--hist-target-mean",
        type=float,
        default=0.0,
        help="Post-gamma histogram target mean (default: 0.0)",
    )
    parser.add_argument(
        "--hist-target-std",
        type=float,
        default=0.0,
        help="Post-gamma histogram target std (default: 0.0)",
    )
    parser.add_argument(
        "--post-denoise-radius",
        type=int,
        default=0,
        help="Post-gamma guided filter denoise radius (default: 0)",
    )
    parser.add_argument(
        "--post-denoise-eps",
        type=float,
        default=0.005,
        help="Post-gamma denoise epsilon (default: 0.005)",
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    isp_params = {
        "awb_max_gain": args.awb_max_gain,
        "awb_lum_mask_low": args.awb_lum_mask_low,
        "awb_lum_mask_high": args.awb_lum_mask_high,
        "denoise_radius": args.denoise_radius,
        "denoise_eps": args.denoise_eps,
        "ltm_a": args.ltm_a,
        "ltm_b": args.ltm_b,
        "ltm_radius": args.ltm_radius,
        "ltm_downsample": args.ltm_downsample,
        "ltm_eps": args.ltm_eps,
        "ltm_target_mean": args.ltm_target_mean,
        "ltm_detail_gain": args.ltm_detail_gain,
        "gamma": args.gamma,
        "sharp_amount": args.sharp_amount,
        "sharp_radius": args.sharp_radius,
        "sharp_threshold": args.sharp_threshold,
        "raw_y_blend": args.raw_y_blend,
        "raw_y_blur_radius": args.raw_y_blur_radius,
        "raw_y_full_blend": args.raw_y_full_blend,
        "ltm_detail_threshold": args.ltm_detail_threshold,
        "hist_target_mean": args.hist_target_mean,
        "hist_target_std": args.hist_target_std,
        "post_denoise_radius": args.post_denoise_radius,
        "post_denoise_eps": args.post_denoise_eps,
    }

    process_video(
        video_path=args.video,
        config_path=args.config,
        output_path=args.output,
        max_frames=args.max_frames,
        verbose=not args.quiet,
        device=args.device,
        **isp_params,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
