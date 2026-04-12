import argparse
from pathlib import Path

import numpy as np


def generate_synthetic_raw(width: int, height: int, frames: int) -> np.ndarray:
    """
    Generate a simple synthetic Bayer RGGB RAW clip for smoke testing.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        frames: Number of frames to generate

    Returns:
        np.ndarray: Array shaped [frames, height, width] with dtype uint16
    """
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("Synthetic Bayer frames require even width and height.")
    if frames <= 0:
        raise ValueError("frames must be greater than zero.")

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = 512.0 + 2048.0 * (0.55 * xx + 0.45 * yy)

    clip = []
    for frame_index in range(frames):
        pulse = 128.0 * np.sin(2.0 * np.pi * (xx + frame_index / max(frames, 1)))
        frame = base + pulse

        bayer = frame.copy()
        bayer[::2, ::2] *= 1.10
        bayer[1::2, 1::2] *= 0.92
        bayer[::2, 1::2] *= 1.00
        bayer[1::2, ::2] *= 1.02

        clip.append(np.clip(bayer, 0, 4095).astype(np.uint16))

    return np.stack(clip, axis=0)


def main() -> int:
    """CLI entry point for synthetic RAW clip generation."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Bayer RAW clip for repository smoke tests."
    )
    parser.add_argument("--output", required=True, help="Output .bin path")
    parser.add_argument("--width", type=int, default=64, help="Frame width (default: 64)")
    parser.add_argument("--height", type=int, default=64, help="Frame height (default: 64)")
    parser.add_argument("--frames", type=int, default=4, help="Number of frames (default: 4)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip = generate_synthetic_raw(width=args.width, height=args.height, frames=args.frames)
    with output_path.open("wb") as file_handle:
        for frame in clip:
            file_handle.write(frame.tobytes())

    print(
        "Saved synthetic RAW clip to "
        f"{output_path} ({args.frames} frames, {args.width}x{args.height})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
