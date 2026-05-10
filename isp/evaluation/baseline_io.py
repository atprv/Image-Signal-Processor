"""
Helpers for loading canonical baseline metrics from artifact files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRIC_KEYS = ("vif", "nrqm", "unique", "l1_y", "l1_uv")


def _to_scene_metrics(scene: str, raw: dict[str, Any]) -> dict[str, float]:
    missing = [key for key in METRIC_KEYS if key not in raw]
    if missing:
        raise KeyError(f"Scene {scene!r} is missing baseline metric(s) {missing}")
    return {key: float(raw[key]) for key in METRIC_KEYS}


def load_baseline_metrics_txt(path: str | Path) -> dict[str, dict[str, float]]:
    """Parse ``artifacts/baselines/baseline_metrics.txt``."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    scenes: dict[str, dict[str, Any]] = {}
    current_scene: str | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Per-scene: "):
            current_scene = line.split(":", 1)[1].strip()
            scenes[current_scene] = {}
            continue
        if current_scene is None or not line.startswith("- "):
            continue

        key, raw_value = line[2:].split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key == "isp_params":
            continue
        scenes[current_scene][key] = float(raw_value)

    if not scenes:
        raise ValueError(f"No per-scene baseline metrics found in {path}")
    return {scene: _to_scene_metrics(scene, raw) for scene, raw in scenes.items()}


def load_pretrain_eval_baseline_json(
    path: str | Path,
) -> dict[str, dict[str, float]]:
    """Load mini-val ISP baseline from ``pretrain_eval_metrics.json``."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    baseline_per_scene = payload.get("baseline_per_scene")
    if not isinstance(baseline_per_scene, dict) or not baseline_per_scene:
        raise KeyError(f"{path} does not contain a non-empty baseline_per_scene mapping")
    return {scene: _to_scene_metrics(scene, raw) for scene, raw in baseline_per_scene.items()}


def discover_pretrain_eval_metrics_json(
    *,
    root: str | Path,
    ckpt_path: str | Path | None = None,
    explicit_path: str | Path | None = None,
) -> Path:
    """Find the pretrain eval JSON that matches the warm-start checkpoint."""
    root = Path(root)
    candidates: list[Path] = []

    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = root / path
        candidates.append(path)
    else:
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            candidates.append(ckpt_path.parent / "pretrain_eval_metrics.json")
        candidates.extend(
            [
                root / "artifacts" / "checkpoints" / "cnn_pretrain" / "pretrain_eval_metrics.json",
                root / "artifacts" / "checkpoints" / "pretrain_eval_metrics.json",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    pretty = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError("Could not find pretrain_eval_metrics.json. Looked in:\n" + pretty)


def load_full_baseline(
    *,
    root: str | Path,
    path: str | Path | None = None,
) -> tuple[dict[str, dict[str, float]], Path]:
    """Load the canonical full-scene baseline from ``baseline_metrics.txt``."""
    root = Path(root)
    baseline_path = (
        Path(path)
        if path is not None
        else (root / "artifacts" / "baselines" / "baseline_metrics.txt")
    )
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path
    baseline_path = baseline_path.resolve()
    return load_baseline_metrics_txt(baseline_path), baseline_path


def load_e2e_minival_baseline(
    *,
    root: str | Path,
    ckpt_path: str | Path | None = None,
    path: str | Path | None = None,
) -> tuple[dict[str, dict[str, float]], Path]:
    """Load the same-split ISP baseline used by E2E validation/guard rails."""
    baseline_path = discover_pretrain_eval_metrics_json(
        root=root,
        ckpt_path=ckpt_path,
        explicit_path=path,
    )
    return load_pretrain_eval_baseline_json(baseline_path), baseline_path
