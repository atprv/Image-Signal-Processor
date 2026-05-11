"""
Optuna sweep over the structural ISP knobs that are not gradient-trainable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isp.config.config_reader import read_config
from isp.evaluation.composite_score import (
    compute_composite,
    compute_composite_terms,
    load_composite_config,
)
from isp.evaluation.evaluation_utils import evaluate, limit_eval_items, load_split_items
from isp.models.residual_cnn import ResidualCNN
from isp.pipeline.pipeline import ISPPipeline

try:
    import optuna
    from optuna.samplers import TPESampler

    _OPTUNA_IMPORT_ERROR = None
except ImportError as exc:
    optuna = None
    TPESampler = None
    _OPTUNA_IMPORT_ERROR = exc


KnobSpec = tuple[str, str, Any, Any, bool]

KNOB_SPECS: tuple[KnobSpec, ...] = (
    ("denoise_radius", "int", 1, 4, False),
    ("ltm_radius", "int", 4, 16, False),
    ("ltm_downsample", "choice", (0.25, 0.5, 0.75, 1.0), None, False),
    ("post_denoise_radius", "int", 0, 6, False),
    ("sharp_radius", "choice", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0), None, False),
    ("raw_y_blur_radius", "int", 1, 16, False),
)

KNOB_NAMES: tuple[str, ...] = tuple(spec[0] for spec in KNOB_SPECS)
KNOB_SPECS_BY_NAME: dict[str, KnobSpec] = {spec[0]: spec for spec in KNOB_SPECS}
STUDY_NAME = "isp_knobs_discrete_v2"

LEGACY_ISP_STATE_ALIASES: dict[str, str] = {
    "denoise.eps": "denoise.log_eps",
    "ltm.eps": "ltm.log_eps",
    "post_denoise.eps": "post_denoise.log_eps",
}

EPS_STATE_FALLBACK_KNOBS: dict[str, str] = {
    "denoise.eps": "denoise_eps",
    "denoise.log_eps": "denoise_eps",
    "ltm.eps": "ltm_eps",
    "ltm.log_eps": "ltm_eps",
    "post_denoise.eps": "post_denoise_eps",
    "post_denoise.log_eps": "post_denoise_eps",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Optuna sweep over non-differentiable ISP knobs with the "
            "residual CNN and trained ISP weights frozen."
        )
    )

    p.add_argument(
        "--ckpt",
        type=str,
        default="artifacts/checkpoints/e2e_quality/e2e_best.pth",
        help="End-to-end checkpoint that provides the frozen residual CNN "
        "and the trained ISP weight tensors loaded into every trial.",
    )
    p.add_argument("--config", type=str, default="data/imx623.toml")
    p.add_argument("--splits-json", type=str, default="dataset/splits_mini.json")
    p.add_argument("--split", type=str, default="val")
    p.add_argument(
        "--norm-weights",
        type=str,
        default="artifacts/baselines/norm_weights.json",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/checkpoints/optuna_isp_knobs",
    )

    p.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Total target number of started trials in the study.",
    )
    p.add_argument(
        "--n-startup-trials",
        type=int,
        default=16,
        help="TPE warmup random trials. Higher is better in "
        "high-dimensional search spaces (default 16).",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--eval-max-frames",
        type=int,
        default=3,
        help="Max frames per scene per trial. Default 3 = mini-val.",
    )
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument(
        "--resume-study",
        dest="resume_study",
        action="store_true",
        default=True,
        help="Continue an existing study DB if present.",
    )
    p.add_argument("--no-resume-study", dest="resume_study", action="store_false")
    p.add_argument(
        "--reset-outputs",
        action="store_true",
        help="Delete the study DB and outputs before starting.",
    )

    p.add_argument(
        "--ranges-json",
        type=str,
        default=None,
        help="Optional JSON file with per-knob range overrides. "
        "Useful for narrowing the search after a coarse pilot.",
    )

    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p).resolve()


def is_google_drive_mount(path: Path) -> bool:
    return str(path).startswith("/content/drive/")


def copy_tree_contents(src_dir: Path, dst_dir: Path) -> None:
    """Recursively copy all files from ``src_dir`` into ``dst_dir``."""
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_path in src_dir.rglob("*"):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
        elif src_path.is_file():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)


def prepare_runtime_output_dir(
    requested_output_dir: Path,
    reset_outputs: bool,
) -> tuple[Path, Path | None]:
    """Use a local staging dir when the requested path is on Google Drive."""
    requested_output_dir.mkdir(parents=True, exist_ok=True)
    if not is_google_drive_mount(requested_output_dir):
        return requested_output_dir, None

    runtime_output_dir = (Path("/tmp") / f"{requested_output_dir.name}_runtime").resolve()
    if runtime_output_dir.exists():
        shutil.rmtree(runtime_output_dir)
    runtime_output_dir.mkdir(parents=True, exist_ok=True)

    if not reset_outputs:
        copy_tree_contents(requested_output_dir, runtime_output_dir)

    return runtime_output_dir, requested_output_dir


def sync_output_dir(
    runtime_output_dir: Path,
    requested_output_dir: Path | None,
    include_reports: bool = False,
) -> None:
    if requested_output_dir is None:
        return

    requested_output_dir.mkdir(parents=True, exist_ok=True)
    names = ["optuna_isp_knobs.db"]
    if include_reports:
        names.extend(
            [
                "optuna_trials.csv",
                "optuna_best_params.json",
                "optuna_history.png",
                "optuna_importance.png",
            ]
        )

    for name in names:
        src = runtime_output_dir / name
        if src.exists():
            dst = requested_output_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def load_range_overrides(path: str | None) -> dict[str, Any]:
    """Load per-knob overrides for narrowing the search."""
    if path is None:
        return {}
    overrides_path = resolve(path)
    if not overrides_path.is_file():
        raise FileNotFoundError(f"Ranges file not found: {overrides_path}")
    with open(overrides_path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in KNOB_NAMES:
            raise KeyError(f"Unknown knob in --ranges-json: {key}")
        _, kind, low, _high, _log_scale = KNOB_SPECS_BY_NAME[key]
        if kind == "choice":
            if not (isinstance(value, (list, tuple)) and len(value) >= 1):
                raise ValueError(
                    f"Choice override for {key} must be a non-empty list of "
                    "allowed discrete values."
                )
            base_choices = list(low)
            if isinstance(base_choices[0], int) and not isinstance(base_choices[0], bool):
                requested = [int(v) for v in value]
            else:
                requested = [float(v) for v in value]
            invalid = [v for v in requested if v not in base_choices]
            if invalid:
                raise ValueError(
                    f"Choice override for {key} contains values outside the base "
                    f"set {base_choices}: {invalid}"
                )
            narrowed = tuple(choice for choice in base_choices if choice in requested)
            if not narrowed:
                raise ValueError(f"Choice override for {key} removed every value.")
            out[key] = narrowed
            continue

        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise ValueError(f"Range for {key} must be [low, high].")
        low, high = float(value[0]), float(value[1])
        if low > high:
            raise ValueError(f"Range for {key} must have low <= high, got {value}.")
        out[key] = (low, high)
    return out


def resolved_knob_specs(
    overrides: dict[str, Any],
) -> list[KnobSpec]:
    """Apply optional range overrides while keeping the original kind/log."""
    specs: list[KnobSpec] = []
    for name, kind, low, high, log_scale in KNOB_SPECS:
        if name in overrides:
            if kind == "choice":
                low = tuple(overrides[name])
            else:
                low, high = overrides[name]
        specs.append((name, kind, low, high, log_scale))
    return specs


def load_cnn(payload: dict[str, Any], device: str) -> ResidualCNN:
    """Reconstruct the residual CNN from the checkpoint's saved state."""
    state = payload.get("cnn_state_dict") or payload.get("model_state_dict") or payload
    cfg = payload.get("config", {})

    cnn = ResidualCNN(
        in_channels=int(cfg.get("in_channels", 3)),
        hidden_channels=int(cfg.get("hidden_ch", 32)),
        out_channels=int(cfg.get("out_channels", 3)),
        num_blocks=int(cfg.get("num_blocks", 5)),
        num_groups=int(cfg.get("num_groups", 8)),
    ).to(device)
    cnn.load_state_dict(state)
    cnn.eval()
    for param in cnn.parameters():
        param.requires_grad_(False)
    return cnn


def trained_param_keys(isp: ISPPipeline) -> frozenset[str]:
    """Return the set of state-dict keys that correspond to ``nn.Parameter``."""
    return frozenset(name for name, _ in isp.named_parameters())


def diff_only_isp_state(
    payload: dict[str, Any], param_keys: frozenset[str]
) -> dict[str, torch.Tensor]:
    """Pick out only the gradient-trained ISP tensors from the checkpoint."""
    full = payload.get("isp_state_dict")
    if not isinstance(full, dict):
        raise KeyError(
            "Checkpoint is missing 'isp_state_dict'. Expected an end-to-end "
            "training checkpoint with both CNN and ISP state."
        )
    accepted_keys = set(param_keys)
    for legacy_key, current_key in LEGACY_ISP_STATE_ALIASES.items():
        if current_key in accepted_keys:
            accepted_keys.add(legacy_key)
    return {k: v for k, v in full.items() if k in accepted_keys}


def _fallback_tensor_for_invalid_key(
    key: str,
    value: torch.Tensor,
    live_state: dict[str, torch.Tensor],
    sampled: dict[str, Any],
) -> torch.Tensor | None:
    """Return the constructor-time fallback tensor for a corrupted state key."""
    live_value = live_state.get(key)
    if live_value is not None:
        return live_value.to(device=value.device, dtype=value.dtype)

    knob_name = EPS_STATE_FALLBACK_KNOBS.get(key)
    if knob_name is None or knob_name not in sampled:
        return None

    fallback_value = float(sampled[knob_name])
    if key.endswith("log_eps"):
        fallback_value = math.log(max(fallback_value, 1e-20))
    return value.new_tensor(fallback_value)


def sanitize_loaded_isp_state(
    isp: ISPPipeline,
    trained_diff_state: dict[str, torch.Tensor],
    sampled: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Replace invalid checkpoint tensors with the trial's constructor values."""
    live_state = isp.state_dict()
    sanitized: dict[str, torch.Tensor] = {}

    for key, value in trained_diff_state.items():
        if not torch.is_tensor(value):
            sanitized[key] = value
            continue

        bad_mask = ~torch.isfinite(value)
        if not bool(bad_mask.any()):
            sanitized[key] = value
            continue

        fallback = _fallback_tensor_for_invalid_key(key, value, live_state, sampled)
        if fallback is None:
            raise RuntimeError(
                f"Checkpoint tensor '{key}' contains NaN/Inf and no fallback "
                "value is available for this trial."
            )
        if fallback.shape != value.shape:
            if fallback.numel() == 1:
                fallback = fallback.expand_as(value)
            else:
                raise RuntimeError(
                    f"Fallback shape mismatch for '{key}': "
                    f"{tuple(fallback.shape)} vs {tuple(value.shape)}"
                )

        repaired = torch.where(bad_mask, fallback, value)
        bad_count = int(bad_mask.sum().item())
        print(
            f"  warning: checkpoint tensor '{key}' has {bad_count} NaN/Inf "
            "value(s); replacing them with the trial constructor value."
        )
        sanitized[key] = repaired

    return sanitized


def sample_knobs(
    trial: optuna.Trial,
    specs: list[KnobSpec],
) -> dict[str, Any]:
    """Sample every knob in ``specs`` from its declared distribution."""
    out: dict[str, Any] = {}
    for name, kind, low, high, log_scale in specs:
        if kind == "int":
            out[name] = trial.suggest_int(name, int(low), int(high))
        elif kind == "float":
            out[name] = trial.suggest_float(name, float(low), float(high), log=log_scale)
        elif kind == "choice":
            out[name] = trial.suggest_categorical(name, list(low))
        else:
            raise ValueError(f"Unknown knob kind {kind!r} for {name}")
    return out


def build_trial_pipeline(
    config: dict[str, Any],
    device: str,
    sampled: dict[str, Any],
    trained_diff_state: dict[str, torch.Tensor],
) -> ISPPipeline:
    """Build a fresh ISPPipeline for one trial."""
    isp = ISPPipeline(config, device=device, **sampled)
    trained_diff_state = sanitize_loaded_isp_state(isp, trained_diff_state, sampled)

    missing, unexpected = isp.load_state_dict(trained_diff_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in trained ISP state: {unexpected}")
    isp.eval()
    for param in isp.parameters():
        param.requires_grad_(False)
    return isp


def run_trial(
    trial: optuna.Trial,
    specs: list[KnobSpec],
    cnn: ResidualCNN,
    config: dict[str, Any],
    config_path: Path,
    splits_json: Path,
    split_name: str,
    composite_cfg: dict[str, Any],
    trained_diff_state: dict[str, torch.Tensor],
    device: str,
    eval_max_frames: int | None,
) -> float:
    """Optuna objective: average composite over scenes for the sampled knobs."""
    sampled = sample_knobs(trial, specs)

    try:
        isp = build_trial_pipeline(config, device, sampled, trained_diff_state)
    except Exception as exc:
        print(f"  trial {trial.number} pipeline build failed: {exc}")
        traceback.print_exc()
        raise optuna.TrialPruned() from exc

    items = load_split_items(str(splits_json), split_name)
    scenes = sorted({item["scene"] for item in items})
    if not scenes:
        raise RuntimeError(f"No items in split '{split_name}' of {splits_json}")

    per_scene: dict[str, dict[str, float]] = {}
    composites: list[float] = []
    vifs: list[float] = []
    nrqms: list[float] = []
    uniques: list[float] = []

    try:
        for scene_name in scenes:
            scene_items = [it for it in items if it["scene"] == scene_name]
            scene_items = limit_eval_items(scene_items, eval_max_frames)
            if not scene_items:
                continue

            try:
                result = evaluate(
                    isp=isp,
                    model=cnn,
                    eval_items=scene_items,
                    config_path=str(config_path),
                    device=device,
                    compute_iqa=True,
                    verbose=False,
                )
            except Exception as exc:
                print(f"  trial {trial.number} scene={scene_name} failed: {exc}")
                traceback.print_exc()
                raise optuna.TrialPruned() from exc

            metrics = {
                "vif": float(result["vif"]),
                "nrqm": float(result["nrqm"]),
                "unique": float(result["unique"]),
                "l1_y": float(result["l1_y"]),
                "l1_uv": float(result["l1_uv"]),
            }
            terms = compute_composite_terms(
                metrics["vif"], metrics["nrqm"], metrics["unique"], composite_cfg
            )
            scene_composite = terms["vif_term"] + terms["a_nrqm_term"] + terms["b_unique_term"]

            metrics["composite"] = scene_composite
            metrics["vif_term"] = terms["vif_term"]
            metrics["a_nrqm_term"] = terms["a_nrqm_term"]
            metrics["b_unique_term"] = terms["b_unique_term"]
            per_scene[scene_name] = metrics

            composites.append(scene_composite)
            vifs.append(metrics["vif"])
            nrqms.append(metrics["nrqm"])
            uniques.append(metrics["unique"])
    finally:
        del isp
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not composites:
        raise optuna.TrialPruned("No scene results produced.")

    avg_composite = sum(composites) / len(composites)
    avg_vif = sum(vifs) / len(vifs)
    avg_nrqm = sum(nrqms) / len(nrqms)
    avg_unique = sum(uniques) / len(uniques)

    trial.set_user_attr("knobs", sampled)
    trial.set_user_attr("per_scene", per_scene)
    trial.set_user_attr("avg_composite", avg_composite)
    trial.set_user_attr("avg_vif", avg_vif)
    trial.set_user_attr("avg_nrqm", avg_nrqm)
    trial.set_user_attr("avg_unique", avg_unique)
    trial.set_user_attr(
        "avg_composite_baseline_metric",
        compute_composite(avg_vif, avg_nrqm, avg_unique, composite_cfg),
    )

    print(
        f"  trial {trial.number:3d}: composite={avg_composite:.4f}  "
        f"avg_vif={avg_vif:.4f}  nrqm={avg_nrqm:.4f}  unique={avg_unique:.4f}"
    )
    return avg_composite


def write_csv(study: optuna.Study, csv_path: Path, scenes: list[str]) -> None:
    """Dump trials with sampled knobs and per-scene metrics to CSV."""
    fieldnames: list[str] = [
        "trial",
        "state",
        "objective",
        "avg_composite",
        "avg_vif",
        "avg_nrqm",
        "avg_unique",
        *KNOB_NAMES,
    ]
    for scene in scenes:
        for metric in ("vif", "nrqm", "unique", "composite"):
            fieldnames.append(f"{scene}_{metric}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in study.trials:
            row: dict[str, Any] = {"trial": t.number, "state": str(t.state).lower()}
            row["objective"] = float(t.value) if t.value is not None else ""
            attrs = t.user_attrs or {}
            row["avg_composite"] = attrs.get("avg_composite", "")
            row["avg_vif"] = attrs.get("avg_vif", "")
            row["avg_nrqm"] = attrs.get("avg_nrqm", "")
            row["avg_unique"] = attrs.get("avg_unique", "")
            knobs = attrs.get("knobs", {}) or t.params or {}
            for key in KNOB_NAMES:
                row[key] = knobs.get(key, "")
            per_scene = attrs.get("per_scene", {}) or {}
            for scene in scenes:
                metrics = per_scene.get(scene, {})
                for metric in ("vif", "nrqm", "unique", "composite"):
                    row[f"{scene}_{metric}"] = metrics.get(metric, "")
            writer.writerow(row)


def write_best_json(
    study: optuna.Study,
    best_path: Path,
    args: argparse.Namespace,
    specs: list[KnobSpec],
    trained_param_names: frozenset[str],
) -> None:
    """Save the best trial summary and a copy-pasteable construction kwargs."""
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        return
    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "best_trial": int(best.number),
        "best_value": float(best.value) if best.value is not None else None,
        "best_params": dict(best.params),
        "best_user_attrs": dict(best.user_attrs or {}),
        "search_ranges": {
            name: (
                {"kind": kind, "choices": list(low)}
                if kind == "choice"
                else {"kind": kind, "low": low, "high": high, "log": log_scale}
            )
            for name, kind, low, high, log_scale in specs
        },
        "objective": (
            "mean_over_scenes(VIF_norm + a*NRQM_norm + b*UNIQUE_norm) with "
            "the residual CNN frozen, the structural knobs sampled (no "
            "per-scene baseline), and the trained ISP weight tensors "
            "loaded from --ckpt."
        ),
        "ckpt": str(resolve(args.ckpt)),
        "construction_kwargs": dict(best.params),
        "diff_trained_state_keys": sorted(trained_param_names),
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_plots(study: optuna.Study, output_dir: Path) -> None:
    """Render Optuna's optimization history + parameter importance figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )
    except ImportError as exc:
        print(f"  skip plots ({exc})")
        return

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        print("  skip plots (no completed trials)")
        return

    try:
        ax = plot_optimization_history(study)
        ax.figure.savefig(output_dir / "optuna_history.png", dpi=120, bbox_inches="tight")
        print(f"  wrote {output_dir / 'optuna_history.png'}")
    except Exception as exc:
        print(f"  history plot failed: {exc}")

    try:
        ax = plot_param_importances(study)
        ax.figure.savefig(output_dir / "optuna_importance.png", dpi=120, bbox_inches="tight")
        print(f"  wrote {output_dir / 'optuna_importance.png'}")
    except Exception as exc:
        print(f"  importance plot failed: {exc}")


def main() -> int:
    if optuna is None:
        raise RuntimeError(
            "Optuna is not installed in the current environment. "
            "Install it with `pip install optuna` and re-run."
        ) from _OPTUNA_IMPORT_ERROR

    args = parse_args()

    requested_output_dir = resolve(args.output_dir)
    output_dir, sync_target_dir = prepare_runtime_output_dir(
        requested_output_dir, reset_outputs=args.reset_outputs
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "optuna_isp_knobs.db"
    csv_path = output_dir / "optuna_trials.csv"
    best_path = output_dir / "optuna_best_params.json"

    if args.reset_outputs:
        for p in (
            db_path,
            csv_path,
            best_path,
            output_dir / "optuna_history.png",
            output_dir / "optuna_importance.png",
        ):
            if p.exists():
                p.unlink()
                print(f"  removed {p.name}")
        if sync_target_dir is not None:
            for p in (
                sync_target_dir / "optuna_isp_knobs.db",
                sync_target_dir / "optuna_trials.csv",
                sync_target_dir / "optuna_best_params.json",
                sync_target_dir / "optuna_history.png",
                sync_target_dir / "optuna_importance.png",
            ):
                if p.exists():
                    p.unlink()
                    print(f"  removed Drive copy {p.name}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    args.device = device

    config_path = resolve(args.config)
    splits_json = resolve(args.splits_json)
    norm_weights = resolve(args.norm_weights)
    ckpt_path = resolve(args.ckpt)

    overrides = load_range_overrides(args.ranges_json)
    specs = resolved_knob_specs(overrides)

    print("=" * 72)
    print("Optuna sweep over non-differentiable ISP knobs")
    print("=" * 72)
    print(f"Output dir:    {requested_output_dir}")
    if sync_target_dir is not None:
        print(f"Runtime dir:   {output_dir}")
        print("Storage mode:  local SQLite staging + Drive sync")
    print(f"Study DB:      {db_path}")
    print(f"Study name:    {STUDY_NAME}")
    print(f"CNN ckpt:      {ckpt_path}")
    print(f"Splits:        {splits_json}  (split={args.split})")
    print(f"Norm weights:  {norm_weights}")
    print(f"Eval frames:   {args.eval_max_frames} per scene")
    print(f"Trials target: {args.n_trials}  (TPE startup={args.n_startup_trials})")
    print()
    print(f"Sampling {len(specs)} non-differentiable knobs:")
    for name, kind, low, high, log_scale in specs:
        if kind == "int":
            rng_str = f"[{int(low)}, {int(high)}] int"
        elif kind == "choice":
            vals = ", ".join(f"{float(v):g}" for v in low)
            rng_str = f"{{{vals}}} choice"
        elif log_scale:
            rng_str = f"[{low:.1e}, {high:.1e}] log"
        else:
            rng_str = f"[{low:g}, {high:g}]"
        marker = " *" if name in overrides else "  "
        print(f"  {marker} {name:<24s} {rng_str}")
    print(f"\nLoading trained ISP weights from {ckpt_path}")
    print(
        "(structural buffers in the checkpoint are dropped; their shapes "
        "depend on the sampled knobs)\n"
    )

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"CNN checkpoint missing: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cnn = load_cnn(payload, device)
    config = read_config(str(config_path), device=device)
    composite_cfg = load_composite_config(norm_weights)

    probe_isp = ISPPipeline(config, device=device)
    trained_param_names = trained_param_keys(probe_isp)
    del probe_isp
    trained_diff_state = diff_only_isp_state(payload, trained_param_names)

    print(f"CNN: {sum(p.numel() for p in cnn.parameters()):,} params, frozen")
    print("Trained ISP tensors loaded:")
    for k in sorted(trained_diff_state.keys()):
        v = trained_diff_state[k]
        print(f"  {k:<35s} shape={tuple(v.shape)}  first={v.flatten()[:3].tolist()}")
    print()

    items = load_split_items(str(splits_json), args.split)
    scenes = sorted({item["scene"] for item in items})
    print(f"Scenes in split '{args.split}': {scenes}\n")

    storage = f"sqlite:///{db_path.as_posix()}"
    sampler = TPESampler(
        seed=args.seed,
        n_startup_trials=max(1, int(args.n_startup_trials)),
    )

    if not args.resume_study and db_path.exists():
        db_path.unlink()
        print(f"  removed existing {db_path.name} (resume_study=False)")
    if args.resume_study:
        print(f"Loading or creating study '{STUDY_NAME}' in {db_path}")
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=bool(args.resume_study),
    )

    existing = len(study.trials)
    remaining = max(0, int(args.n_trials) - existing)
    print(f"Existing trials in study: {existing}")
    print(f"Trials to add this session: {remaining}\n")

    if remaining == 0:
        print("Trial budget already met; running reporting only.")
    else:
        t0 = time.time()

        def _sync_db_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            try:
                sync_output_dir(output_dir, sync_target_dir, include_reports=False)
            except Exception as exc:
                print(f"  warning: failed to sync study DB after trial {trial.number}: {exc}")

        try:
            study.optimize(
                lambda t: run_trial(
                    t,
                    specs,
                    cnn,
                    config,
                    config_path,
                    splits_json,
                    args.split,
                    composite_cfg,
                    trained_diff_state,
                    device,
                    args.eval_max_frames,
                ),
                n_trials=remaining,
                gc_after_trial=True,
                callbacks=[_sync_db_callback],
            )
        finally:
            try:
                sync_output_dir(output_dir, sync_target_dir, include_reports=False)
            except Exception as exc:
                print(f"  warning: final DB sync after optimize() failed: {exc}")
        print(f"\nFinished {remaining} trials in {time.time() - t0:.1f}s.")

    print("\nWriting outputs...")
    write_csv(study, csv_path, scenes)
    print(f"  wrote {csv_path}")
    write_best_json(study, best_path, args, specs, trained_param_names)
    print(f"  wrote {best_path}")
    save_plots(study, output_dir)
    try:
        sync_output_dir(output_dir, sync_target_dir, include_reports=True)
        if sync_target_dir is not None:
            print(f"  synced reports to {sync_target_dir}")
    except Exception as exc:
        print(f"  warning: final Drive sync failed: {exc}")

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if completed:
        best = study.best_trial
        attrs = best.user_attrs or {}
        print()
        print("Best trial summary:")
        print(f"  trial #:        {best.number}")
        print(f"  composite:      {best.value:.6f}")
        print(f"  avg VIF:        {attrs.get('avg_vif', float('nan')):.4f}")
        print(f"  avg NRQM:       {attrs.get('avg_nrqm', float('nan')):.4f}")
        print(f"  avg UNIQUE:     {attrs.get('avg_unique', float('nan')):.4f}")
        print("  knobs:")
        for name in KNOB_NAMES:
            print(f"    {name:<24s} = {best.params.get(name)}")
    else:
        print("\nNo completed trials yet.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
