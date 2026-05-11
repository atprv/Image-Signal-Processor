"""
Optuna search for E2E ISP quality-training hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isp.evaluation.baseline_io import load_e2e_minival_baseline

try:
    import optuna
except ModuleNotFoundError as exc:
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune E2E quality-training hyperparameters with Optuna."
    )
    p.add_argument("--train-h5", default="dataset/train_patches.h5")
    p.add_argument("--splits-json", default="dataset/splits_mini.json")
    p.add_argument("--config", default="data/imx623.toml")
    p.add_argument("--ckpt", default="artifacts/checkpoints/cnn_pretrained.pth")
    p.add_argument(
        "--pretrain-eval-json",
        default=None,
        help="Optional path to pretrain_eval_metrics.json. When "
        "omitted, the script looks next to --ckpt and then in "
        "artifacts/checkpoints/cnn_pretrain/.",
    )
    p.add_argument("--norm-weights", default="artifacts/baselines/norm_weights.json")
    p.add_argument("--output-dir", default="artifacts/checkpoints/e2e_optuna")

    p.add_argument("--n-trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-max-frames", type=int, default=3)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds.")

    p.add_argument("--study-name", default="e2e_quality_optuna")
    p.add_argument(
        "--storage", default=None, help="Optuna storage URL. Defaults to sqlite in output-dir."
    )
    p.add_argument(
        "--resume-study",
        action="store_true",
        help="Resume an existing Optuna study instead of failing.",
    )
    p.add_argument(
        "--resume-incomplete-trials",
        dest="resume_incomplete_trials",
        action="store_true",
        default=True,
        help="Before starting new trials, finish interrupted trials from their trial_resume.pth.",
    )
    p.add_argument(
        "--no-resume-incomplete-trials",
        dest="resume_incomplete_trials",
        action="store_false",
        help="Skip interrupted-trial repair and start new trials.",
    )
    p.add_argument("--n-startup-trials", type=int, default=4)

    p.add_argument(
        "--scene-aware-train", dest="scene_aware_train", action="store_true", default=True
    )
    p.add_argument("--no-scene-aware-train", dest="scene_aware_train", action="store_false")
    p.add_argument("--balance-scenes", dest="balance_scenes", action="store_true", default=True)
    p.add_argument("--no-balance-scenes", dest="balance_scenes", action="store_false")

    p.add_argument(
        "--day-vif-target",
        type=float,
        default=0.0,
        help="Target day-scene VIF used by the Optuna objective. "
        "Default 0.0 disables the day-VIF deficit penalty.",
    )
    p.add_argument(
        "--night-vif-floor-ratio",
        type=float,
        default=0.90,
        help="Night VIF floor as a fraction of mini-val baseline.",
    )
    p.add_argument(
        "--day-vif-weight",
        type=float,
        default=0.0,
        help="Positive reward for day VIF in the Optuna objective. "
        "Default 0.0 makes the objective composite-driven only.",
    )
    p.add_argument(
        "--day-vif-penalty",
        type=float,
        default=0.0,
        help="Linear penalty for day VIF below --day-vif-target. "
        "Default 0.0 disables the day-VIF pursuit term.",
    )
    p.add_argument(
        "--night-vif-penalty",
        type=float,
        default=10.0,
        help="Linear penalty for night VIF below its floor.",
    )
    p.add_argument(
        "--unique-floor",
        type=float,
        default=0.0,
        help="Per-scene UNIQUE floor used by the Optuna objective.",
    )
    p.add_argument(
        "--unique-penalty",
        type=float,
        default=12.0,
        help="Linear penalty for day/night UNIQUE below floor.",
    )
    p.add_argument(
        "--force-isp-reg",
        dest="force_isp_reg",
        action="store_true",
        default=True,
        help="Always sample a positive ISP regularization weight.",
    )
    p.add_argument(
        "--allow-zero-isp-reg",
        dest="force_isp_reg",
        action="store_false",
        help="Allow trials with isp_reg_weight=0.0.",
    )

    p.add_argument("--lr-isp-min", type=float, default=1e-6)
    p.add_argument("--lr-isp-max", type=float, default=3e-5)
    p.add_argument("--lr-cnn-min", type=float, default=3e-6)
    p.add_argument("--lr-cnn-max", type=float, default=1e-4)
    p.add_argument("--w-ssim-min", type=float, default=0.2)
    p.add_argument("--w-ssim-max", type=float, default=1.2)
    p.add_argument("--w-vif-min", type=float, default=2.0)
    p.add_argument("--w-vif-max", type=float, default=12.0)
    p.add_argument("--w-unique-min", type=float, default=0.10)
    p.add_argument("--w-unique-max", type=float, default=1.00)
    p.add_argument("--w-l1-y-min", type=float, default=0.0)
    p.add_argument("--w-l1-y-max", type=float, default=0.50)
    p.add_argument("--w-uv-min", type=float, default=0.5)
    p.add_argument("--w-uv-max", type=float, default=2.0)
    p.add_argument("--isp-reg-min", type=float, default=5e-3)
    p.add_argument("--isp-reg-max", type=float, default=1e-1)
    p.add_argument("--day-loss-weight-min", type=float, default=1.0)
    p.add_argument("--day-loss-weight-max", type=float, default=6.0)
    p.add_argument("--night-loss-weight-min", type=float, default=1.0)
    p.add_argument("--night-loss-weight-max", type=float, default=3.0)
    p.add_argument("--vif-guard-ratio-min", type=float, default=0.85)
    p.add_argument("--vif-guard-ratio-max", type=float, default=1.0)
    p.add_argument("--vif-guard-penalty-min", type=float, default=5.0)
    p.add_argument("--vif-guard-penalty-max", type=float, default=30.0)

    p.add_argument(
        "--stream-logs", action="store_true", help="Mirror trial subprocess logs to stdout."
    )
    p.add_argument("--keep-failed-trials", action="store_true")
    return p.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (ROOT / path).resolve()


def check_inputs(args: argparse.Namespace) -> None:
    required = {
        "train_h5": resolve(args.train_h5),
        "splits_json": resolve(args.splits_json),
        "config": resolve(args.config),
        "ckpt": resolve(args.ckpt),
        "norm_weights": resolve(args.norm_weights),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def sample_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    use_isp_reg = (
        True if args.force_isp_reg else trial.suggest_categorical("use_isp_reg", [True, False])
    )
    return {
        "use_isp_reg": bool(use_isp_reg),
        "lr_isp": trial.suggest_float(
            "lr_isp",
            args.lr_isp_min,
            args.lr_isp_max,
            log=True,
        ),
        "lr_cnn": trial.suggest_float(
            "lr_cnn",
            args.lr_cnn_min,
            args.lr_cnn_max,
            log=True,
        ),
        "w_ssim": trial.suggest_float(
            "w_ssim",
            args.w_ssim_min,
            args.w_ssim_max,
        ),
        "w_vif": trial.suggest_float(
            "w_vif",
            args.w_vif_min,
            args.w_vif_max,
        ),
        "w_unique": trial.suggest_float(
            "w_unique",
            args.w_unique_min,
            args.w_unique_max,
        ),
        "w_l1_y": trial.suggest_float(
            "w_l1_y",
            args.w_l1_y_min,
            args.w_l1_y_max,
        ),
        "w_uv": trial.suggest_float(
            "w_uv",
            args.w_uv_min,
            args.w_uv_max,
        ),
        "isp_reg_weight": (
            trial.suggest_float(
                "isp_reg_weight",
                args.isp_reg_min,
                args.isp_reg_max,
                log=True,
            )
            if use_isp_reg
            else 0.0
        ),
        "day_loss_weight": trial.suggest_float(
            "day_loss_weight",
            args.day_loss_weight_min,
            args.day_loss_weight_max,
        ),
        "night_loss_weight": trial.suggest_float(
            "night_loss_weight",
            args.night_loss_weight_min,
            args.night_loss_weight_max,
        ),
        "vif_guard_ratio": trial.suggest_float(
            "vif_guard_ratio",
            args.vif_guard_ratio_min,
            args.vif_guard_ratio_max,
        ),
        "vif_guard_penalty": trial.suggest_float(
            "vif_guard_penalty",
            args.vif_guard_penalty_min,
            args.vif_guard_penalty_max,
        ),
    }


def build_train_command(
    args: argparse.Namespace, params: dict[str, Any], trial_dir: Path, resume: bool = False
) -> list[str]:
    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else args.epochs
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "run_e2e_train.py"),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr-isp",
        str(params["lr_isp"]),
        "--lr-cnn",
        str(params["lr_cnn"]),
        "--lambda-uv",
        "1.0",
        "--loss",
        "quality",
        "--best-criterion",
        "composite",
        "--w-ssim",
        str(params["w_ssim"]),
        "--w-vif",
        str(params["w_vif"]),
        "--w-unique",
        str(params["w_unique"]),
        "--w-l1-y",
        str(params["w_l1_y"]),
        "--w-uv",
        str(params["w_uv"]),
        "--isp-reg-weight",
        str(params["isp_reg_weight"]),
        "--vif-guard-ratio",
        str(params["vif_guard_ratio"]),
        "--vif-guard-penalty",
        str(params["vif_guard_penalty"]),
        "--eval-every",
        str(args.eval_every),
        "--eval-max-frames",
        str(args.eval_max_frames),
        "--checkpoint-every",
        str(checkpoint_every),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--day-loss-weight",
        str(params["day_loss_weight"]),
        "--night-loss-weight",
        str(params["night_loss_weight"]),
        "--train-h5",
        str(resolve(args.train_h5)),
        "--splits-json",
        str(resolve(args.splits_json)),
        "--config",
        str(resolve(args.config)),
        "--ckpt",
        str(resolve(args.ckpt)),
        "--norm-weights",
        str(resolve(args.norm_weights)),
        "--output-dir",
        str(trial_dir),
        "--history-name",
        "e2e_history.json",
        "--ckpt-prefix",
        "trial",
    ]
    if args.pretrain_eval_json:
        cmd.extend(
            [
                "--pretrain-eval-json",
                str(resolve(args.pretrain_eval_json)),
            ]
        )
    if args.scene_aware_train:
        cmd.append("--scene-aware-train")
    if args.balance_scenes:
        cmd.append("--balance-scenes")
    if args.max_train_batches is not None:
        cmd.extend(["--max-train-batches", str(args.max_train_batches)])
    if resume:
        cmd.append("--resume")
    return cmd


def run_subprocess(cmd: list[str], log_path: Path, stream_logs: bool = False) -> int:
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            if stream_logs:
                print(line, end="")
        return proc.wait()


def load_history(history_path: Path) -> list[dict[str, Any]]:
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list):
        raise ValueError(f"Expected list history in {history_path}")
    return history


def _scene_metric(row: dict[str, Any], scene: str, metric: str) -> float | None:
    scene_data = (row.get("val_per_scene") or {}).get(scene)
    if not scene_data or metric not in scene_data:
        return None
    value = scene_data[metric]
    return float(value) if value is not None else None


def score_record(
    row: dict[str, Any], args: argparse.Namespace, baseline: dict[str, dict[str, float]]
) -> dict[str, float]:
    val_composite = row.get("val_guarded_composite")
    if val_composite is None:
        val_composite = row.get("val_composite")
    if val_composite is None:
        return {"objective": float("-inf")}

    day_vif = _scene_metric(row, "day", "vif")
    night_vif = _scene_metric(row, "night", "vif")
    day_unique = _scene_metric(row, "day", "unique")
    night_unique = _scene_metric(row, "night", "unique")
    if day_vif is None:
        day_vif = float(row.get("val_vif", 0.0))
    if night_vif is None:
        night_vif = float(row.get("val_vif", 0.0))
    if day_unique is None:
        day_unique = float(row.get("val_unique", 0.0))
    if night_unique is None:
        night_unique = float(row.get("val_unique", 0.0))

    night_floor = args.night_vif_floor_ratio * float(baseline["night"]["vif"])
    day_deficit = max(0.0, args.day_vif_target - day_vif)
    night_deficit = max(0.0, night_floor - night_vif)
    day_unique_deficit = max(0.0, args.unique_floor - day_unique)
    night_unique_deficit = max(0.0, args.unique_floor - night_unique)

    objective = (
        float(val_composite)
        + args.day_vif_weight * day_vif
        - args.day_vif_penalty * day_deficit
        - args.night_vif_penalty * night_deficit
        - args.unique_penalty * (day_unique_deficit + night_unique_deficit)
    )
    return {
        "objective": float(objective),
        "base_score": float(val_composite),
        "day_vif": float(day_vif),
        "night_vif": float(night_vif),
        "day_unique": float(day_unique),
        "night_unique": float(night_unique),
        "night_floor": float(night_floor),
        "day_vif_deficit": float(day_deficit),
        "night_vif_deficit": float(night_deficit),
        "day_unique_deficit": float(day_unique_deficit),
        "night_unique_deficit": float(night_unique_deficit),
    }


def summarize_history(
    history: list[dict[str, Any]], args: argparse.Namespace, baseline: dict[str, dict[str, float]]
) -> dict[str, Any]:
    val_rows = [row for row in history if row.get("val_composite") is not None]
    if not val_rows:
        raise ValueError("No validation rows with val_composite were found.")

    scored = []
    for row in val_rows:
        metrics = score_record(row, args, baseline)
        if math.isfinite(metrics["objective"]):
            scored.append((metrics["objective"], row, metrics))
    if not scored:
        raise ValueError("All validation rows produced non-finite objective.")

    _, best_row, best_score = max(scored, key=lambda item: item[0])
    per_scene = best_row.get("val_per_scene") or {}
    return {
        "best_epoch": int(best_row["epoch"]),
        "objective": best_score["objective"],
        "base_score": best_score["base_score"],
        "day_vif": best_score["day_vif"],
        "night_vif": best_score["night_vif"],
        "day_unique": best_score["day_unique"],
        "night_unique": best_score["night_unique"],
        "night_floor": best_score["night_floor"],
        "day_vif_deficit": best_score["day_vif_deficit"],
        "night_vif_deficit": best_score["night_vif_deficit"],
        "day_unique_deficit": best_score["day_unique_deficit"],
        "night_unique_deficit": best_score["night_unique_deficit"],
        "val_composite": best_row.get("val_composite"),
        "val_guarded_composite": best_row.get("val_guarded_composite"),
        "val_vif": best_row.get("val_vif"),
        "val_nrqm": best_row.get("val_nrqm"),
        "val_unique": best_row.get("val_unique"),
        "val_l1y": best_row.get("val_l1y"),
        "val_l1uv": best_row.get("val_l1uv"),
        "day_nrqm": (per_scene.get("day") or {}).get("nrqm"),
        "day_unique_raw": (per_scene.get("day") or {}).get("unique"),
        "night_nrqm": (per_scene.get("night") or {}).get("nrqm"),
        "night_unique_raw": (per_scene.get("night") or {}).get("unique"),
    }


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    return str(obj)


def append_trial_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "trial",
        "state",
        "optuna_state",
        "objective",
        "base_score",
        "best_epoch",
        "day_vif",
        "night_vif",
        "day_unique",
        "night_unique",
        "day_vif_deficit",
        "night_vif_deficit",
        "day_unique_deficit",
        "night_unique_deficit",
        "val_composite",
        "val_guarded_composite",
        "val_vif",
        "val_nrqm",
        "val_unique",
        "val_l1y",
        "val_l1uv",
        "lr_isp",
        "lr_cnn",
        "w_ssim",
        "w_vif",
        "w_unique",
        "w_l1_y",
        "w_uv",
        "use_isp_reg",
        "isp_reg_weight",
        "day_loss_weight",
        "night_loss_weight",
        "vif_guard_ratio",
        "vif_guard_penalty",
        "trial_dir",
        "elapsed_s",
    ]
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def upsert_trial_csv(csv_path: Path, row: dict[str, Any]) -> None:
    """Write one trial row, replacing older rows for the same trial number."""
    if not csv_path.exists():
        append_trial_csv(csv_path, row)
        return

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            existing for existing in reader if str(existing.get("trial")) != str(row.get("trial"))
        ]
    csv_path.unlink()
    for existing in rows:
        append_trial_csv(csv_path, existing)
    append_trial_csv(csv_path, row)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(payload), f, indent=2, ensure_ascii=False)


def trial_params_from_frozen(trial: Any) -> dict[str, Any]:
    """Reconstruct train-script params from an existing Optuna FrozenTrial."""
    params = dict(getattr(trial, "params", {}) or {})
    required = [
        "lr_isp",
        "lr_cnn",
        "w_ssim",
        "w_vif",
        "w_unique",
        "w_l1_y",
        "w_uv",
        "day_loss_weight",
        "night_loss_weight",
        "vif_guard_ratio",
        "vif_guard_penalty",
    ]
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"Trial {trial.number} is missing sampled params: {missing}")
    if "use_isp_reg" in params:
        use_isp_reg = bool(params["use_isp_reg"])
    else:
        use_isp_reg = float(params.get("isp_reg_weight", 0.0)) > 0.0
    params["use_isp_reg"] = use_isp_reg
    if not use_isp_reg:
        params["isp_reg_weight"] = 0.0
    elif "isp_reg_weight" not in params:
        raise ValueError(f"Trial {trial.number} has use_isp_reg=True but no isp_reg_weight.")
    return params


def max_train_epoch(history: list[dict[str, Any]]) -> int:
    """Return the highest completed training epoch in an E2E history list."""
    epochs = [
        int(row["epoch"]) for row in history if int(row.get("epoch", 0)) > 0 and "train_loss" in row
    ]
    return max(epochs) if epochs else 0


def has_complete_summary(trial_dir: Path) -> bool:
    summary_path = trial_dir / "trial_summary.json"
    if not summary_path.exists():
        return False
    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return False
    return summary.get("state") == "complete"


def set_trial_user_attrs(study: Any, trial: Any, attrs: dict[str, Any]) -> None:
    """Best-effort user-attr update for repaired FrozenTrials."""
    storage = getattr(study, "_storage", None)
    trial_id = getattr(trial, "_trial_id", None)
    if storage is None or trial_id is None:
        return
    for key, value in attrs.items():
        try:
            storage.set_trial_user_attr(
                trial_id,
                key,
                make_serializable(value),
            )
        except Exception:
            pass


def tell_trial_complete(study: Any, trial: Any, value: float) -> bool:
    """Mark an interrupted RUNNING trial as COMPLETE after repair."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        return True
    if trial.state != optuna.trial.TrialState.RUNNING:
        return False

    state = optuna.trial.TrialState.COMPLETE
    try:
        study.tell(trial.number, value, state=state, skip_if_finished=True)
    except TypeError:
        study.tell(trial.number, value, state=state)
    return True


def write_current_best(
    study: Any, best_path: Path, csv_path: Path, output_dir: Path, args: argparse.Namespace
) -> None:
    """Persist the current Optuna best trial, if one exists."""
    complete_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        return
    best = study.best_trial
    save_json(
        best_path,
        {
            "study_name": study.study_name,
            "best_trial": best.number,
            "best_value": best.value,
            "best_params": best.params,
            "best_user_attrs": best.user_attrs,
            "output_dir": str(output_dir),
            "trials_csv": str(csv_path),
            "objective": {
                "formula": (
                    "score = guarded_or_raw_composite + day_vif_weight*day_vif "
                    "- day_vif_penalty*max(0, day_vif_target-day_vif) "
                    "- night_vif_penalty*max(0, night_floor-night_vif) "
                    "- unique_penalty*(max(0, unique_floor-day_unique) + "
                    "max(0, unique_floor-night_unique))"
                ),
                "day_vif_target": args.day_vif_target,
                "night_vif_floor_ratio": args.night_vif_floor_ratio,
                "unique_floor": args.unique_floor,
                "day_vif_weight": args.day_vif_weight,
                "day_vif_penalty": args.day_vif_penalty,
                "night_vif_penalty": args.night_vif_penalty,
                "unique_penalty": args.unique_penalty,
            },
        },
    )


def finish_incomplete_trials(
    study: Any,
    args: argparse.Namespace,
    baseline: dict[str, dict[str, float]],
    output_dir: Path,
    csv_path: Path,
    best_path: Path,
) -> int:
    """Resume interrupted Optuna trials before asking for new ones."""
    if not args.resume_incomplete_trials:
        return 0

    repaired = 0
    repairable_states = {
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    for trial in sorted(study.trials, key=lambda t: t.number):
        if trial.state not in repairable_states:
            continue

        trial_dir = output_dir / f"trial_{trial.number:03d}"
        if not trial_dir.exists():
            continue

        history_path = trial_dir / "e2e_history.json"
        if not history_path.exists():
            print(
                f"[repair] trial_{trial.number:03d} has no e2e_history.json; "
                "cannot resume automatically."
            )
            continue

        params = trial_params_from_frozen(trial)
        history = load_history(history_path)
        completed_epoch = max_train_epoch(history)
        command_text = ""
        elapsed = 0.0

        if completed_epoch >= args.epochs:
            print(
                f"[repair] trial_{trial.number:03d} already reached "
                f"epoch {completed_epoch}/{args.epochs}; scoring existing "
                "history."
            )
        else:
            print(
                f"[repair] Resuming trial_{trial.number:03d}: "
                f"state={trial.state.name}, completed_epoch={completed_epoch}, "
                f"target_epochs={args.epochs}"
            )
            cmd = build_train_command(args, params, trial_dir, resume=True)
            command_text = " ".join(cmd)
            log_path = trial_dir / f"resume_{int(time.time())}.log"
            start = time.time()
            ret = run_subprocess(cmd, log_path, stream_logs=args.stream_logs)
            elapsed = time.time() - start
            if ret != 0:
                raise RuntimeError(
                    f"Could not resume trial_{trial.number:03d}: subprocess "
                    f"exit={ret}. See {log_path}"
                )

            history = load_history(history_path)
            completed_epoch = max_train_epoch(history)
            if completed_epoch < args.epochs:
                raise RuntimeError(
                    f"trial_{trial.number:03d} resumed but only reached epoch "
                    f"{completed_epoch}/{args.epochs}. See {log_path}"
                )

        summary = summarize_history(history, args, baseline)
        row = {
            **params,
            **summary,
            "trial": trial.number,
            "state": "complete",
            "optuna_state": trial.state.name,
            "trial_dir": str(trial_dir),
            "elapsed_s": elapsed,
        }
        save_json(trial_dir / "trial_summary.json", row)
        upsert_trial_csv(csv_path, row)
        if trial.state == optuna.trial.TrialState.RUNNING:
            set_trial_user_attrs(
                study,
                trial,
                {
                    **summary,
                    "trial_dir": str(trial_dir),
                    "command": command_text,
                    "elapsed_s": elapsed,
                },
            )
            if tell_trial_complete(study, trial, float(summary["objective"])):
                state_msg = "marked COMPLETE in Optuna"
            else:
                state_msg = f"left Optuna state as {trial.state.name}"
        else:
            state_msg = f"recorded result; Optuna state remains {trial.state.name}"
        repaired += 1
        print(
            f"[repair] trial_{trial.number:03d} complete: "
            f"objective={summary['objective']:.6f}, "
            f"day_vif={summary['day_vif']:.4f}, "
            f"night_vif={summary['night_vif']:.4f}, "
            f"{state_msg}"
        )

    if repaired:
        write_current_best(study, best_path, csv_path, output_dir, args)
    return repaired


def create_study(args: argparse.Namespace, output_dir: Path) -> optuna.Study:
    storage = args.storage
    if storage is None:
        storage = f"sqlite:///{(output_dir / 'optuna_study.db').resolve().as_posix()}"

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=0,
    )
    return optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume_study,
    )


def main() -> None:
    args = parse_args()
    if optuna is None:
        raise SystemExit(
            "Optuna is required for this script. Install it with:\n  pip install optuna"
        ) from _OPTUNA_IMPORT_ERROR

    check_inputs(args)
    try:
        baseline, baseline_path = load_e2e_minival_baseline(
            root=ROOT,
            ckpt_path=resolve(args.ckpt),
            path=args.pretrain_eval_json,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Optuna E2E tuning needs pretrain_eval_metrics.json so the "
            "night-VIF floor matches the mini-val ISP baseline."
        ) from exc

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "optuna_trials.csv"
    best_path = output_dir / "optuna_best_params.json"

    print("=" * 72)
    print("Optuna E2E quality tuning")
    print("=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Trials: {args.n_trials}  Epochs/trial: {args.epochs}")
    print(f"Mini-val baseline source: {baseline_path}")
    if args.day_vif_penalty > 0.0 or args.day_vif_weight > 0.0:
        print(
            f"Day-VIF shaping: target={args.day_vif_target:.3f}, "
            f"weight={args.day_vif_weight:.3f}, "
            f"penalty={args.day_vif_penalty:.3f}"
        )
    else:
        print("Day-VIF shaping: disabled (composite-driven objective)")
    print(f"Night VIF floor: {args.night_vif_floor_ratio:.2f} * {baseline['night']['vif']:.6f}")
    print(
        f"UNIQUE floor: day/night unique >= {args.unique_floor:.3f} (penalty={args.unique_penalty})"
    )
    print(f"Force ISP regularization: {args.force_isp_reg}")
    print(f"Scene-aware train: {args.scene_aware_train}")
    print(f"Balanced scenes:   {args.balance_scenes}")

    study = create_study(args, output_dir)
    repaired = finish_incomplete_trials(
        study=study,
        args=args,
        baseline=baseline,
        output_dir=output_dir,
        csv_path=csv_path,
        best_path=best_path,
    )
    if repaired:
        print(f"[repair] Finished {repaired} interrupted trial(s) before starting new trials.")

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, args)
        trial_dir = output_dir / f"trial_{trial.number:03d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_train_command(args, params, trial_dir)
        log_path = trial_dir / "train.log"
        start = time.time()
        ret = run_subprocess(cmd, log_path, stream_logs=args.stream_logs)
        elapsed = time.time() - start

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("command", " ".join(cmd))
        trial.set_user_attr("elapsed_s", elapsed)

        if ret != 0:
            fail_summary = {
                "trial": trial.number,
                "state": "failed",
                "return_code": ret,
                "params": params,
                "trial_dir": str(trial_dir),
                "elapsed_s": elapsed,
                "log_path": str(log_path),
            }
            save_json(trial_dir / "trial_summary.json", fail_summary)
            upsert_trial_csv(
                csv_path,
                {
                    **params,
                    "trial": trial.number,
                    "state": "failed",
                    "trial_dir": str(trial_dir),
                    "elapsed_s": elapsed,
                },
            )
            if not args.keep_failed_trials:
                for pth in trial_dir.glob("*.pth"):
                    pth.unlink(missing_ok=True)
            raise optuna.TrialPruned(
                f"Training subprocess failed with exit code {ret}. See {log_path}"
            )

        history_path = trial_dir / "e2e_history.json"
        try:
            summary = summarize_history(
                load_history(history_path),
                args,
                baseline,
            )
        except Exception as exc:
            save_json(
                trial_dir / "trial_summary.json",
                {
                    "trial": trial.number,
                    "state": "invalid_history",
                    "params": params,
                    "error": repr(exc),
                    "trial_dir": str(trial_dir),
                    "elapsed_s": elapsed,
                },
            )
            raise optuna.TrialPruned(f"Could not score trial: {exc}") from exc

        objective_value = float(summary["objective"])
        trial.report(objective_value, step=int(summary["best_epoch"]))
        if trial.should_prune():
            raise optuna.TrialPruned("Pruned after scoring validation history.")

        row = {
            **params,
            **summary,
            "trial": trial.number,
            "state": "complete",
            "trial_dir": str(trial_dir),
            "elapsed_s": elapsed,
        }
        save_json(trial_dir / "trial_summary.json", row)
        upsert_trial_csv(csv_path, row)

        for key, value in summary.items():
            trial.set_user_attr(key, value)

        try:
            existing_best = study.best_trial
        except ValueError:
            existing_best = None
        if existing_best is None or objective_value >= float(existing_best.value):
            best_payload = {
                "best_trial": trial.number,
                "best_value": objective_value,
                "best_params": params,
                "best_user_attrs": summary,
            }
        else:
            best_payload = {
                "best_trial": existing_best.number,
                "best_value": existing_best.value,
                "best_params": existing_best.params,
                "best_user_attrs": existing_best.user_attrs,
            }
        save_json(
            best_path,
            {
                **best_payload,
                "current_trial": trial.number,
                "current_objective": objective_value,
                "current_params": params,
                "current_summary": summary,
                "csv": str(csv_path),
            },
        )

        print(
            f"Trial {trial.number:03d}: objective={objective_value:.6f} "
            f"base={summary['base_score']:.6f} "
            f"day_vif={summary['day_vif']:.4f} "
            f"night_vif={summary['night_vif']:.4f} "
            f"day_unique={summary['day_unique']:.4f} "
            f"night_unique={summary['night_unique']:.4f} "
            f"best_epoch={summary['best_epoch']}"
        )
        return objective_value

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    complete_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise SystemExit(
            f"Optuna finished without a completed trial. Check trial logs in {output_dir}."
        )

    best = study.best_trial
    best_payload = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "output_dir": str(output_dir),
        "trials_csv": str(csv_path),
        "objective": {
            "formula": (
                "score = guarded_or_raw_composite + day_vif_weight*day_vif "
                "- day_vif_penalty*max(0, day_vif_target-day_vif) "
                "- night_vif_penalty*max(0, night_floor-night_vif) "
                "- unique_penalty*(max(0, unique_floor-day_unique) + "
                "max(0, unique_floor-night_unique))"
            ),
            "day_vif_target": args.day_vif_target,
            "night_vif_floor_ratio": args.night_vif_floor_ratio,
            "unique_floor": args.unique_floor,
            "day_vif_weight": args.day_vif_weight,
            "day_vif_penalty": args.day_vif_penalty,
            "night_vif_penalty": args.night_vif_penalty,
            "unique_penalty": args.unique_penalty,
        },
    }
    save_json(best_path, best_payload)

    print("\nBest trial")
    print(f"  number:    {best.number}")
    print(f"  objective: {best.value:.6f}")
    print(f"  day_vif:   {best.user_attrs.get('day_vif')}")
    print(f"  night_vif: {best.user_attrs.get('night_vif')}")
    print(f"  day_unique:   {best.user_attrs.get('day_unique')}")
    print(f"  night_unique: {best.user_attrs.get('night_unique')}")
    print(f"  params:    {json.dumps(best.params, indent=2)}")
    print(f"\nSaved: {best_path}")
    print(f"Trials CSV: {csv_path}")


if __name__ == "__main__":
    main()
