from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def composite(vif: float, nrqm: float, unique: float) -> float:
    return vif + 0.1 * nrqm + (1.0 / 3.0) * unique


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f7f3",
            "axes.edgecolor": "#333333",
            "grid.color": "#d4d0c8",
        }
    )


def save_summary(pretrain: dict, best_e2e: dict, optuna_best: dict) -> None:
    base_avg = pretrain["baseline_average"]
    pre_avg = pretrain["average"]
    summary = {
        "baseline_composite": composite(base_avg["vif"], base_avg["nrqm"], base_avg["unique"]),
        "warmstart_composite": composite(pre_avg["vif"], pre_avg["nrqm"], pre_avg["unique"]),
        "best_e2e_epoch": best_e2e["epoch"],
        "best_e2e_composite": best_e2e["val_composite"],
        "best_e2e_vif": best_e2e["val_vif"],
        "best_e2e_nrqm": best_e2e["val_nrqm"],
        "best_e2e_unique": best_e2e["val_unique"],
        "best_optuna_trial": optuna_best["best_trial"],
        "best_optuna_value": optuna_best["best_value"],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def plot_stage_comparison(pretrain: dict, best_e2e: dict) -> None:
    base_avg = pretrain["baseline_average"]
    pre_avg = pretrain["average"]
    values = [
        composite(base_avg["vif"], base_avg["nrqm"], base_avg["unique"]),
        composite(pre_avg["vif"], pre_avg["nrqm"], pre_avg["unique"]),
        best_e2e["val_composite"],
    ]
    labels = ["Baseline\nISP", "Warm-start\nCNN", "Best E2E\nISP+CNN"]
    colors = ["#6a994e", "#bc4749", "#1d3557"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylabel("J")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage_comparison.png", dpi=180)
    plt.close(fig)


def plot_e2e_progress(pretrain: dict, history: list[dict]) -> None:
    base_avg = pretrain["baseline_average"]
    baseline = composite(base_avg["vif"], base_avg["nrqm"], base_avg["unique"])
    val_rows = [row for row in history if "val_composite" in row]
    epochs = [row["epoch"] for row in val_rows]
    composite_values = [row["val_composite"] for row in val_rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        epochs,
        composite_values,
        marker="o",
        color="#1d3557",
        linewidth=2.5,
        label="Validation J",
    )
    ax.axhline(
        baseline,
        color="#6a994e",
        linestyle="--",
        linewidth=2,
        label="Baseline J",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("J")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e2e_progress.png", dpi=180)
    plt.close(fig)


def plot_optuna_progress(trials: pd.DataFrame) -> None:
    completed = trials[trials["state"] == "trialstate.complete"].copy()
    completed = completed.sort_values("trial")
    completed["best_so_far"] = completed["objective"].cummax()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(
        completed["trial"],
        completed["objective"],
        s=18,
        color="#a8dadc",
        alpha=0.8,
        label="Trial value",
    )
    ax.plot(
        completed["trial"],
        completed["best_so_far"],
        color="#1d3557",
        linewidth=2.5,
        label="Best so far",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("J")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "optuna_progress.png", dpi=180)
    plt.close(fig)


def plot_per_scene(pretrain: dict, best_e2e: dict) -> None:
    scenes = ["day", "night"]
    base_vals = []
    e2e_vals = []
    for scene in scenes:
        base = pretrain["baseline_per_scene"][scene]
        best = best_e2e["val_per_scene"][scene]
        base_vals.append(composite(base["vif"], base["nrqm"], base["unique"]))
        e2e_vals.append(composite(best["vif"], best["nrqm"], best["unique"]))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(scenes))
    width = 0.32
    ax.bar(x - width / 2, base_vals, width, label="Baseline ISP", color="#6a994e")
    ax.bar(x + width / 2, e2e_vals, width, label="E2E ISP+CNN", color="#1d3557")
    ax.set_xticks(x)
    ax.set_xticklabels([scene.capitalize() for scene in scenes])
    ax.set_ylabel("J")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper left")
    for idx, value in enumerate(base_vals):
        ax.text(idx - width / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    for idx, value in enumerate(e2e_vals):
        ax.text(idx + width / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_scene_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    setup_style()
    pretrain = load_json(
        ROOT / "artifacts" / "checkpoints" / "cnn_pretrain" / "pretrain_eval_metrics.json"
    )
    history = load_json(
        ROOT
        / "artifacts"
        / "checkpoints"
        / "e2e_quality"
        / "e2e_quality_outputs"
        / "e2e_history.json"
    )
    optuna_best = load_json(
        ROOT / "artifacts" / "checkpoints" / "optuna_tuning" / "optuna_best_params.json"
    )
    trials = pd.read_csv(ROOT / "artifacts" / "checkpoints" / "optuna_tuning" / "optuna_trials.csv")

    best_e2e = max(
        (row for row in history if "val_composite" in row),
        key=lambda row: row["val_composite"],
    )

    save_summary(pretrain, best_e2e, optuna_best)
    plot_stage_comparison(pretrain, best_e2e)
    plot_e2e_progress(pretrain, history)
    plot_optuna_progress(trials)
    plot_per_scene(pretrain, best_e2e)


if __name__ == "__main__":
    main()
