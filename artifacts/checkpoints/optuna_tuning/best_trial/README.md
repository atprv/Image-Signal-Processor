# Best Optuna trial

After the sweep finishes, copy the selected trial's artifacts here. Typical
contents:

- `trial_summary.json` — sampled params, objective, val metrics.
- `e2e_history.json` — per-epoch train/val records of the trial.
- `e2e_best.pth` — the trial's best-by-composite checkpoint, if saved.
- `e2e_training_curves.png` — curves for the trial, if generated.

The selected configuration is the input to a longer confirmation run; the
confirmation outputs go under `artifacts/checkpoints/e2e_quality/` (or a
sibling folder named `e2e_quality_optuna_confirm/` if both runs need to be
kept side-by-side).
