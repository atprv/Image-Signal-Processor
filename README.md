# Image Signal Processor

[![License: MIT](https://img.shields.io/github/license/atprv/Image-Signal-Processor)](LICENSE)
[![CI](https://github.com/atprv/Image-Signal-Processor/actions/workflows/ci.yml/badge.svg)](https://github.com/atprv/Image-Signal-Processor/actions/workflows/ci.yml)
[![CodeQL](https://github.com/atprv/Image-Signal-Processor/actions/workflows/codeql.yml/badge.svg)](https://github.com/atprv/Image-Signal-Processor/actions/workflows/codeql.yml)
[![Gitleaks](https://github.com/atprv/Image-Signal-Processor/actions/workflows/gitleaks.yml/badge.svg)](https://github.com/atprv/Image-Signal-Processor/actions/workflows/gitleaks.yml)
![Lint: Ruff](https://img.shields.io/badge/lint-Ruff-46A758)
![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)

PyTorch-based image signal processing research repository.

The project goal is automatic end-to-end optimization of neural-network image processing algorithm parameters. The repository contains a differentiable traditional ISP baseline, residual CNN training components, quality/evaluation metrics, Optuna-based tuning utilities, Colab notebooks, and published experiment artifacts from the diploma workflow.

## What The Project Does

- Loads camera-specific ISP parameters from TOML
- Reads RAW Bayer video streams stored as 16-bit `.bin` files
- Runs a modular ISP pipeline implemented with `torch.nn.Module`
- Produces NV12 YUV output suitable for downstream video processing
- Trains a residual CNN on top of differentiable ISP outputs
- Evaluates quality with VIF, NRQM, UNIQUE, L1_Y, and L1_UV metrics
- Runs sanity checks, warm-start pretraining, E2E training, full-video evaluation, and Optuna tuning
- Publishes compact checkpoints, reports, and research artifacts alongside the code
- Supports both CPU execution and CUDA acceleration

## Repository Layout

```text
artifacts/
  baselines/            # baseline metrics and normalization weights
  checkpoints/          # pretraining, E2E, and Optuna outputs
  results/              # rendered evaluation reports
  sanity/               # overfit and smoke-test artifacts
isp/
  color/                # YUV/RGB conversion helpers
  config/               # TOML loading, tensor preparation, scene presets
  data/                 # HDF5 patch dataset utilities
  evaluation/           # quality metric aggregation and composite scoring
  io/                   # RAW/NV12 readers and YUV writers
  models/               # residual CNN model
  pipeline/             # ISP pipeline and individual stages
  training/             # differentiable training losses and training steps
scripts/
  run_traditional_isp.py
  run_baseline.py
  run_quality_overfit_test.py
  run_pretrain_cnn.py
  run_e2e_setup.py
  run_e2e_train.py
  run_checkpoint_full_render_eval.py
  run_optuna_isp_knobs.py
  utility scripts for data prep, plots, diagnostics, and evaluation
metrics/
  standalone VIF and metric calculation helpers
notebooks/
  pretrain_cnn_colab.ipynb
  e2e_train_colab.ipynb
  full_video_eval_colab.ipynb
  optuna_isp_knobs_colab.ipynb
  sanity/
dataset/
  public split/manifest descriptors, not large HDF5 payloads
examples/
  minimal_camera.toml
tests/
  unit tests and smoke tests
docs/
  ARCHITECTURE.md
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- toml
- h5py, Pillow, tqdm for training/data utilities
- pandas, matplotlib, pyiqa, and Optuna for research metrics and tuning

## Quick Smoke Test

Generate a small synthetic RAW clip:

```powershell
python -m scripts.generate_synthetic_raw `
  --output .\demo\synthetic.bin `
  --width 64 `
  --height 64 `
  --frames 4
```

Run the ISP pipeline on CPU using the included example camera profile:

```powershell
python -m scripts.run_traditional_isp `
  --video .\demo\synthetic.bin `
  --config .\examples\minimal_camera.toml `
  --output .\demo\synthetic_nv12.yuv `
  --device cpu `
  --max-frames 4 `
  --ltm-radius 2 `
  --raw-y-blur-radius 2
```

## Real Input Data

To process a real clip you need:

1. A Bayer RAW video stream saved as a contiguous 16-bit `.bin` file.
2. A camera TOML configuration with at least `[img]`, `[decompanding]`, and `[ccm]` sections.
3. An output path for the generated NV12 `.yuv` file.

The repository ships a minimal example config at [examples/minimal_camera.toml](examples/minimal_camera.toml), but it is only a neutral demo profile. For production or research use, replace it with camera-specific calibration data.

## Typical Usage

```powershell
python -m scripts.run_traditional_isp `
  --video .\input\capture.bin `
  --config .\config\camera.toml `
  --output .\output\capture_nv12.yuv `
  --device cuda
```

## Development Workflow

Run the same checks locally that CI runs:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

To auto-format the repository:

```powershell
python -m ruff format .
```

Optional pre-commit hooks are configured in [.pre-commit-config.yaml](.pre-commit-config.yaml).

## Experiments

The notebooks in [notebooks](notebooks) are cleaned for publication: outputs are stripped, execution counts are reset, and each notebook starts with a formal experiment description. They are intended to be run in Colab or on a local environment with the required datasets and checkpoints available.

This repository now intentionally commits selected experiment artifacts that are useful for reading the diploma workflow end-to-end:

- baseline metrics and normalization weights in [artifacts/baselines](artifacts/baselines)
- quality-overfit sanity history and figures in [artifacts/sanity](artifacts/sanity)
- warm-start CNN checkpoint and evaluation summaries in [artifacts/checkpoints/cnn_pretrain](artifacts/checkpoints/cnn_pretrain)
- E2E smoke artifacts in [artifacts/sanity/e2e_smoke](artifacts/sanity/e2e_smoke)
- final E2E checkpoints in [artifacts/checkpoints/e2e_quality](artifacts/checkpoints/e2e_quality)
- full-video evaluation reports in [artifacts/results](artifacts/results)
- Optuna search outputs in [artifacts/checkpoints/optuna_tuning](artifacts/checkpoints/optuna_tuning)

Large source data such as RAW/YUV clips, local HDF5 training datasets, and private environment files are still excluded from git.

## Documentation

- Project architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## How To Help

- Open a bug report with a failing command, expected behavior, and minimal reproducible input.
- Open a feature request describing the user scenario and expected output.
- Before creating a pull request, run `ruff`, formatting checks, and `pytest`.
- Use clear commit messages that describe one logical block of work.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
