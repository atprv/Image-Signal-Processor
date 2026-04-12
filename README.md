# Image Signal Processor

![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)
![CI: GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)
![Lint: Ruff](https://img.shields.io/badge/lint-Ruff-46A758)
![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)
![Analysis: CodeQL](https://img.shields.io/badge/analysis-CodeQL-24292F)

PyTorch-based image signal processing repository.

The long-term project goal is automatic end-to-end optimization of neural-network image processing algorithm parameters. The current repository state contains the traditional ISP baseline for RAW Bayer to NV12 YUV processing, together with the repository infrastructure needed for publication and further development.

> The badge row above is intentionally static until the repository is published. After the first push to GitHub, replace it with live workflow badges using your actual GitHub owner and repository name.

## What The Project Does

- Establishes the base repository for the broader Image Signal Processor project
- Loads camera-specific ISP parameters from TOML
- Reads RAW Bayer video streams stored as 16-bit `.bin` files
- Runs a modular ISP pipeline implemented with `torch.nn.Module`
- Produces NV12 YUV output suitable for downstream video processing
- Supports both CPU execution and CUDA acceleration

## Repository Layout

```text
isp/
  config/              # TOML loading and tensor preparation
  io/                  # RAW/NV12 readers and YUV writers
  pipeline/            # ISP pipeline and individual stages
scripts/
  generate_synthetic_raw.py
  run_traditional_isp.py
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

## Documentation

- Project architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## How To Help

- Open a bug report with a failing command, expected behavior, and minimal reproducible input.
- Open a feature request describing the user scenario and expected output.
- Before creating a pull request, run `ruff`, formatting checks, and `pytest`.
- Use clear commit messages; Conventional Commits are recommended.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
