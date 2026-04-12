# Third-Party Notices

This repository currently contains original project code plus references to open-source Python dependencies. No third-party source files, images, datasets, or generated binaries are committed into the repository.

The direct dependencies below were inspected from local package metadata on 2026-04-12:

| Component | Purpose | Version inspected locally | License |
| --- | --- | --- | --- |
| PyTorch (`torch`) | tensor math and pipeline execution | 2.5.1 | BSD-3-Clause |
| NumPy (`numpy`) | RAW/YUV array I/O and synthetic data generation | 2.2.6 | BSD-style permissive license |
| toml | camera configuration parsing | 0.10.2 | MIT |
| pytest | unit testing | 9.0.2 | MIT |
| Ruff | linting and formatting | 0.15.9 | MIT |

## Compatibility Notes

- The project itself is distributed under the MIT License.
- The inspected runtime and development dependencies above use permissive licenses and are compatible with MIT distribution.
- If you later pin different dependency versions or add external assets, re-check their licenses before making the repository public.
