# Third-Party Notices

This repository currently contains original project code plus references to open-source Python dependencies. No third-party source files, images, datasets, or generated binaries are committed into the repository.

The direct dependencies below were inspected from local package metadata on 2026-04-26:

| Component | Purpose | Version inspected locally | License |
| --- | --- | --- | --- |
| PyTorch (`torch`) | tensor math and pipeline execution | 2.5.1 | BSD-3-Clause |
| NumPy (`numpy`) | RAW/YUV array I/O and synthetic data generation | 2.2.6 | BSD-style permissive license |
| toml | camera configuration parsing | 0.10.2 | MIT |
| h5py | HDF5 patch datasets | 3.12.1 | BSD-3-Clause |
| Pillow (`PIL`) | image loading/saving in metrics and notebooks | 11.0.0 | HPND-style permissive license |
| tqdm | progress bars for training/evaluation scripts | 4.67.1 | MPL-2.0 and MIT |
| pandas | CSV metric summaries | 2.2.3 | BSD-3-Clause |
| matplotlib | plots and notebook visualizations | 3.9.2 | PSF/BSD-style permissive license |
| pyiqa | NRQM, UNIQUE, MS-SSIM quality metrics | 0.1.15.post2 | Apache-2.0 |
| pytest | unit testing | 9.0.2 | MIT |
| Ruff | linting and formatting | 0.15.9 | MIT |

## Compatibility Notes

- The project itself is distributed under the MIT License.
- The inspected runtime and development dependencies above use permissive licenses and are compatible with MIT distribution.
- If you later pin different dependency versions or add external assets, re-check their licenses before making the repository public.
