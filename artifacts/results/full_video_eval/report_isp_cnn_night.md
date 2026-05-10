# Full video render + metric report

## Run setup
- mode: `isp_cnn`
- checkpoint: `/content/ISP/artifacts/checkpoints/e2e_quality/e2e_best.pth`
- optuna overrides: `/content/ISP/artifacts/checkpoints/optuna_tuning/optuna_best_params.json`
- device: `cuda`
- scenes: `night`
- metric_stride: `1`
- compute_iqa: `True`

## Overall
- frames: `1261`
- L1_Y: `0.215573`
- L1_UV: `0.026108`
- VIF: `0.894793`
- NRQM: `8.319410`
- UNIQUE: `1.545832`
- Composite: `2.242011`

## Per Scene
### night
- output: `/content/full_video_eval_outputs/videos/night_isp_cnn.yuv`
- rendered frames: `1261`
- render FPS: `2.369`
- eval frames: `1261`
- L1_Y: `0.215573`
- L1_UV: `0.026108`
- VIF: `0.894793`
- NRQM: `8.319410`
- UNIQUE: `1.545832`
- Composite: `2.242011`
