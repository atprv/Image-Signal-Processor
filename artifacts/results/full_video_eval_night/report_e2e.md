# Full video render + metric report

## Run setup
- mode: `e2e`
- checkpoint: `/content/ISP/artifacts/checkpoints/e2e_quality/e2e_best.pth`
- optuna overrides: `None`
- device: `cuda`
- scenes: `night`
- metric_stride: `1`
- compute_iqa: `True`

## Overall
- frames: `1261`
- L1_Y: `0.233364`
- L1_UV: `0.024139`
- VIF: `0.947405`
- NRQM: `8.065421`
- UNIQUE: `1.196709`
- Composite: `2.152850`

## Per Scene
### night
- output: `/content/full_video_eval_outputs/videos/night_e2e.yuv`
- rendered frames: `1261`
- render FPS: `2.263`
- eval frames: `1261`
- L1_Y: `0.233364`
- L1_UV: `0.024139`
- VIF: `0.947405`
- NRQM: `8.065421`
- UNIQUE: `1.196709`
- Composite: `2.152850`
