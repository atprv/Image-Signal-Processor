# Full video render + metric report

## Run setup
- mode: `e2e`
- checkpoint: `/content/ISP/artifacts/checkpoints/e2e_quality/e2e_best.pth`
- optuna overrides: `None`
- device: `cuda`
- scenes: `day, tunnel`
- metric_stride: `1`
- compute_iqa: `True`

## Overall
- frames: `744`
- L1_Y: `0.085873`
- L1_UV: `0.033330`
- VIF: `0.612372`
- NRQM: `6.610120`
- UNIQUE: `0.637391`
- Composite: `1.485847`

## Per Scene
### day
- output: `/content/full_video_eval_outputs/videos/day_e2e.yuv`
- rendered frames: `331`
- render FPS: `2.263`
- eval frames: `331`
- L1_Y: `0.087405`
- L1_UV: `0.033995`
- VIF: `0.631936`
- NRQM: `6.342055`
- UNIQUE: `0.713867`
- Composite: `1.504097`

### tunnel
- output: `/content/full_video_eval_outputs/videos/tunnel_e2e.yuv`
- rendered frames: `413`
- render FPS: `2.283`
- eval frames: `413`
- L1_Y: `0.084644`
- L1_UV: `0.032797`
- VIF: `0.596692`
- NRQM: `6.824961`
- UNIQUE: `0.576099`
- Composite: `1.471221`
