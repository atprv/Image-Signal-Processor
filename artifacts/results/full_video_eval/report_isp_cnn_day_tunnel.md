# Full video render + metric report

## Run setup
- mode: `isp_cnn`
- checkpoint: `/content/ISP/artifacts/checkpoints/e2e_quality/e2e_best.pth`
- optuna overrides: `/content/ISP/artifacts/checkpoints/optuna_tuning/optuna_best_params.json`
- device: `cuda`
- scenes: `day, tunnel`
- metric_stride: `1`
- compute_iqa: `True`

## Overall
- frames: `744`
- L1_Y: `0.083869`
- L1_UV: `0.034823`
- VIF: `0.523825`
- NRQM: `7.252136`
- UNIQUE: `0.814061`
- Composite: `1.520392`

## Per Scene
### day
- output: `/content/full_video_eval_outputs/videos/day_isp_cnn.yuv`
- rendered frames: `331`
- render FPS: `2.346`
- eval frames: `331`
- L1_Y: `0.084629`
- L1_UV: `0.034999`
- VIF: `0.538675`
- NRQM: `7.165368`
- UNIQUE: `0.881442`
- Composite: `1.549025`

### tunnel
- output: `/content/full_video_eval_outputs/videos/tunnel_isp_cnn.yuv`
- rendered frames: `413`
- render FPS: `2.316`
- eval frames: `413`
- L1_Y: `0.083260`
- L1_UV: `0.034681`
- VIF: `0.511924`
- NRQM: `7.321677`
- UNIQUE: `0.760058`
- Composite: `1.497444`
