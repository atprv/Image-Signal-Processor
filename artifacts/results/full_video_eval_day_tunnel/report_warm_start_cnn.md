# Full video render + metric report

## Run setup
- mode: `warm_start_cnn`
- checkpoint: `/content/ISP/artifacts/checkpoints/cnn_pretrain/cnn_pretrained.pth`
- optuna overrides: `None`
- device: `cuda`
- scenes: `day, tunnel`
- metric_stride: `1`
- compute_iqa: `True`

## Overall
- frames: `744`
- L1_Y: `0.069372`
- L1_UV: `0.016274`
- VIF: `0.300075`
- NRQM: `6.548827`
- UNIQUE: `0.147833`
- Composite: `1.004235`

## Per Scene
### day
- output: `/content/full_video_eval_outputs/videos/day_warm_start_cnn.yuv`
- rendered frames: `331`
- render FPS: `2.317`
- eval frames: `331`
- L1_Y: `0.065197`
- L1_UV: `0.015566`
- VIF: `0.346340`
- NRQM: `6.065709`
- UNIQUE: `0.145497`
- Composite: `1.001410`

### tunnel
- output: `/content/full_video_eval_outputs/videos/tunnel_warm_start_cnn.yuv`
- rendered frames: `413`
- render FPS: `2.267`
- eval frames: `413`
- L1_Y: `0.072717`
- L1_UV: `0.016842`
- VIF: `0.262996`
- NRQM: `6.936022`
- UNIQUE: `0.149704`
- Composite: `1.006500`
