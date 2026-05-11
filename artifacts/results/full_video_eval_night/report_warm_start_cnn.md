# Full video render + metric report

## Run setup
- mode: `warm_start_cnn`
- checkpoint: `/content/ISP/artifacts/checkpoints/cnn_pretrain/cnn_pretrained.pth`
- optuna overrides: `None`
- device: `cuda`
- scenes: `night`
- metric_stride: `1`
- compute_iqa: `True`

## Overall
- frames: `1261`
- L1_Y: `0.023348`
- L1_UV: `0.007853`
- VIF: `0.329107`
- NRQM: `8.033609`
- UNIQUE: `-0.311179`
- Composite: `1.028741`

## Per Scene
### night
- output: `/content/full_video_eval_outputs/videos/night_warm_start_cnn.yuv`
- rendered frames: `1261`
- render FPS: `2.311`
- eval frames: `1261`
- L1_Y: `0.023348`
- L1_UV: `0.007853`
- VIF: `0.329107`
- NRQM: `8.033609`
- UNIQUE: `-0.311179`
- Composite: `1.028741`
