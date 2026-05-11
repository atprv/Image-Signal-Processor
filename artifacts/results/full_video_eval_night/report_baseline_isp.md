# Full video render + metric report

## Run setup
- mode: `baseline_isp`
- checkpoint: `baseline_isp`
- optuna overrides: `None`
- device: `cuda`
- scenes: `night`
- metric_stride: `1`
- compute_iqa: `True`

## Overall
- frames: `1261`
- L1_Y: `0.065110`
- L1_UV: `0.010294`
- VIF: `0.587452`
- NRQM: `8.177230`
- UNIQUE: `-0.009012`
- Composite: `1.402171`

## Per Scene
### night
- output: `/content/full_video_eval_outputs/videos/night_baseline_isp.yuv`
- rendered frames: `1261`
- render FPS: `14.782`
- eval frames: `1261`
- L1_Y: `0.065110`
- L1_UV: `0.010294`
- VIF: `0.587452`
- NRQM: `8.177230`
- UNIQUE: `-0.009012`
- Composite: `1.402171`
