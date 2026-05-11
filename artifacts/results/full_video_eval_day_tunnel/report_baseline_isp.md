# Full video render + metric report

## Run setup
- mode: `baseline_isp`
- checkpoint: `baseline_isp`
- optuna overrides: `None`
- device: `cuda`
- scenes: `day, tunnel`
- metric_stride: `1`
- compute_iqa: `True`

## Overall
- frames: `744`
- L1_Y: `0.057843`
- L1_UV: `0.012138`
- VIF: `0.637671`
- NRQM: `6.678060`
- UNIQUE: `-0.028488`
- Composite: `1.295981`

## Per Scene
### day
- output: `/content/full_video_eval_outputs/videos/day_baseline_isp.yuv`
- rendered frames: `331`
- render FPS: `13.880`
- eval frames: `331`
- L1_Y: `0.053207`
- L1_UV: `0.011997`
- VIF: `0.684043`
- NRQM: `6.320261`
- UNIQUE: `0.015635`
- Composite: `1.321281`

### tunnel
- output: `/content/full_video_eval_outputs/videos/tunnel_baseline_isp.yuv`
- rendered frames: `413`
- render FPS: `15.004`
- eval frames: `413`
- L1_Y: `0.061559`
- L1_UV: `0.012251`
- VIF: `0.600506`
- NRQM: `6.964819`
- UNIQUE: `-0.063850`
- Composite: `1.275705`
