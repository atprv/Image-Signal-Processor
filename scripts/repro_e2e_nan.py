"""
Reproduce NaN-on-batch-1 directly, no scene-aware path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from isp.config.config_reader import read_config
from isp.data.dataset_utils import create_dataloader
from isp.models.residual_cnn import ResidualCNN
from isp.pipeline.pipeline import ISPPipeline
from isp.training.quality_loss import QualityLossWeights, compute_quality_loss
from isp.training.training_utils import forward_isp_cnn_diff

ISP_PARAMS_DAY = dict(
    denoise_eps=1e-12,
    ltm_a=0.5,
    ltm_detail_gain=30,
    ltm_detail_threshold=0.35,
    hist_target_mean=0.445,
    hist_target_std=0.162,
    post_denoise_radius=4,
    post_denoise_eps=0.001,
    raw_y_full_blend=0.4,
    sharp_amount=0.3,
)


def dump_one(label, p):
    v = p.detach()
    finite = bool(torch.isfinite(v).all().item())
    g = p.grad
    g_finite = bool(torch.isfinite(g).all().item()) if g is not None else None
    g_max = float(g.abs().max().item()) if g is not None else None
    print(
        f"  {label:24s} val={float(v.flatten()[0].item()):+.5e} val_finite={finite}  "
        f"grad_finite={g_finite} |g|max={g_max}"
    )


def main():
    torch.manual_seed(42)
    device = "cpu"
    config = read_config(str(ROOT / "data" / "imx623.toml"), device=device)
    pattern = str(config["img"].get("bayer_pattern", "RGGB"))

    isp = ISPPipeline(config, device=device, **ISP_PARAMS_DAY)
    for p in isp.parameters():
        p.requires_grad = True
    isp.train()

    cnn = ResidualCNN(
        in_channels=3, hidden_channels=32, out_channels=3, num_blocks=5, num_groups=8
    ).to(device)
    cnn.train()

    optimizer = torch.optim.Adam(
        [
            {"params": [p for p in isp.parameters() if p.requires_grad], "lr": 1e-6, "name": "isp"},
            {"params": [p for p in cnn.parameters() if p.requires_grad], "lr": 5e-6, "name": "cnn"},
        ]
    )

    weights = QualityLossWeights(
        w_ssim=1.0,
        w_vif=5.0,
        w_unique=0.3,
        w_l1_y=0.0,
        w_uv=1.0,
        w_vif_floor=50.0,
        vif_target=0.7,
        w_unique_floor=10.0,
        unique_target=0.0,
    )

    loader = create_dataloader(
        h5_path=str(ROOT / "dataset" / "e2e_smoke_train.h5"),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        return_meta=True,
    )

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 3:
            break
        scene_str = batch.get("scene_id").tolist()
        print(f"\n=== batch {batch_idx} (scene_ids={scene_str}) ===")

        raw = batch["raw"]
        y_ref = batch["y_ref"]
        uv_ref = batch["uv_ref"]

        optimizer.zero_grad(set_to_none=True)
        outputs = forward_isp_cnn_diff(isp, cnn, raw)
        loss_dict = compute_quality_loss(
            raw_batch=outputs["raw_12bit"],
            y_pred=outputs["y_pred"],
            uv_pred=outputs["uv_pred"],
            y_ref=y_ref,
            uv_ref=uv_ref,
            pattern=pattern,
            weights=weights,
        )
        loss = loss_dict["loss"]
        loss_value = float(loss.item())
        print(f"  forward loss = {loss_value:.6f}  finite={math.isfinite(loss_value)}")
        if not math.isfinite(float(loss.item())):
            print("  *** NaN already in forward, stopping ***")
            break

        loss.backward()

        print("  --- per-param grad after backward (before step) ---")
        for name, p in sorted(isp.named_parameters()):
            dump_one(name, p)

        optimizer.step()

        print("  --- per-param post-step ---")
        for name, p in sorted(isp.named_parameters()):
            v = p.detach()
            finite = bool(torch.isfinite(v).all().item())
            print(f"  {name:24s} val={float(v.flatten()[0].item()):+.5e}  finite={finite}")


if __name__ == "__main__":
    main()
