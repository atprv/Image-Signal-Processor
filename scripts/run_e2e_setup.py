"""
End-to-end training setup verification.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.optim import Adam

from isp.config.config_reader import read_config
from isp.data.dataset_utils import create_dataloader
from isp.models.residual_cnn import ResidualCNN, count_trainable_parameters
from isp.pipeline.pipeline import ISPPipeline
from isp.training.training_utils import e2e_train_step

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


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end setup verification")
    p.add_argument("--train-h5", default="dataset/train_patches.h5")
    p.add_argument("--config", default="data/imx623.toml")
    p.add_argument("--ckpt", default="artifacts/checkpoints/cnn_pretrained.pth")
    p.add_argument("--lr-isp", type=float, default=1e-4)
    p.add_argument("--lr-cnn", type=float, default=5e-4)
    p.add_argument("--lambda-uv", type=float, default=1.0)
    p.add_argument(
        "--batch-size", type=int, default=4, help="Small batch for setup check (ISP runs per-patch)"
    )
    p.add_argument("--device", default="cpu", help="cpu recommended — only 1 step needed")
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p).resolve()


def unfreeze_isp(isp):
    """Unfreeze all ISP parameters for E2E training."""
    n_params = 0
    n_tensors = 0
    for name, param in isp.named_parameters():
        param.requires_grad = True
        n_params += param.numel()
        n_tensors += 1
        print(
            f"  ISP param: {name}  shape={tuple(param.shape)}  requires_grad={param.requires_grad}"
        )
    print(f"ISP unfrozen: {n_tensors} tensors, {n_params} scalar params")
    return n_tensors


def snapshot_isp_params(isp):
    """Take a snapshot of all ISP parameter values."""
    return {name: param.detach().clone() for name, param in isp.named_parameters()}


def compare_snapshots(before, after):
    """Compare two parameter snapshots, return per-tensor max abs diff."""
    diffs = {}
    for name in before:
        diff = (after[name] - before[name]).abs().max().item()
        diffs[name] = diff
    return diffs


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    config_path = resolve(args.config)
    config = read_config(str(config_path), device=device)
    pattern = str(config["img"].get("bayer_pattern", "RGGB"))

    train_h5 = resolve(args.train_h5)
    ckpt_path = resolve(args.ckpt)

    if not ckpt_path.exists():
        print(f"ERROR: pretrained checkpoint not found at {ckpt_path}")
        print("Run scripts/run_pretrain_cnn.py first to create cnn_pretrained.pth")
        sys.exit(1)

    print("=" * 70)
    print("End-to-end setup verification")
    print("=" * 70)

    print("\n[1/5] Building ISP with day params and unfreezing...")
    isp = ISPPipeline(config, device=device, **ISP_PARAMS_DAY)
    isp.train()
    n_isp_tensors = unfreeze_isp(isp)
    if n_isp_tensors == 0:
        print("ERROR: ISP has no nn.Parameter — cannot do E2E")
        sys.exit(1)

    print(f"\n[2/5] Loading pretrained CNN from {ckpt_path}...")
    cnn = ResidualCNN(
        in_channels=3, hidden_channels=32, out_channels=3, num_blocks=5, num_groups=8
    ).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    cnn.load_state_dict(ckpt["model_state_dict"])
    cnn.train()
    print(f"  CNN loaded from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"  CNN params: {count_trainable_parameters(cnn):,}")

    print(f"\n[3/5] Building Adam with lr_isp={args.lr_isp}, lr_cnn={args.lr_cnn}...")
    isp_params = [p for p in isp.parameters() if p.requires_grad]
    cnn_params = [p for p in cnn.parameters() if p.requires_grad]
    optimizer = Adam(
        [
            {"params": isp_params, "lr": args.lr_isp, "name": "isp"},
            {"params": cnn_params, "lr": args.lr_cnn, "name": "cnn"},
        ]
    )
    for g in optimizer.param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  group '{g['name']}': lr={g['lr']}, {len(g['params'])} tensors, {n} scalars")

    print("\n[4/5] Loading 1 batch and running 1 e2e step...")
    loader = create_dataloader(
        h5_path=str(train_h5),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        return_meta=False,
    )
    batch = next(iter(loader))
    print(
        f"  Batch shapes: raw={tuple(batch['raw'].shape)}, "
        f"y_ref={tuple(batch['y_ref'].shape)}, "
        f"uv_ref={tuple(batch['uv_ref'].shape)}"
    )

    isp_before = snapshot_isp_params(isp)

    metrics = e2e_train_step(
        isp=isp,
        cnn=cnn,
        optimizer=optimizer,
        batch=batch,
        pattern=pattern,
        lambda_uv=args.lambda_uv,
    )

    print(
        f"\n  loss={metrics['loss']:.6f}  "
        f"l1_y={metrics['l1_y']:.6f}  l1_uv={metrics['l1_uv']:.6f}  "
        f"vif={metrics['vif']:.6f}"
    )
    print(
        f"  CNN grads — head: {metrics['head_grad_mean']:.2e}  "
        f"body: {metrics['body_grad_mean']:.2e}  "
        f"tail: {metrics['tail_grad_mean']:.2e}"
    )
    print(
        f"  ISP grads — ccm: {metrics['isp_ccm_grad_mean']:.2e}  "
        f"gamma: {metrics['isp_gamma_grad_mean']:.2e}"
    )

    print("\n[5/5] Verifying ISP gradients and parameter changes...")
    print(f"\n  {'ISP param':<35s} {'has_grad':>10s} {'grad_mean':>14s} {'param_diff':>14s}")
    print("  " + "-" * 75)

    isp_after = snapshot_isp_params(isp)
    diffs = compare_snapshots(isp_before, isp_after)

    all_grads_ok = True
    all_changes_ok = True
    for name, param in isp.named_parameters():
        has_grad = param.grad is not None
        grad_mean = param.grad.abs().mean().item() if has_grad else 0.0
        diff = diffs[name]
        grad_status = "OK" if has_grad and grad_mean > 0 else "FAIL"
        if not (has_grad and grad_mean > 0):
            all_grads_ok = False
        if diff <= 0:
            all_changes_ok = False
        print(f"  {name:<35s} {grad_status:>10s} {grad_mean:>14.2e} {diff:>14.2e}")

    print("\n" + "=" * 70)
    if all_grads_ok and all_changes_ok:
        print("SUCCESS: E2E setup works.")
        print("  - ISP parameters unfrozen and receive gradients")
        print("  - ISP parameters change after optimizer.step()")
        print("  - CNN gradients are non-zero")
        print("  - L1 proxy-loss is the training loss; VIF is monitor-only")
    else:
        print("FAILED:")
        if not all_grads_ok:
            print("  - Some ISP parameters did NOT receive non-zero gradients")
        if not all_changes_ok:
            print("  - Some ISP parameters did NOT change after step()")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
