"""
Smoke test: verify hand-applied ISP optimizations preserve numerical behavior
and gradient flow.

Covers:
1. Sharpening separable Gaussian == reference 2D Gaussian
2. HistogramNormalization fast-path == full path at zero targets
3. RGBtoYUV fast-paths == full path (with and without raw_green)
4. PostGammaDenoise separable box filter == reference 2D box
5. LTM separable box filter == reference 2D box
6. ISPPipeline._validate_params accepts canonical day/night/tunnel configs
7. Full pipeline forward+backward propagates finite gradient to all
   trainable nn.Parameter tensors
8. RAW-green gamma path stays finite on fully dark input
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isp.config.config_reader import read_config
from isp.pipeline.pipeline import ISPPipeline
from isp.pipeline.stages.histogram_normalization import HistogramNormalization
from isp.pipeline.stages.ltm import LTM
from isp.pipeline.stages.post_gamma_denoise import PostGammaDenoise
from isp.pipeline.stages.rgb2yuv import RGBtoYUV
from isp.pipeline.stages.sharpening import Sharpening

CONFIG_PATH = ROOT / "data" / "imx623.toml"
TOL = 1e-5


def _load_config():
    return read_config(str(CONFIG_PATH), device="cpu")


def _ok(name: str) -> None:
    print(f"  [OK] {name}")


def _fail(name: str, msg: str) -> None:
    print(f"  [FAIL] {name}: {msg}")
    raise SystemExit(1)


def _check_close(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    a32 = a.detach().float()
    b32 = b.detach().float()
    if a32.shape != b32.shape:
        _fail(name, f"shape mismatch {tuple(a32.shape)} vs {tuple(b32.shape)}")
    diff = (a32 - b32).abs().max().item()
    if not torch.isfinite(torch.tensor(diff)):
        _fail(name, "non-finite diff")
    if diff > tol:
        _fail(name, f"max abs diff = {diff:.3e} (tol={tol:.0e})")
    _ok(f"{name} (max abs diff = {diff:.3e})")


# ---------------------------------------------------------------------------
# 1. Sharpening separable blur
# ---------------------------------------------------------------------------
def test_sharpening_separable_blur():
    print("[1] Sharpening separable Gaussian blur")
    torch.manual_seed(0)
    H, W = 64, 96
    x = torch.rand(H, W, 3)

    # Compare separable conv (current implementation) against a reference
    # 2D Gaussian convolution applied per-channel.
    sharp = Sharpening(amount=0.0, radius=1.5, threshold=0.0)
    sharp.eval()
    blurred_sep = sharp._gaussian_blur(x)

    kernel_2d = sharp.kernel  # [1, 1, k, k]
    pad = sharp.pad
    x_4d = x.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    x_pad = F.pad(x_4d, (pad, pad, pad, pad), mode="reflect")
    kernel_dw = kernel_2d.expand(3, 1, -1, -1).contiguous()
    blurred_ref = F.conv2d(x_pad, kernel_dw, groups=3).squeeze(0).permute(1, 2, 0)

    _check_close("Sharpening separable vs 2D blur", blurred_sep, blurred_ref, tol=1e-5)

    # Identity fast-path at amount=0, eval, no-grad
    sharp.eval()
    with torch.no_grad():
        out_fast = sharp(x)
    _check_close("Sharpening identity fast-path (amount=0)", out_fast, x, tol=0.0)

    # Non-trivial amount - just verify forward runs and stays finite
    sharp_active = Sharpening(amount=0.5, radius=1.5, threshold=0.02)
    sharp_active.eval()
    with torch.no_grad():
        out_active = sharp_active(x)
    if not torch.isfinite(out_active).all():
        _fail("Sharpening active forward", "non-finite output")
    _ok("Sharpening active forward stays finite")


# ---------------------------------------------------------------------------
# 2. HistogramNormalization fast-path
# ---------------------------------------------------------------------------
def test_hist_norm_fast_path():
    print("[2] HistogramNormalization fast-path")
    torch.manual_seed(1)
    x = torch.rand(48, 64, 3)

    hist = HistogramNormalization(target_mean=0.0, target_std=0.0)
    hist.eval()

    # Fast path active: at zero targets and eval+no-grad, output should be
    # identical to input.
    with torch.no_grad():
        out_fast = hist(x)
    _check_close("Hist fast-path identity (zero targets)", out_fast, x, tol=0.0)

    # Active path with non-zero targets, output stays in [0, 1]
    hist2 = HistogramNormalization(target_mean=0.4, target_std=0.15)
    hist2.eval()
    with torch.no_grad():
        out_active = hist2(x)
    if not torch.isfinite(out_active).all():
        _fail("Hist active forward", "non-finite output")
    if out_active.min() < 0.0 or out_active.max() > 1.0 + 1e-5:
        _fail(
            "Hist active range",
            f"out of [0,1]: min={out_active.min().item()}, max={out_active.max().item()}",
        )
    _ok("Hist active forward stays in [0, 1]")

    # Gradient flows through trainable params at non-zero targets
    hist2.train()
    x_g = x.clone().requires_grad_(False)
    out_g = hist2(x_g)
    out_g.mean().backward()
    if hist2.target_mean.grad is None or not torch.isfinite(hist2.target_mean.grad).all():
        _fail("Hist target_mean grad", "missing or non-finite")
    if hist2.target_std.grad is None or not torch.isfinite(hist2.target_std.grad).all():
        _fail("Hist target_std grad", "missing or non-finite")
    _ok("Hist gradients flow to target_mean and target_std")


# ---------------------------------------------------------------------------
# 3. RGBtoYUV fast-paths and pack equivalence
# ---------------------------------------------------------------------------
def test_rgb2yuv_paths():
    print("[3] RGBtoYUV fast-paths and packing")
    torch.manual_seed(2)
    H, W = 32, 48
    rgb = torch.rand(H, W, 3)
    raw_green = torch.rand(H, W)

    # Identity blends in eval + no-grad: output must equal "no raw_green" path
    yuv = RGBtoYUV(raw_y_blend=0.0, raw_y_blur_radius=8, raw_y_full_blend=0.0)
    yuv.eval()
    with torch.no_grad():
        nv12_with_rg = yuv(rgb, raw_green=raw_green)
        nv12_no_rg = yuv(rgb, raw_green=None)
    _check_close(
        "RGBtoYUV identity blends (rg vs no-rg)",
        nv12_with_rg.float(),
        nv12_no_rg.float(),
        tol=0.0,
    )

    # forward_components nv12 must match forward()
    with torch.no_grad():
        comps = yuv.forward_components(rgb, raw_green=raw_green)
        nv12_direct = yuv(rgb, raw_green=raw_green)
    _check_close(
        "RGBtoYUV forward vs forward_components.nv12",
        comps["nv12"].float(),
        nv12_direct.float(),
        tol=0.0,
    )

    # Active blend: output stays finite and shape correct
    yuv_active = RGBtoYUV(raw_y_blend=0.3, raw_y_blur_radius=8, raw_y_full_blend=0.2)
    yuv_active.eval()
    with torch.no_grad():
        out_active = yuv_active(rgb, raw_green=raw_green)
    expected_size = H * W + 2 * (H // 2) * (W // 2)
    if out_active.shape != (expected_size,):
        _fail(
            "RGBtoYUV active shape",
            f"got {tuple(out_active.shape)}, expected ({expected_size},)",
        )
    _ok("RGBtoYUV active forward shape correct")

    # Gradient flows through both blend params in training mode
    yuv_active.train()
    comps_t = yuv_active.forward_components(rgb, raw_green=raw_green)
    loss = comps_t["y"].mean() + comps_t["uv"].mean()
    loss.backward()
    if yuv_active.raw_y_blend.grad is None or not torch.isfinite(yuv_active.raw_y_blend.grad).all():
        _fail("RGBtoYUV raw_y_blend grad", "missing or non-finite")
    if (
        yuv_active.raw_y_full_blend.grad is None
        or not torch.isfinite(yuv_active.raw_y_full_blend.grad).all()
    ):
        _fail("RGBtoYUV raw_y_full_blend grad", "missing or non-finite")
    _ok("RGBtoYUV gradients flow to both blend parameters")


# ---------------------------------------------------------------------------
# 4. PostGammaDenoise separable vs 2D box
# ---------------------------------------------------------------------------
def test_post_gamma_denoise_box():
    print("[4] PostGammaDenoise separable box filter")
    torch.manual_seed(3)
    H, W = 48, 64
    radius = 4
    pgd = PostGammaDenoise(radius=radius, eps=0.005)
    pgd.eval()

    t = torch.rand(1, 1, H, W)
    out_sep = pgd._box_filter(t)

    ks = 2 * radius + 1
    box_2d = torch.ones(1, 1, ks, ks) / (ks * ks)
    t_pad = F.pad(t, (radius, radius, radius, radius), mode="reflect")
    out_ref = F.conv2d(t_pad, box_2d)

    _check_close("PostGammaDenoise box filter", out_sep, out_ref, tol=1e-6)

    # Forward stays finite and shaped correctly on RGB input
    x = torch.rand(H, W, 3)
    with torch.no_grad():
        out = pgd(x)
    if out.shape != (H, W, 3):
        _fail("PostGammaDenoise output shape", f"got {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        _fail("PostGammaDenoise output", "non-finite")
    _ok("PostGammaDenoise forward shape and finiteness")


# ---------------------------------------------------------------------------
# 5. LTM separable box filter
# ---------------------------------------------------------------------------
def test_ltm_box_and_forward():
    print("[5] LTM separable box filter")
    torch.manual_seed(4)
    radius = 8
    ltm = LTM(a=0.7, b=0.0, radius=radius, eps=1e-3, downsample_factor=1.0)
    ltm.eval()

    H, W = 64, 96
    t = torch.rand(1, 1, H, W)
    out_sep = ltm._separable_box_filter(t)

    ks = 2 * radius + 1
    box_2d = torch.ones(1, 1, ks, ks) / (ks * ks)
    t_pad = F.pad(t, (radius, radius, radius, radius), mode="reflect")
    out_ref = F.conv2d(t_pad, box_2d)

    _check_close("LTM box filter (downsample=1.0)", out_sep, out_ref, tol=1e-6)

    # Forward at downsample<1 stays finite and in [0, 1]
    ltm_ds = LTM(a=0.5, b=0.0, radius=8, eps=1e-3, downsample_factor=0.5)
    ltm_ds.eval()
    x = torch.rand(H, W, 3)
    with torch.no_grad():
        out = ltm_ds(x)
    if out.shape != (H, W, 3):
        _fail("LTM downsample output shape", f"got {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        _fail("LTM downsample output", "non-finite")
    if out.min() < -1e-5 or out.max() > 1.0 + 1e-5:
        _fail("LTM downsample output range", f"min={out.min().item()}, max={out.max().item()}")
    _ok("LTM downsample forward shape, range, finiteness")


# ---------------------------------------------------------------------------
# 6. _validate_params accepts canonical configs
# ---------------------------------------------------------------------------
def test_validate_params():
    print("[6] _validate_params accepts canonical configs")

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
    ISP_PARAMS_NIGHT = dict(
        denoise_eps=1e-12,
        ltm_a=0.3,
        ltm_detail_gain=8,
        ltm_detail_threshold=0.4,
        sharp_amount=0.8,
    )
    ISP_PARAMS_TUNNEL: dict = {}

    # Apply defaults like ISPPipeline does, then validate.
    defaults = {
        "awb_max_gain": 4.0,
        "awb_lum_mask_low": 0.0,
        "awb_lum_mask_high": 1.0,
        "denoise_radius": 2,
        "denoise_eps": 1e-6,
        "ltm_a": 0.7,
        "ltm_b": 0.0,
        "ltm_radius": 8,
        "ltm_downsample": 0.5,
        "ltm_eps": 1e-3,
        "ltm_target_mean": 0.0,
        "ltm_detail_gain": 1.0,
        "ltm_detail_threshold": 0.0,
        "hist_target_mean": 0.0,
        "hist_target_std": 0.0,
        "gamma": 2.2,
        "sharp_amount": 0.0,
        "sharp_radius": 1.0,
        "sharp_threshold": 0.01,
        "raw_y_blend": 0.0,
        "raw_y_blur_radius": 8,
        "raw_y_full_blend": 0.0,
        "post_denoise_radius": 0,
        "post_denoise_eps": 0.005,
    }

    for name, override in [
        ("DAY", ISP_PARAMS_DAY),
        ("NIGHT", ISP_PARAMS_NIGHT),
        ("TUNNEL", ISP_PARAMS_TUNNEL),
    ]:
        merged = {**defaults, **override}
        try:
            ISPPipeline._validate_params(merged)
        except Exception as e:
            _fail(f"_validate_params {name}", str(e))
        _ok(f"_validate_params accepts {name}")

    # Negative case: malformed config must raise
    bad = {**defaults, "awb_max_gain": 0.5}
    try:
        ISPPipeline._validate_params(bad)
        _fail("_validate_params bad awb", "did not raise")
    except ValueError:
        _ok("_validate_params rejects awb_max_gain < 1.0")

    bad2 = {**defaults, "ltm_downsample": 0.0}
    try:
        ISPPipeline._validate_params(bad2)
        _fail("_validate_params bad ltm_downsample", "did not raise")
    except ValueError:
        _ok("_validate_params rejects ltm_downsample = 0.0")

    # Drift case: trainable knobs that wandered slightly outside their
    # nominal range during gradient descent must NOT crash eval. This
    # mirrors the failure that bit ``e2e_smoke_sanity`` (awb_lum_mask_high
    # ended an epoch at 1.0001 and triggered a hard validation error).
    drift_cases = [
        ("awb_lum_mask_high", 1.0001),
        ("awb_lum_mask_low", -0.00005),
        ("awb_max_gain", 0.9995),
        ("ltm_target_mean", 1.00012),
        ("hist_target_mean", -0.0002),
        ("hist_target_std", 1.0008),
        ("raw_y_blend", -0.0001),
        ("raw_y_full_blend", 1.00007),
        ("ltm_a", -0.00009),
        ("sharp_amount", -0.0003),
        ("sharp_threshold", -0.0001),
        ("ltm_detail_gain", -0.0008),
        ("ltm_detail_threshold", -0.0001),
        ("denoise_eps", -0.0001),
        ("post_denoise_eps", -0.00007),
    ]
    for key, value in drift_cases:
        cfg = {**defaults, key: value}
        try:
            ISPPipeline._validate_params(cfg)
        except ValueError as exc:
            _fail(f"_validate_params drift on {key}", str(exc))
    _ok(f"_validate_params tolerates {len(drift_cases)} small-drift trainable values")

    # And drift past tolerance is still rejected.
    egregious_cases = [
        ("awb_lum_mask_high", 1.5),
        ("awb_max_gain", 0.5),
        ("ltm_target_mean", 1.5),
        ("ltm_a", -0.5),
        ("sharp_amount", -0.5),
    ]
    for key, value in egregious_cases:
        cfg = {**defaults, key: value}
        try:
            ISPPipeline._validate_params(cfg)
            _fail(f"_validate_params egregious {key}={value}", "did not raise")
        except ValueError:
            pass
    _ok(f"_validate_params still rejects {len(egregious_cases)} egregious violations")


# ---------------------------------------------------------------------------
# 7. Full pipeline forward+backward gradient flow
# ---------------------------------------------------------------------------
def test_pipeline_gradients():
    print("[7] Full pipeline forward + backward gradient flow")
    config = _load_config()

    # Modest patch and active params so all stages do real work and
    # gradients propagate everywhere.
    pipe = ISPPipeline(
        config,
        device="cpu",
        denoise_eps=1e-5,
        ltm_a=0.5,
        ltm_b=0.05,
        ltm_target_mean=0.4,
        ltm_detail_gain=2.0,
        ltm_detail_threshold=0.05,
        hist_target_mean=0.4,
        hist_target_std=0.15,
        sharp_amount=0.3,
        sharp_threshold=0.02,
        post_denoise_radius=4,
        post_denoise_eps=0.005,
        raw_y_blend=0.2,
        raw_y_full_blend=0.2,
    )
    pipe.train()

    H, W = 64, 96
    raw = (torch.rand(H, W) * 4095.0).to(torch.uint16)

    out = pipe.forward_diff(raw)
    loss = out["y"].mean() + out["uv"].mean()
    loss.backward()

    # All trainable params except awb.max_gain (no gradient through clamp at
    # extreme — historically expected zero) should receive a finite gradient.
    expected_zero = {"awb.max_gain"}
    bad = []
    finite_count = 0
    nonzero_count = 0
    total = 0
    for name, p in pipe.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None:
            bad.append(f"{name}: grad is None")
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(f"{name}: grad has non-finite values")
            continue
        finite_count += 1
        nz = p.grad.abs().sum().item() > 0
        if nz:
            nonzero_count += 1
        elif name not in expected_zero:
            bad.append(f"{name}: zero gradient (unexpected)")

    print(f"     trainable parameters: {total}")
    print(f"     finite gradients:      {finite_count}/{total}")
    print(f"     non-zero gradients:    {nonzero_count}/{total}")

    if bad:
        for line in bad:
            print(f"     [issue] {line}")
        _fail("Pipeline gradient flow", f"{len(bad)} issue(s)")
    _ok("Pipeline gradient flow ok")


def test_dark_raw_green_gamma_backward():
    """
    Regression test for the RAW-green gamma branch used by RGB->YUV blending.

    A fully dark RAW patch can drive the extracted green guide to exactly
    zero. Applying ``pow(inv_gamma)`` directly at zero is numerically
    singular for the canonical ``inv_gamma < 1`` setting and used to poison
    backward with non-finite gradients.
    """
    print("[7b] RAW-green gamma path on dark input")
    config = _load_config()

    pipe = ISPPipeline(
        config,
        device="cpu",
        denoise_eps=1e-12,
        ltm_a=0.5,
        ltm_detail_gain=30.0,
        ltm_detail_threshold=0.35,
        hist_target_mean=0.445,
        hist_target_std=0.162,
        post_denoise_radius=4,
        post_denoise_eps=0.001,
        raw_y_blend=0.2,
        raw_y_full_blend=0.4,
        sharp_amount=0.3,
    )
    pipe.train()

    raw_dark = torch.zeros(64, 96, dtype=torch.uint16)
    out = pipe.forward_diff(raw_dark)
    loss = out["y"].mean() + out["uv"].mean()
    loss.backward()

    check_names = (
        "denoise.log_eps",
        "gamma.inv_gamma",
        "rgb2yuv.raw_y_blend",
        "rgb2yuv.raw_y_full_blend",
    )
    params = dict(pipe.named_parameters())
    bad = []
    for name in check_names:
        grad = params[name].grad
        if grad is None:
            bad.append(f"{name}: grad is None")
        elif not torch.isfinite(grad).all():
            bad.append(f"{name}: grad has non-finite values")

    if bad:
        for line in bad:
            print(f"     [issue] {line}")
        _fail("Dark RAW-green backward", f"{len(bad)} issue(s)")
    _ok("Dark RAW-green backward stays finite")


def test_project_trainable_params():
    """
    Train-time projection should pull scalar ISP knobs back into valid ranges.
    """
    print("[7c] Trainable ISP projection")
    config = _load_config()
    pipe = ISPPipeline(config, device="cpu")

    with torch.no_grad():
        pipe.gamma.inv_gamma.fill_(-0.25)
        pipe.awb.max_gain.fill_(0.5)
        pipe.awb.lum_mask_low.fill_(1.2)
        pipe.awb.lum_mask_high.fill_(-0.3)
        pipe.ltm.a.fill_(-0.1)
        pipe.ltm.target_mean.fill_(1.4)
        pipe.ltm.detail_gain.fill_(-2.0)
        pipe.ltm.detail_threshold.fill_(-0.2)
        pipe.hist_norm.target_mean.fill_(-0.4)
        pipe.hist_norm.target_std.fill_(1.6)
        pipe.sharpening.amount.fill_(-0.1)
        pipe.sharpening.threshold.fill_(-0.2)
        pipe.rgb2yuv.raw_y_blend.fill_(1.3)
        pipe.rgb2yuv.raw_y_full_blend.fill_(-0.4)

    pipe.project_trainable_params_()

    checks = [
        ("gamma.inv_gamma > 0", float(pipe.gamma.inv_gamma.item()) > 0.0),
        ("awb.max_gain >= 1", float(pipe.awb.max_gain.item()) >= 1.0),
        ("awb low in [0,1]", 0.0 <= float(pipe.awb.lum_mask_low.item()) <= 1.0),
        ("awb high in [0,1]", 0.0 <= float(pipe.awb.lum_mask_high.item()) <= 1.0),
        (
            "awb low <= high",
            float(pipe.awb.lum_mask_low.item()) <= float(pipe.awb.lum_mask_high.item()),
        ),
        ("ltm.a > 0", float(pipe.ltm.a.item()) > 0.0),
        ("ltm.target_mean in [0,1]", 0.0 <= float(pipe.ltm.target_mean.item()) <= 1.0),
        ("ltm.detail_gain >= 0", float(pipe.ltm.detail_gain.item()) >= 0.0),
        ("ltm.detail_threshold >= 0", float(pipe.ltm.detail_threshold.item()) >= 0.0),
        ("hist.target_mean in [0,1]", 0.0 <= float(pipe.hist_norm.target_mean.item()) <= 1.0),
        ("hist.target_std in [0,1]", 0.0 <= float(pipe.hist_norm.target_std.item()) <= 1.0),
        ("sharp.amount >= 0", float(pipe.sharpening.amount.item()) >= 0.0),
        ("sharp.threshold >= 0", float(pipe.sharpening.threshold.item()) >= 0.0),
        ("raw_y_blend in [0,1]", 0.0 <= float(pipe.rgb2yuv.raw_y_blend.item()) <= 1.0),
        ("raw_y_full_blend in [0,1]", 0.0 <= float(pipe.rgb2yuv.raw_y_full_blend.item()) <= 1.0),
    ]
    bad = [name for name, ok in checks if not ok]
    if bad:
        for name in bad:
            print(f"     [issue] {name}")
        _fail("Trainable ISP projection", f"{len(bad)} issue(s)")
    _ok("Trainable ISP projection clamps all bounded knobs")


def test_scene_switch_preserves_trainables():
    """
    Verify that ``_apply_scene_params_preserving_trainables`` does NOT
    clobber any trainable nn.Parameter when switching scenes.

    The diploma-relevant invariant: ALL trainable knobs are shared across
    scenes and accumulate gradients across batches. Only structural
    (kernel-size / radius / downsample) knobs may switch per scene.
    """
    from isp.training.training_utils import (
        TRAINABLE_ISP_PARAM_KEYS,
        _apply_scene_params_preserving_trainables,
    )

    print("[8] Scene-switch preserves all trainable nn.Parameter tensors")
    config = _load_config()

    pipe = ISPPipeline(config, device="cpu")
    pipe.train()

    # Snapshot every trainable nn.Parameter before scene switch.
    before = {name: p.detach().clone() for name, p in pipe.named_parameters()}

    # Simulate a 'night' scene override that includes BOTH structural
    # (radii, downsample) and trainable knobs. The function under test must
    # filter out all trainable ones so they keep their pre-switch values.
    night_full_overrides = {
        # structural (these may change)
        "denoise_radius": 3,
        "ltm_radius": 12,
        "ltm_downsample": 0.75,
        "post_denoise_radius": 6,
        "raw_y_blur_radius": 16,
        "sharp_radius": 1.5,
        # trainable (these MUST be filtered out and not applied)
        "denoise_eps": 1e-3,
        "ltm_a": 0.1,
        "ltm_b": 0.05,
        "ltm_eps": 1e-2,
        "ltm_target_mean": 0.4,
        "ltm_detail_gain": 8.0,
        "ltm_detail_threshold": 0.4,
        "hist_target_mean": 0.6,
        "hist_target_std": 0.2,
        "post_denoise_eps": 1e-2,
        "sharp_amount": 0.99,
        "sharp_threshold": 0.5,
        "raw_y_blend": 0.5,
        "raw_y_full_blend": 0.6,
        "awb_max_gain": 8.0,
        "awb_lum_mask_low": 0.1,
        "awb_lum_mask_high": 0.9,
    }
    _apply_scene_params_preserving_trainables(pipe, night_full_overrides)

    # All trainable nn.Parameter tensors must be byte-identical to before.
    after = {name: p.detach().clone() for name, p in pipe.named_parameters()}
    diffs = []
    for name in before:
        if not torch.equal(before[name], after[name]):
            diffs.append(name)
    if diffs:
        for d in diffs:
            print(f"     [issue] {d} changed after scene switch")
        _fail("Scene switch preserves trainables", f"{len(diffs)} param(s) clobbered")

    n_trainable = sum(1 for _ in pipe.named_parameters())
    print(f"     all {n_trainable} trainable params survived the scene switch")
    _ok("Scene switch preserves all trainable nn.Parameters")

    # And the structural changes did take effect.
    if pipe.denoise.radius != 3:
        _fail("Scene switch structural", f"denoise.radius={pipe.denoise.radius} (expected 3)")
    if pipe.ltm.radius != 12:
        _fail("Scene switch structural", f"ltm.radius={pipe.ltm.radius} (expected 12)")
    if abs(pipe.ltm.downsample_factor - 0.75) > 1e-9:
        _fail("Scene switch structural", f"ltm.downsample={pipe.ltm.downsample_factor}")
    if pipe.post_denoise.radius != 6:
        _fail("Scene switch structural", f"post_denoise.radius={pipe.post_denoise.radius}")
    if pipe.rgb2yuv.raw_y_blur_radius != 16:
        _fail("Scene switch structural", f"raw_y_blur_radius={pipe.rgb2yuv.raw_y_blur_radius}")
    _ok("Scene switch applies structural knobs (radii, downsample)")

    # Sanity check: TRAINABLE_ISP_PARAM_KEYS covers every nn.Parameter.
    # Mapping from set-key to nn.Parameter attribute used by ISPPipeline.set_params.
    key_to_param = {
        "ccm_matrix": pipe.ccm.ccm,
        "gamma": pipe.gamma.inv_gamma,
        "awb_max_gain": pipe.awb.max_gain,
        "awb_lum_mask_low": pipe.awb.lum_mask_low,
        "awb_lum_mask_high": pipe.awb.lum_mask_high,
        "denoise_eps": pipe.denoise.log_eps,
        "ltm_a": pipe.ltm.a,
        "ltm_b": pipe.ltm.b,
        "ltm_eps": pipe.ltm.log_eps,
        "ltm_target_mean": pipe.ltm.target_mean,
        "ltm_detail_gain": pipe.ltm.detail_gain,
        "ltm_detail_threshold": pipe.ltm.detail_threshold,
        "hist_target_mean": pipe.hist_norm.target_mean,
        "hist_target_std": pipe.hist_norm.target_std,
        "sharp_amount": pipe.sharpening.amount,
        "sharp_threshold": pipe.sharpening.threshold,
        "raw_y_blend": pipe.rgb2yuv.raw_y_blend,
        "raw_y_full_blend": pipe.rgb2yuv.raw_y_full_blend,
        "post_denoise_eps": pipe.post_denoise.log_eps,
    }
    for key in TRAINABLE_ISP_PARAM_KEYS:
        if key not in key_to_param:
            _fail("TRAINABLE_ISP_PARAM_KEYS coverage", f"{key} has no mapped nn.Parameter")
    if set(key_to_param.keys()) != set(TRAINABLE_ISP_PARAM_KEYS):
        missing = set(key_to_param.keys()) - set(TRAINABLE_ISP_PARAM_KEYS)
        extra = set(TRAINABLE_ISP_PARAM_KEYS) - set(key_to_param.keys())
        _fail(
            "TRAINABLE_ISP_PARAM_KEYS coverage", f"missing in set: {missing}; extra in set: {extra}"
        )
    _ok(f"TRAINABLE_ISP_PARAM_KEYS covers all {len(TRAINABLE_ISP_PARAM_KEYS)} trainable knobs")


def test_build_scene_isp_with_learned():
    """
    Verify that ``build_scene_isp_with_learned`` transfers EVERY trainable
    knob from the learned snapshot into the per-scene eval ISP. A regression
    on this is silent (per-scene eval would use stale ISP_PARAMS values for
    the 17 non-CCM/gamma trainables) and biases composite reporting.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    run_e2e_train = importlib.import_module("run_e2e_train")

    print("[9] build_scene_isp_with_learned transfers every learned knob")
    config = _load_config()

    # Build a training ISP, perturb every trainable knob away from the
    # day baseline, snapshot it.
    train_isp = ISPPipeline(config, device="cpu", **run_e2e_train.ISP_PARAMS["day"])
    with torch.no_grad():
        train_isp.ccm.ccm.add_(0.05)
        train_isp.gamma.inv_gamma.fill_(1.0 / 2.4)
        train_isp.awb.max_gain.fill_(5.5)
        train_isp.awb.lum_mask_low.fill_(0.07)
        train_isp.awb.lum_mask_high.fill_(0.93)
        train_isp.denoise.set_eps(2.5e-4)
        train_isp.ltm.a.fill_(0.42)
        train_isp.ltm.b.fill_(0.03)
        train_isp.ltm.set_eps(7e-4)
        train_isp.ltm.target_mean.fill_(0.33)
        train_isp.ltm.detail_gain.fill_(17.5)
        train_isp.ltm.detail_threshold.fill_(0.28)
        train_isp.hist_norm.target_mean.fill_(0.41)
        train_isp.hist_norm.target_std.fill_(0.13)
        train_isp.sharpening.amount.fill_(0.61)
        train_isp.sharpening.threshold.fill_(0.018)
        train_isp.rgb2yuv.raw_y_blend.fill_(0.27)
        train_isp.rgb2yuv.raw_y_full_blend.fill_(0.55)
        train_isp.post_denoise.set_eps(1.7e-3)

    learned = run_e2e_train.snapshot_isp_params(train_isp)

    # Build the per-scene eval ISP under the night baseline (so structural
    # bits like radii change), then check every learned trainable shows up.
    night_isp = run_e2e_train.build_scene_isp_with_learned(
        config,
        device="cpu",
        scene_name="night",
        learned=learned,
    )

    def _get(attr_path):
        obj = night_isp
        for name in attr_path:
            obj = getattr(obj, name)
        return float(obj.detach().item())

    cases = [
        ("gamma", ("gamma", "inv_gamma"), 1.0 / learned["gamma"]),
        ("awb_max_gain", ("awb", "max_gain"), learned["awb_max_gain"]),
        ("awb_lum_mask_low", ("awb", "lum_mask_low"), learned["awb_lum_mask_low"]),
        ("awb_lum_mask_high", ("awb", "lum_mask_high"), learned["awb_lum_mask_high"]),
        ("denoise_eps", ("denoise", "eps"), learned["denoise_eps"]),
        ("ltm_a", ("ltm", "a"), learned["ltm_a"]),
        ("ltm_b", ("ltm", "b"), learned["ltm_b"]),
        ("ltm_eps", ("ltm", "eps"), learned["ltm_eps"]),
        ("ltm_target_mean", ("ltm", "target_mean"), learned["ltm_target_mean"]),
        ("ltm_detail_gain", ("ltm", "detail_gain"), learned["ltm_detail_gain"]),
        ("ltm_detail_threshold", ("ltm", "detail_threshold"), learned["ltm_detail_threshold"]),
        ("hist_target_mean", ("hist_norm", "target_mean"), learned["hist_target_mean"]),
        ("hist_target_std", ("hist_norm", "target_std"), learned["hist_target_std"]),
        ("sharp_amount", ("sharpening", "amount"), learned["sharp_amount"]),
        ("sharp_threshold", ("sharpening", "threshold"), learned["sharp_threshold"]),
        ("raw_y_blend", ("rgb2yuv", "raw_y_blend"), learned["raw_y_blend"]),
        ("raw_y_full_blend", ("rgb2yuv", "raw_y_full_blend"), learned["raw_y_full_blend"]),
        ("post_denoise_eps", ("post_denoise", "eps"), learned["post_denoise_eps"]),
    ]
    for name, path, expected in cases:
        got = _get(path)
        if abs(got - expected) > 1e-6:
            _fail("build_scene_isp_with_learned", f"{name}: got {got}, expected {expected}")
    _ok(f"build_scene_isp_with_learned transfers all {len(cases) + 1} learned knobs")

    # CCM matrix transfer (T-conjugated).
    learned_ccm = torch.tensor(learned["ccm_matrix"], dtype=torch.float32)
    night_ccm = night_isp.ccm.ccm.detach().T
    if (learned_ccm - night_ccm).abs().max().item() > 1e-6:
        _fail("build_scene_isp_with_learned", "CCM not transferred")
    _ok("build_scene_isp_with_learned transfers CCM correctly (T-conjugated)")

    # Regression: slightly out-of-range learned scalars should be sanitized
    # instead of crashing per-scene eval construction.
    drifted = dict(learned)
    drifted["awb_max_gain"] = 0.25
    drifted["awb_lum_mask_low"] = -0.02
    drifted["awb_lum_mask_high"] = 1.03
    drifted["ltm_a"] = -0.4
    drifted["hist_target_mean"] = 1.2
    drifted["hist_target_std"] = -0.1
    drifted["sharp_amount"] = -0.2
    drifted["raw_y_blend"] = 1.4
    drifted["raw_y_full_blend"] = -0.3

    drifted_isp = run_e2e_train.build_scene_isp_with_learned(
        config,
        device="cpu",
        scene_name="day",
        learned=drifted,
    )
    if float(drifted_isp.awb.max_gain.item()) < 1.0:
        _fail("build_scene_isp_with_learned sanitize", "awb_max_gain not clamped")
    if not (
        0.0
        <= float(drifted_isp.awb.lum_mask_low.item())
        <= float(drifted_isp.awb.lum_mask_high.item())
        <= 1.0
    ):
        _fail("build_scene_isp_with_learned sanitize", "AWB mask bounds/order not clamped")
    if float(drifted_isp.ltm.a.item()) <= 0.0:
        _fail("build_scene_isp_with_learned sanitize", "ltm_a not clamped positive")
    if not 0.0 <= float(drifted_isp.hist_norm.target_mean.item()) <= 1.0:
        _fail("build_scene_isp_with_learned sanitize", "hist_target_mean not clamped")
    if not 0.0 <= float(drifted_isp.hist_norm.target_std.item()) <= 1.0:
        _fail("build_scene_isp_with_learned sanitize", "hist_target_std not clamped")
    if float(drifted_isp.sharpening.amount.item()) < 0.0:
        _fail("build_scene_isp_with_learned sanitize", "sharp_amount not clamped")
    if not 0.0 <= float(drifted_isp.rgb2yuv.raw_y_blend.item()) <= 1.0:
        _fail("build_scene_isp_with_learned sanitize", "raw_y_blend not clamped")
    if not 0.0 <= float(drifted_isp.rgb2yuv.raw_y_full_blend.item()) <= 1.0:
        _fail("build_scene_isp_with_learned sanitize", "raw_y_full_blend not clamped")
    _ok("build_scene_isp_with_learned clamps drifted learned scalars")


def main():
    print("=" * 60)
    print("ISP optimization smoke test")
    print("=" * 60)
    test_sharpening_separable_blur()
    test_hist_norm_fast_path()
    test_rgb2yuv_paths()
    test_post_gamma_denoise_box()
    test_ltm_box_and_forward()
    test_validate_params()
    test_pipeline_gradients()
    test_dark_raw_green_gamma_backward()
    test_project_trainable_params()
    test_scene_switch_preserves_trainables()
    test_build_scene_isp_with_learned()
    print("=" * 60)
    print("All smoke tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
