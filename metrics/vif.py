import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# PyTorch Implementation of Noise‐Corrected Visual Information Fidelity (VIF)
#
# Based on:
#   Sheikh, H. R., & Bovik, A. C. (2006).
#   "Image Information and Visual Quality."
#   IEEE Transactions on Image Processing, 15(2), 430–444.
# -----------------------------------------------------------------------------


def _gaussian_kernel(window_size: int, sigma: float, device=None) -> torch.Tensor:
    """
    Create a 2D Gaussian smoothing kernel for local statistics.

    Args:
      window_size: side length of the square kernel (odd integer)
      sigma:       standard deviation of the Gaussian
      device:      torch device for tensor allocation

    Returns:
      kernel:      shape (1,1,window_size,window_size), normalized to sum=1
    """
    coords = torch.arange(window_size, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel2d = g[:, None] * g[None, :]
    return kernel2d.unsqueeze(0).unsqueeze(0)  # shape (1,1,ws,ws)


def local_median(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Compute the local median of tensor x using a sliding window of radius r.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        r (int): Radius of the local window.

    Returns:
        torch.Tensor: Local median map of shape (B, C, H, W).
    """
    k = 2 * r + 1
    # Reflect padding to handle borders
    x_padded = F.pad(x, (r, r, r, r), mode="reflect")
    # Extract sliding windows: result shape (B, C, H, W, k, k)
    patches = x_padded.unfold(2, k, 1).unfold(3, k, 1)
    # Reshape to (B, C, H, W, k*k)
    B, C, H, W, _, _ = patches.shape
    patches = patches.contiguous().view(B, C, H, W, k * k)
    # Compute median along the last dimension
    med, _ = patches.median(dim=-1)
    return med


def estimate_noise_variance(
    noisy: torch.Tensor, window_size: int = 7, eps: float = 1e-6
) -> torch.Tensor:
    """
    Estimate per-pixel noise variance using robust median-based local statistics.

    Args:
        noisy (torch.Tensor): Noisy input tensor of shape (B, C, H, W).
        window_size (int): Radius of the local window.
        eps (float): Small constant to avoid zero variance.

    Returns:
        torch.Tensor: Estimated noise variance map of shape (B, C, H, W).
    """
    # 1. Local median (signal estimate)
    med = local_median(noisy, window_size)
    # 2. Absolute deviations from the median
    abs_dev = (noisy - med).abs()
    # 3. Local MAD
    mad = F.avg_pool2d(abs_dev, window_size, stride=1, padding=window_size // 2)
    # 4. Convert MAD to standard deviation estimate
    sigma = 1.4826 * mad
    # 5. Variance map and clamp to avoid zeros
    var_map = sigma.pow(2).clamp(min=eps)
    return var_map


def vif(
    ref: torch.Tensor,
    dist: torch.Tensor,
    num_scales: int = 4,
    win_size: int = 5,
    win_sigma: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute the Visual Information Fidelity (VIF) index between two images,
    automatically correcting for per-pixel noise estimated from local variation.

    VIF is defined as the ratio of:
      - Numerator: mutual information bits between ref → dist under noise
      - Denominator: mutual information bits between ref → HVS noise only

    See Sheikh & Bovik (2006) for full derivation.

    Args:
      ref:        Bx1xHxW reference (pristine) image
      dist:       Bx1xHxW distorted image
      num_scales: number of wavelet scales (default 4)
      win_size:   Gaussian window size for local stats (default 5)
      win_sigma:  Gaussian sigma for window (default 1.0)
      eps:        small constant for numerical stability

    Returns:
      vif_score:  B-length tensor, VIF in [0,1] per batch element
    """
    device = ref.device

    # Prepare Gaussian window for local moment computation
    kernel = _gaussian_kernel(win_size, win_sigma, device)

    num_all = torch.zeros(ref.shape[0], device=device)  # accumulate numerator
    den_all = torch.zeros_like(num_all)  # accumulate denominator

    x, y = ref, dist

    # Multi-scale loop: mimic steerable-pyramid without explicit filters
    for _scale in range(num_scales):
        x = (x - x.mean(dim=(2, 3))) / (x.std(dim=(2, 3)) + eps)
        y = (y - y.mean(dim=(2, 3))) / (y.std(dim=(2, 3)) + eps)

        # Local means via Gaussian smoothing
        mu_x = F.conv2d(x, kernel, padding=win_size // 2)
        mu_y = F.conv2d(y, kernel, padding=win_size // 2)

        # Local second moments
        mu_x2 = F.conv2d(x * x, kernel, padding=win_size // 2)
        mu_y2 = F.conv2d(y * y, kernel, padding=win_size // 2)
        mu_xy = F.conv2d(x * y, kernel, padding=win_size // 2)

        # Variance and covariance (per Sheikh & Bovik)
        sigma_x2 = (mu_x2 - mu_x * mu_x).clamp(min=0.0)
        sigma_y2 = (mu_y2 - mu_y * mu_y).clamp(min=0.0)
        sigma_xy = mu_xy - mu_x * mu_y

        # Gain factor and residual variance
        g = sigma_xy / (sigma_x2 + eps)
        sigma_v2 = (sigma_y2 - g * sigma_xy).clamp(min=0.0)

        # Estimate per-pixel perceptual noise variances from local total variation
        sn_ref = estimate_noise_variance(x, window_size=7)
        sn_dist = estimate_noise_variance(y, window_size=7)

        # Denominator: bits ref -> HVS noisy channel only
        den = 0.5 * torch.log2(1 + sigma_x2 / (sn_ref + eps))
        # Numerator: bits surviving distortion + HVS noise
        num = 0.5 * torch.log2(1 + (g * g * sigma_x2) / (sigma_v2 + sn_dist + eps))

        # Sum over spatial dims, accumulate per-batch
        num_all += num.flatten(1).sum(dim=1)
        den_all += den.flatten(1).sum(dim=1)

        # Downsample images for next scale
        x = mu_x[:, :, ::2, ::2]
        y = mu_y[:, :, ::2, ::2]

    # Final VIF ratio per batch element, clipped to [0,1]
    vif_score = (num_all / (den_all + eps)).clamp(0.0, 1.0)
    return vif_score


def interpolate_green_even(cfa: torch.Tensor, pattern: str) -> torch.Tensor:
    """
    Interpolate the Green channel from a single-channel CFA tensor
    at half-pixel shift (+0.5) using an even 2x2 box filter.

    Args:
      cfa:     Bx1xHxW CFA tensor (mosaic of R,G,B)
      pattern: 'RGGB' or 'GRBG'

    Returns:
      g_interp: Bx1xHxW interpolated Green channel
    """
    B, C, H, W = cfa.shape
    if C != 1:
        raise ValueError("CFA tensor must be single-channel (Bx1xHxW)")
    mask = torch.zeros((B, 1, H, W), device=cfa.device)
    pattern = pattern.upper()
    if pattern == "RGGB" or pattern == "BGGR":
        mask[:, :, 0::2, 1::2] = 1.0
        mask[:, :, 1::2, 0::2] = 1.0
    elif pattern == "GRBG" or pattern == "GBRG":
        mask[:, :, 0::2, 0::2] = 1.0
        mask[:, :, 1::2, 1::2] = 1.0
    else:
        raise ValueError(f"Unsupported CFA pattern: {pattern}")
    g_known = cfa * mask
    kernel = torch.ones((1, 1, 2, 2), device=cfa.device) / 2.0
    pad = (0, 1, 0, 1)
    g_padded = F.pad(g_known, pad=pad, mode="replicate")
    g_interp = F.conv2d(g_padded, kernel)
    return g_interp[:, :, :H, :W]


def interpolate_green_odd(cfa: torch.Tensor, pattern: str) -> torch.Tensor:
    """
    Interpolate the Green channel from a single-channel CFA tensor

    Args:
      cfa:     Bx1xHxW CFA tensor (mosaic of R,G,B)
      pattern: 'RGGB' or 'GRBG'

    Returns:
      g_interp: Bx1xHxW interpolated Green channel
    """
    B, C, H, W = cfa.shape
    if C != 1:
        raise ValueError("CFA tensor must be single-channel (Bx1xHxW)")
    mask = torch.zeros((B, 1, H, W), device=cfa.device)
    pattern = pattern.upper()
    if pattern == "RGGB" or pattern == "BGGR":
        mask[:, :, 0::2, 1::2] = 1.0
        mask[:, :, 1::2, 0::2] = 1.0
    elif pattern == "GRBG" or pattern == "GBRG":
        mask[:, :, 0::2, 0::2] = 1.0
        mask[:, :, 1::2, 1::2] = 1.0
    else:
        raise ValueError(f"Unsupported CFA pattern: {pattern}")
    g_known = cfa * mask
    kernel = torch.ones((1, 1, 3, 3), device=cfa.device) / 4.0
    pad = (1, 1, 1, 1)
    g_padded = F.pad(g_known, pad=pad, mode="replicate")
    g_interp = F.conv2d(g_padded, kernel)
    return g_known + g_interp[:, :, :H, :W] * (1 - mask)


def vif_cfa_to_y(
    cfa: torch.Tensor,
    y: torch.Tensor,
    pattern: str = "RGGB",
    num_scales: int = 4,
    win_size: int = 5,
    win_sigma: float = 1.0,
    eps: float = 1e-10,
    even: bool = False,
) -> torch.Tensor:
    """
    Compute VIF between the interpolated Green channel of a CFA tensor
    and the luma (Y) channel of a YUV tensor.

    Args:
      cfa:         Bx1xHxW raw CFA tensor (uint16)
      y:           Bx1xHxW Y tensor
      pattern:     CFA pattern: 'RGGB' or 'GRBG'

    Returns:
      vif_score:   B-length tensor of VIF scores
    """
    cfa = cfa.to(torch.float32) / (2**16 - 1)
    g_interp = interpolate_green_even(cfa, pattern) if even else interpolate_green_odd(cfa, pattern)
    y = y.to(torch.float32) / 255.0
    return vif(g_interp, y, num_scales=num_scales, win_size=win_size, win_sigma=win_sigma, eps=eps)
