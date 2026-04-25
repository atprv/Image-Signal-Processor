import torch
import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm.
    """

    def __init__(self, channels: int = 32, num_groups: int = 8):
        """
        Args:
            channels: Number of feature channels.
            num_groups: Number of GroupNorm groups.
        """
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if channels % num_groups != 0:
            raise ValueError(
                "channels must be divisible by num_groups, "
                f"got channels={channels}, num_groups={num_groups}"
            )

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [B, C, H, W]

        Returns:
            torch.Tensor: Tensor with shape [B, C, H, W]
        """
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.relu(out)

        return out


class ResidualCNN(nn.Module):
    """
    Small residual CNN for YUV residual prediction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        out_channels: int = 3,
        num_blocks: int = 5,
        num_groups: int = 8,
    ):
        """
        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden feature channels.
            out_channels: Number of output channels.
            num_blocks: Number of residual blocks.
            num_groups: Number of GroupNorm groups.
        """
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels must be divisible by num_groups, "
                f"got hidden_channels={hidden_channels}, num_groups={num_groups}"
            )

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[ResBlock(channels=hidden_channels, num_groups=num_groups) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True)

        self._init_tail_for_residual_start()

    def _init_tail_for_residual_start(self):
        """
        Initialize the tail so that the network starts close to identity,
        while gradients can still flow through the whole network.
        """
        nn.init.normal_(self.tail.weight, mean=0.0, std=1e-3)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape [B, 3, H, W] in YUV444 format.

        Returns:
            torch.Tensor: Residual tensor with shape [B, 3, H, W].
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels for YUV444, got {x.shape[1]}")

        out = self.head(x)
        out = self.body(out)
        out = self.tail(out)

        return out
