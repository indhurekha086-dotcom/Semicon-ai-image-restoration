import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class MSRUNet(nn.Module):
    """Compact multi-scale residual U-Net for image restoration."""
    def __init__(self, in_channels=1, out_channels=1, base=32, num_res=4, scale=4):
        super().__init__()
        self.scale = scale
        self.in_proj = nn.Conv2d(in_channels, base, 3, padding=1)

        self.enc1 = ConvBlock(base, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.enc3 = ConvBlock(base * 4, base * 4)

        self.res = nn.Sequential(
            *[ResidualBlock(base * 4) for _ in range(num_res)]
        )

        self.up2 = nn.Conv2d(base * 4, base * 2, 3, padding=1)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_proj = nn.Conv2d(base, out_channels, 3, padding=1)

    def forward(self, x, output_size=None):
        if output_size is not None and tuple(x.shape[-2:]) != tuple(output_size):
            x_up = F.interpolate(
                x,
                size=output_size,
                mode="bicubic",
                align_corners=False
            )
        elif self.scale != 1:
            x_up = F.interpolate(
                x,
                scale_factor=self.scale,
                mode="bicubic",
                align_corners=False
            )
        else:
            x_up = x

        x0 = self.in_proj(x_up)
        e1 = self.enc1(x0)

        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.res(e3)

        d2 = F.interpolate(
            b,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        d2 = self.up2(d2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(
            d2,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        d1 = self.up1(d1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        residual = self.out_proj(d1)

        baseline = x_up
        if baseline.shape[1] != residual.shape[1]:
            baseline = baseline[:, :1]

        return baseline + residual
