#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn

class SmallDenoiser(nn.Module):
    """
    Simple residual CNN denoiser on the phase map.
    Input/Output: (B,1,H,W) phase in radians.
    """
    def __init__(self, n_ch=1, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, n_ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class HIOStep(nn.Module):
    """
    One HIO iteration for phase-only object:
    - meas_mag: measured Fourier magnitude (linear), shape (B,1,H,W)
    - psi: complex object field in real domain, shape (B,H,W), |psi|=1
    - support: binary mask in object domain, shape (H,W) or (1,H,W)
    """
    def __init__(self, beta=0.9):
        super().__init__()
        self.beta = beta

    def forward(self, psi, meas_mag, support):
        B, H, W = psi.shape

        # Forward FFT to far-field plane
        Fk = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(psi, dim=(-2, -1))
            ),
            dim=(-2, -1)
        )  # (B,H,W), complex

        # Enforce measured magnitude, keep phase
        phase = torch.angle(Fk)          # (B,H,W)
        mag = meas_mag.squeeze(1)       # (B,H,W)
        Fk_new = mag * torch.exp(1j * phase)

        # Back to object plane
        psi_new = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(Fk_new, dim=(-2, -1))
            ),
            dim=(-2, -1)
        )  # (B,H,W), complex

        # HIO update with support constraint
        supp = support.to(psi_new.real.dtype)  # (H,W) or (1,H,W)
        psi_updated = psi_new * supp + (psi - self.beta * psi_new) * (1.0 - supp)

        # Phase-only constraint: unit amplitude
        psi_updated = torch.exp(1j * torch.angle(psi_updated))

        return psi_updated


def create_centered_support(H, W, ratio):
    mask = torch.zeros(H, W)
    h = int(H * ratio)
    w = int(W * ratio)
    y0, x0 = (H - h) // 2, (W - w) // 2
    mask[y0:y0 + h, x0:x0 + w] = 1.0
    return mask  # (H,W)


class UnrolledHIOPhaseNet(nn.Module):
    """
    Physically-correct unrolled HIO with CNN denoising.
    - log_mag: ∈ [0,1], normalized
    - lin_mag: ∈ [0,1], linear Fourier magnitude
    Output phase constrained to [0, π].
    """
    def __init__(self, img_size=256, n_steps=8, beta=0.9, support_ratio=0.5, base_ch=32):
        super().__init__()
        self.img_size = img_size
        self.n_steps = n_steps

        H = W = img_size
        self.register_buffer("support", create_centered_support(H, W, support_ratio))

        # log-mag → initial phase ∈ [0, π]
        self.init_net = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.denoiser = SmallDenoiser(n_ch=1, base_ch=base_ch)
        self.hio_steps = nn.ModuleList([HIOStep(beta=beta) for _ in range(n_steps)])

    def forward(self, log_mag, lin_mag):
        meas_mag = lin_mag  # (B,1,H,W)

        # Initial phase prediction in [0, π]
        phase = self.init_net(log_mag) * math.pi  # (B,1,H,W) → [0, π]
        psi = torch.exp(1j * phase.squeeze(1))    # (B,H,W), complex

        support = self.support  # (H,W)

        for hio_step in self.hio_steps:
            psi = hio_step(psi, meas_mag, support)

            # Raw phase from complex field in [-π, π]
            phase = torch.angle(psi)

            # Map to [0, 2π)
            phase = (phase + 2 * math.pi) % (2 * math.pi)

            # Fold into [0, π]
            phase = torch.where(phase > math.pi, 2 * math.pi - phase, phase)

            # Denoise in phase domain
            phase = self.denoiser(phase.unsqueeze(1)).squeeze(1)

            # Enforce [0, π] after CNN
            phase = torch.clamp(phase, 0.0, math.pi)

            # Rebuild psi with unit amplitude
            psi = torch.exp(1j * phase)

        return phase.unsqueeze(1)  # (B,1,H,W), ∈ [0, π]
