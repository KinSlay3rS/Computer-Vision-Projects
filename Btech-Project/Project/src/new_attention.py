#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, spatial_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if isinstance(spatial_size, int):
            H = W = spatial_size
        else:
            H, W = spatial_size

        # learnable complex multipliers for each frequency and head and channel
        # shape: (1, num_heads, H, W//2 + 1, head_dim)
        # dtype is complex to match torch.fft.rfft2 output
        self.register_parameter(
            'spectral_weight',
            nn.Parameter(torch.randn(1, num_heads, H, W // 2 + 1, self.head_dim, dtype=torch.cfloat) * 0.02 + 1.0)
        )

        # small real-valued bias in spatial domain after inverse transform
        self.spatial_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1, self.head_dim))

        # optional residual projection to mix head outputs back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, 'Input sequence length must be a perfect square'

        # reshape into (B, num_heads, H, W, head_dim)
        x = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x_img = x.view(B, self.num_heads, H, W, self.head_dim)

        # FFT over spatial dims -> complex tensor shape (B, num_heads, H, W//2+1, head_dim)
        x_fft = torch.fft.rfft2(x_img.float(), dim=(2, 3), norm='ortho')

        # ensure dtype alignment (x_fft is complex64 on most platforms)
        Wc = self.spectral_weight
        if x_fft.dtype != Wc.dtype:
            Wc = Wc.to(x_fft.dtype)

        # elementwise spectral multiplication
        x_fft = x_fft * Wc

        # inverse FFT to spatial domain
        x_ifft = torch.fft.irfft2(x_fft, s=(H, W), dim=(2, 3), norm='ortho')

        # add small learned bias in spatial domain
        # x_ifft: (B, num_heads, H, W, head_dim)
        x_ifft = x_ifft + self.spatial_bias

        # reshape back to (B, N, C)
        x = x_ifft.view(B, self.num_heads, N, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C)

        # final projection
        x = self.out_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, spatial_size, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FourierAttention(embed_dim, num_heads, spatial_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Downsample(nn.Module):
    """Downsamples by merging patches."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        return x.flatten(2).transpose(1, 2)


class Upsample(nn.Module):
    """Upsamples by expanding patches."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.convT(x)
        return x.flatten(2).transpose(1, 2)


class FourierTransformerUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=1, out_chans=1,
                 embed_dims=[64, 128, 256], num_heads=[2, 4, 8], depth=[2, 2, 2]):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])

        H, W = img_size // patch_size, img_size // patch_size

        # --- Encoder ---
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(embed_dims)):
            stage_blocks = nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i], (H, W)) for _ in range(depth[i])
            ])
            self.encoder_stages.append(stage_blocks)
            if i < len(embed_dims) - 1:
                self.downsamples.append(Downsample(embed_dims[i], embed_dims[i+1]))
                H, W = H // 2, W // 2

        # --- Bottleneck ---
        self.bottleneck = nn.ModuleList([
            TransformerBlock(embed_dims[-1], num_heads[-1], (H, W)) for _ in range(depth[-1])
        ])

        # --- Decoder ---
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoder_projs = nn.ModuleList()

        for i in reversed(range(len(embed_dims) - 1)):
            H, W = H * 2, W * 2
            self.upsamples.append(Upsample(embed_dims[i+1], embed_dims[i]))
            self.decoder_projs.append(nn.Linear(embed_dims[i] * 2, embed_dims[i]))
            stage_blocks = nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i], (H, W)) for _ in range(depth[i])
            ])
            self.decoder_stages.append(stage_blocks)

        # --- Final Projection Head ---
        self.final_proj = nn.Conv2d(embed_dims[0], out_chans * patch_size**2, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x):
        skip_connections = []
        x = self.patch_embed(x)

        # Encoder Path
        for i, stage in enumerate(self.encoder_stages):
            for block in stage:
                x = block(x)
            if i < len(self.encoder_stages) - 1:
                skip_connections.append(x)
                x = self.downsamples[i](x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)

        # Decoder Path
        for i, stage in enumerate(self.decoder_stages):
            x = self.upsamples[i](x)
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=2)
            x = self.decoder_projs[i](x)
            for block in stage:
                x = block(x)

        # Final Projection and Reshaping
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.final_proj(x)
        x = self.pixel_shuffle(x)

        return x


# In[ ]:




