#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class AFNOSpectralAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 spatial_size,
                 n_blocks=8,
                 modes=None,
                 block_size=None,
                 shrinkage=True):
        super().__init__()
        assert embed_dim > 0
        if isinstance(spatial_size, int):
            H = W = spatial_size
        else:
            H, W = spatial_size
        self.H = H
        self.W = W
        self.N = H * W
        self.embed_dim = embed_dim

        # Determine blocks and block_size
        if block_size is None:
            block_size = max(1, embed_dim // n_blocks)
            n_blocks = math.ceil(embed_dim / block_size)

        self.n_blocks = n_blocks
        self.block_size = block_size

        total_block_ch = self.n_blocks * self.block_size
        if total_block_ch != embed_dim:
            self._pad_channels = total_block_ch - embed_dim
        else:
            self._pad_channels = 0

        freq_bins = W // 2 + 1
        if modes is None:
            self.modes = max(1, freq_bins // 4)
        else:
            self.modes = min(freq_bins, modes)

        self.block_weights = nn.ParameterList()
        for _ in range(self.n_blocks):
            w = nn.Parameter(torch.randn(self.block_size, self.block_size) * 0.02)
            self.block_weights.append(w)

        self.block_bias = nn.ParameterList()
        for _ in range(self.n_blocks):
            b = nn.Parameter(torch.zeros(self.block_size))
            self.block_bias.append(b)

        if shrinkage:
            self.use_shrinkage = True
            self.shrinkage_gates = nn.Parameter(torch.ones(self.n_blocks) * 0.5)
        else:
            self.use_shrinkage = False

        self.out_proj = nn.Linear(total_block_ch, embed_dim)

        self._rescale = nn.Parameter(torch.tensor(1e-1))

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.N, f"Expected N={self.N} (H*W) but got {N}"
        device = x.device
        dtype = x.dtype
        H, W = self.H, self.W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous() 

        # If padding channels are required, pad on channel dim
        if self._pad_channels > 0:
            pad = torch.zeros(B, self._pad_channels, H, W, device=device, dtype=dtype)
            x = torch.cat([x, pad], dim=1)  # (B, total_block_ch, H, W)

        # split channels into blocks: (B, n_blocks, block_size, H, W)
        B, total_ch, H, W = x.shape
        x = x.view(B, self.n_blocks, self.block_size, H, W)

        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')

        m = self.modes
        x_fft_trunc = x_fft[..., :m]  

        out_fft_trunc = torch.zeros_like(x_fft_trunc)  
        for b_idx in range(self.n_blocks):
            blk_coeff = x_fft_trunc[:, b_idx]  
            blk_perm = blk_coeff.permute(0, 2, 3, 1).contiguous()  

            real = blk_perm.real
            imag = blk_perm.imag

            BHm = real.shape[0] * real.shape[1] * real.shape[2]
            real_flat = real.reshape(BHm, self.block_size)
            imag_flat = imag.reshape(BHm, self.block_size)

            W_blk = self.block_weights[b_idx]  
            b_blk = self.block_bias[b_idx]     
            out_real_flat = real_flat.matmul(W_blk) + b_blk.unsqueeze(0)
            out_imag_flat = imag_flat.matmul(W_blk) + b_blk.unsqueeze(0)

            out_real = out_real_flat.view(real.shape)
            out_imag = out_imag_flat.view(imag.shape)

            if self.use_shrinkage:
                gate = torch.sigmoid(self.shrinkage_gates[b_idx])
                out_real = out_real * gate
                out_imag = out_imag * gate

            out_blk_perm = torch.complex(out_real, out_imag)
            out_blk = out_blk_perm.permute(0, 3, 1, 2).contiguous()

            out_fft_trunc[:, b_idx] = out_blk

        x_fft_new = x_fft.clone()
        x_fft_new[..., :m] = out_fft_trunc
        x_ifft = torch.fft.irfft2(x_fft_new, s=(H, W), dim=(-2, -1), norm='ortho')

        x_merged = x_ifft.view(B, self.n_blocks * self.block_size, H, W)
        if self._pad_channels > 0:
            x_merged = x_merged[:, :self.embed_dim, :, :].contiguous()

        x_out = x_merged.permute(0, 2, 3, 1).contiguous().view(B, H * W, self.embed_dim)
        x_proj = self.out_proj(x_out) * self._rescale

        return x_proj + x_out


# In[ ]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, spatial_size, mlp_ratio=4.0, attn_blocks=8, attn_modes=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AFNOSpectralAttention(embed_dim=embed_dim,
                                          spatial_size=spatial_size,
                                          n_blocks=attn_blocks,
                                          modes=attn_modes)
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


# In[ ]:


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Downsample(nn.Module):
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.convT(x)
        return x.flatten(2).transpose(1, 2)


# In[ ]:


class FourierTransformerUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=1, out_chans=1,
                 embed_dims=[64, 128, 256], depth=[2, 2, 2],
                 attn_blocks=[4, 8, 8], attn_modes=None):
        super().__init__()

        if isinstance(attn_blocks, int):
            attn_blocks = [attn_blocks] * len(embed_dims)
        if attn_modes is None:
            attn_modes = [None] * len(embed_dims)
        elif isinstance(attn_modes, int):
            attn_modes = [attn_modes] * len(embed_dims)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])

        H, W = img_size // patch_size, img_size // patch_size

        # Encoder
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(embed_dims)):
            stage_blocks = nn.ModuleList([
                TransformerBlock(embed_dims[i], (H, W),
                                 mlp_ratio=4.0,
                                 attn_blocks=attn_blocks[i],
                                 attn_modes=attn_modes[i]) for _ in range(depth[i])
            ])
            self.encoder_stages.append(stage_blocks)
            if i < len(embed_dims) - 1:
                self.downsamples.append(Downsample(embed_dims[i], embed_dims[i+1]))
                H, W = H // 2, W // 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            TransformerBlock(embed_dims[-1], (H, W),
                             mlp_ratio=4.0,
                             attn_blocks=attn_blocks[-1],
                             attn_modes=attn_modes[-1]) for _ in range(depth[-1])
        ])

        # Decoder
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoder_projs = nn.ModuleList()
        for i in reversed(range(len(embed_dims) - 1)):
            H, W = H * 2, W * 2
            self.upsamples.append(Upsample(embed_dims[i+1], embed_dims[i]))
            self.decoder_projs.append(nn.Linear(embed_dims[i] * 2, embed_dims[i]))
            stage_blocks = nn.ModuleList([
                TransformerBlock(embed_dims[i], (H, W),
                                 mlp_ratio=4.0,
                                 attn_blocks=attn_blocks[i],
                                 attn_modes=attn_modes[i]) for _ in range(depth[i])
            ])
            self.decoder_stages.append(stage_blocks)

        # Final projection
        self.final_proj = nn.Conv2d(embed_dims[0], out_chans * patch_size**2, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x):
        skip_connections = []
        x = self.patch_embed(x)

        # Encoder
        for i, stage in enumerate(self.encoder_stages):
            for block in stage:
                x = block(x)
            if i < len(self.encoder_stages) - 1:
                skip_connections.append(x)
                x = self.downsamples[i](x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)

        # Decoder
        for i, stage in enumerate(self.decoder_stages):
            x = self.upsamples[i](x)
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=2)
            x = self.decoder_projs[i](x)
            for block in stage:
                x = block(x)

        # Final projection and reshaping
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.final_proj(x)
        x = self.pixel_shuffle(x)
        return x

