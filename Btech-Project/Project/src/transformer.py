#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierAttention(nn.Module):
    """
    Implements Global Convolution through a learnable filter in the frequency domain.
    This version is robust to changes in spatial size.
    """
    def __init__(self, embed_dim, num_heads, spatial_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Ensure spatial_size is a tuple
        if isinstance(spatial_size, int):
            H = W = spatial_size
        else:
            H, W = spatial_size
        
        # Create a learnable complex-valued weight parameter with the correct shape
        # for this specific layer's input resolution.
        self.complex_weights = nn.Parameter(torch.randn(
            1, num_heads, H, W // 2 + 1, self.head_dim, dtype=torch.cfloat
        ))
        
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)

        x = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x_img = x.view(B, self.num_heads, H, W, self.head_dim)

        # Apply 2D Fourier Transform
        # We add a dimension for broadcasting with the complex weights
        x_fft = torch.fft.rfft2(x_img.float(), dim=(2, 3), norm='ortho')

        # Perform element-wise multiplication with the learnable weights.
        # Broadcasting handles the batch and head dimensions automatically.
        # The line that caused the error is replaced by this correct implementation.
        x_fft = x_fft * self.complex_weights
        
        # Squeeze the extra dimension before the inverse transform
        x_fft = x_fft

        # Apply Inverse Fourier Transform
        x_ifft = torch.fft.irfft2(x_fft, s=(H, W), dim=(2, 3), norm='ortho')

        # Reshape back to sequence format
        x = x_ifft.view(B, self.num_heads, N, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block that now accepts spatial_size for its attention layer."""
    def __init__(self, embed_dim, num_heads, spatial_size, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # Pass the spatial_size to the attention layer
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
        H = W = int(N**0.5)
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
        H = W = int(N**0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.convT(x)
        return x.flatten(2).transpose(1, 2)

class FourierTransformerUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=1, out_chans=1,
                 embed_dims=[64, 128, 256], num_heads=[2, 4, 8], depth=[2, 2, 2]):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        
        # Calculate initial spatial size of the patch grid
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
                H, W = H // 2, W // 2 # Update spatial size for the next stage

        # --- Bottleneck ---
        self.bottleneck = nn.ModuleList([
            TransformerBlock(embed_dims[-1], num_heads[-1], (H, W)) for _ in range(depth[-1])
        ])

        # --- Decoder ---
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        # --- FIX START ---
        # Add a list for projection layers in the decoder
        self.decoder_projs = nn.ModuleList()
        # --- FIX END ---
        
        for i in reversed(range(len(embed_dims) - 1)):
            H, W = H * 2, W * 2 # Update spatial size for the decoder stage
            self.upsamples.append(Upsample(embed_dims[i+1], embed_dims[i]))
            
            # --- FIX START ---
            # Add a projection layer to handle the doubled channel dimension after concatenation.
            # This projects from embed_dims[i]*2 back down to embed_dims[i].
            self.decoder_projs.append(nn.Linear(embed_dims[i] * 2, embed_dims[i]))
            
            # The transformer blocks should now operate on the original (projected) embed_dim.
            stage_blocks = nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i], (H, W)) for _ in range(depth[i])
            ])
            # --- FIX END ---
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
            
            # --- FIX START ---
            # Apply the projection layer to fix the channel dimension
            x = self.decoder_projs[i](x)
            # --- FIX END ---
            
            for block in stage:
                x = block(x)
        
        # Final Projection and Reshaping
        B, N, C = x.shape
        H = W = int(N**0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.final_proj(x)
        x = self.pixel_shuffle(x)
        
        return x


# In[ ]:




