#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act2 = nn.SiLU(inplace=True)

        # match channels for residual
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.res_gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        else:
            self.res_conv = None

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.dropout(out)

        if self.res_conv is not None:
            residual = self.res_conv(residual)
            residual = self.res_gn(residual)

        out = out + residual
        out = self.act2(out)
        return out

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, F_int//4), num_channels=F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, F_int//4), num_channels=F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.sigmoid(self.psi(torch.relu(g1 + x1)))
        return x * psi

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResidualConv(in_ch, out_ch, groups=groups, dropout=dropout)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, groups=8, dropout=0.0, attention=False):
        super().__init__()
        self.bilinear = bilinear
        self.attention = attention
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv_after_up = nn.Conv2d(in_ch//2, in_ch//2, kernel_size=1)  
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)

        self.conv = ResidualConv(in_ch, out_ch, groups=groups, dropout=dropout)

        if attention:
            self.att_gate = AttentionGate(F_g=in_ch//2, F_l=in_ch//2, F_int=in_ch//4)

    def forward(self, x, skip):
        if self.bilinear:
            x = self.up(x)
        else:
            x = self.up(x)

        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        if self.attention:
            skip = self.att_gate(skip, x)

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_ch=64, bilinear=True, attention=False, dropout=0.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ResidualConv(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2, dropout=dropout)
        self.down2 = Down(base_ch*2, base_ch*4, dropout=dropout)
        self.down3 = Down(base_ch*4, base_ch*8, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch*8, base_ch*16 // factor, dropout=dropout)

        self.up1 = Up(base_ch*16, base_ch*8 // factor, bilinear=bilinear, attention=attention, dropout=dropout)
        self.up2 = Up(base_ch*8, base_ch*4 // factor, bilinear=bilinear, attention=attention, dropout=dropout)
        self.up3 = Up(base_ch*4, base_ch*2 // factor, bilinear=bilinear, attention=attention, dropout=dropout)
        self.up4 = Up(base_ch*2, base_ch, bilinear=bilinear, attention=attention, dropout=dropout)

        self.outc = OutConv(base_ch, n_classes)

        self.apply(init_weights)

    def forward(self, x):
        x1 = self.inc(x)        
        x2 = self.down1(x1)     
        x3 = self.down2(x2)     
        x4 = self.down3(x3)     
        x5 = self.down4(x4)     

        x = self.up1(x5, x4)    
        x = self.up2(x, x3)     
        x = self.up3(x, x2)     
        x = self.up4(x, x1)     
        logits = self.outc(x)   
        return logits


# In[ ]:




