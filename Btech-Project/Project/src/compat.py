import torch
import torch.nn as nn
import math

# Check what torch.fft actually is
print(f"torch.fft type: {type(torch.fft) if hasattr(torch, 'fft') else 'does not exist'}")

# ComplexTensor class
class ComplexTensor:
    def _init_(self, real, imag):
        self._real = real
        self._imag = imag
        self.shape = real.shape
        self.device = real.device
        self.dtype = real.dtype
    
    @property
    def real(self):
        return self._real
    
    @property
    def imag(self):
        return self._imag
    
    def _getitem_(self, key):
        return ComplexTensor(self._real[key], self._imag[key])
    
    def _setitem_(self, key, value):
        if hasattr(value, '_real'):
            self._real[key] = value._real
            self._imag[key] = value._imag
        elif hasattr(value, 'real'):
            self._real[key] = value.real
            self._imag[key] = value.imag
    
    def clone(self):
        return ComplexTensor(self._real.clone(), self._imag.clone())
    
    def permute(self, *dims):
        return ComplexTensor(self._real.permute(*dims), self._imag.permute(*dims))
    
    def contiguous(self):
        return ComplexTensor(self._real.contiguous(), self._imag.contiguous())

# FFT compatibility functions
def rfft2_compat(input, s=None, dim=(-2, -1), norm=None):
    normalized = (norm == 'ortho')
    result_old = torch.rfft(input, signal_ndim=2, normalized=normalized, onesided=True)
    return ComplexTensor(result_old[..., 0], result_old[..., 1])

def irfft2_compat(input, s=None, dim=(-2, -1), norm=None):
    normalized = (norm == 'ortho')
    if hasattr(input, '_real'):
        input_old = torch.stack([input._real, input._imag], dim=-1)
    else:
        input_old = torch.stack([input.real, input.imag], dim=-1)
    return torch.irfft(input_old, signal_ndim=2, normalized=normalized,
                     onesided=True, signal_sizes=s)

# Check if torch.fft module has rfft2
needs_patch = True
if hasattr(torch, 'fft'):
    if hasattr(torch.fft, 'rfft2') and callable(torch.fft.rfft2):
        needs_patch = False
        print("✓ torch.fft.rfft2 exists, no patch needed")

if needs_patch:
    print("⚠ Patching torch.fft...")
    
    # Save old torch.fft if it exists (it might be a function)
    old_fft = getattr(torch, 'fft', None)
    
    # Create new fft module
    class FFTModule:
        rfft2 = staticmethod(rfft2_compat)
        irfft2 = staticmethod(irfft2_compat)
        
        # Keep old fft function if it existed
        if old_fft is not None and callable(old_fft):
            fft = staticmethod(old_fft)
    
    # Replace torch.fft
    torch.fft = FFTModule()
    print("✓ torch.fft.rfft2 patched")

# Patch torch.complex if needed
if not hasattr(torch, 'complex') or not callable(torch.complex):
    torch.complex = lambda r, i: ComplexTensor(r, i)
    print("✓ torch.complex patched")

# Patch torch.zeros_like for ComplexTensor
original_zeros_like = torch.zeros_like
def zeros_like_compat(input, *args, **kwargs):
    if isinstance(input, ComplexTensor):
        return ComplexTensor(
            torch.zeros_like(input._real, *args, **kwargs),
            torch.zeros_like(input._imag, *args, **kwargs)
        )
    return original_zeros_like(input, *args, **kwargs)
torch.zeros_like = zeros_like_compat

# Patch nn.GELU if needed
if not hasattr(nn, 'GELU'):
    class GELU(nn.Module):
        def _init_(self):
            super()._init_()
        def forward(self, x):
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    nn.GELU = GELU
    print("✓ nn.GELU patched")

print("=" * 50)
print("Compatibility patches applied successfully!")
print("=" * 50)
