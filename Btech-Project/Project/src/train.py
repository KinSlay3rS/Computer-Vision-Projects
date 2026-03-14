#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.fft as fft
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from afno import FourierTransformerUNet
import lpips



# In[ ]:


class PhaseRetrievalDataset(Dataset):
    def __init__(self, data_path, illumination_type='standard'):
        self.gt_path = os.path.join(data_path, 'ground_truth')
        self.input_path = os.path.join(data_path, f'input_{illumination_type}')
        self.filenames = [f for f in os.listdir(self.gt_path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        gt = np.load(os.path.join(self.gt_path, self.filenames[idx]))
        ipt = np.load(os.path.join(self.input_path, self.filenames[idx]))
        
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()
        ipt_tensor = torch.from_numpy(ipt).unsqueeze(0).float()
        
        return ipt_tensor, gt_tensor


# In[ ]:


def psnr(target, prediction, max_pixel=1.0):
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def save_plots(train_losses, val_losses, val_psnrs, results_path, model_name, illum_name, lr, batch_size, tag, loss_function_name):
    plots_dir = os.path.join(results_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    base_filename = f"{model_name}_{illum_name}_lr{lr}_bs{batch_size}"
    
    if tag:
        base_filename += f"_{tag}"

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    loss_title = loss_function_name.upper().replace('_', ' + ')
    plt.title(f'Loss ({loss_title}) vs. Epochs for {base_filename}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{base_filename}_loss.png'))
    plt.close()

    # PSNR Plot
    plt.figure(figsize=(10, 5))
    plt.plot(val_psnrs, label='Validation PSNR', color='green')
    plt.title(f'PSNR vs. Epochs for {base_filename}')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{base_filename}_psnr.png'))
    plt.close()


# In[ ]:
class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1=100.0, device='cuda'):
        super().__init__()
        self.lambda_l1 = lambda_l1
        # Initialize the LPIPS model. It will download the VGG weights on first run.
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction, target):
        # LPIPS expects a 3-channel image in the range [-1, 1].
        # Our images are 1-channel [0, 1], so we must adapt them.
        
        # 1. Replicate grayscale channel to create a 3-channel image
        prediction_3ch = prediction.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)

        # 2. Rescale from [0, 1] to [-1, 1] for the LPIPS model
        prediction_lpips = prediction_3ch * 2 - 1
        target_lpips = target_3ch * 2 - 1

        # --- Calculate the two loss components ---
        # Perceptual Loss (for realism and texture)
        loss_lpips = self.lpips_loss(prediction_lpips, target_lpips).mean()
        
        # L1 Loss (for pixel-level accuracy)
        loss_l1 = self.l1_loss(prediction, target)
        
        # Return the weighted sum
        return self.lambda_l1 * loss_l1 + loss_lpips
    
def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction, target):
        # Get the 2D Fourier Transform of the images
        pred_fft = torch.fft.fft2(prediction)
        targ_fft = torch.fft.fft2(target)

        # Calculate the L1 loss on the magnitude of the Fourier transforms
        # This forces the frequency distribution to be similar.
        return self.l1_loss(torch.abs(pred_fft), torch.abs(targ_fft))
    
class UltimateLoss(nn.Module):
    def __init__(self, lambda_l1=100.0, lambda_freq=10.0, lambda_tv=1e-4, device='cuda'):
        super().__init__()
        # --- Loss Weights (Hyperparameters) ---
        self.lambda_l1 = lambda_l1       # Main driver for pixel accuracy
        self.lambda_freq = lambda_freq # Enforces correct sharpness
        self.lambda_tv = lambda_tv     # Smooths out artifacts
        # LPIPS lambda is implicitly 1.0

        # --- Initialize Loss Components ---
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.freq_loss = FrequencyLoss()

    def forward(self, prediction, target):
        # --- LPIPS Loss Pre-processing ---
        prediction_3ch = prediction.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)
        prediction_lpips = prediction_3ch * 2 - 1
        target_lpips = target_3ch * 2 - 1

        # --- Calculate all four loss components ---
        loss_l1 = self.l1_loss(prediction, target)
        loss_lpips = self.lpips_loss(prediction_lpips, target_lpips).mean()
        loss_freq = self.freq_loss(prediction, target)
        loss_tv = total_variation_loss(prediction)
        
        # Return the final, four-part weighted sum
        return self.lambda_l1 * loss_l1 + loss_lpips + self.lambda_freq * loss_freq + self.lambda_tv * loss_tv

class FinalLoss(nn.Module):
    def __init__(self, lambda_l1=100.0, lambda_fourier=10.0, lambda_tv=1e-4, device='cuda'):
        super().__init__()
        # --- Loss Weights (Hyperparameters) ---
        self.lambda_l1 = lambda_l1         # For pixel accuracy
        self.lambda_fourier = lambda_fourier # For self-consistency in Fourier domain
        self.lambda_tv = lambda_tv       # For denoising
        # LPIPS lambda is implicitly 1.0

        # --- Initialize Loss Components ---
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss() # For the Fourier component
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)

    def forward(self, prediction, target, input_log_magnitude, use_vortex=False):
        # --- 1. L1 Loss (Pixel Fidelity) ---
        loss_l1 = self.l1_loss(prediction, target)

        # --- 2. LPIPS Loss (Perceptual Realism) ---
        prediction_3ch = prediction.repeat(1, 3, 1, 1); target_3ch = target.repeat(1, 3, 1, 1)
        prediction_lpips = prediction_3ch * 2 - 1; target_lpips = target_3ch * 2 - 1
        loss_lpips = self.lpips_loss(prediction_lpips, target_lpips).mean()

        # --- 3. Fourier Log-Magnitude Loss (Self-Consistency) ---
        # Re-create the phase object from the model's prediction
        pred_phase_map = np.pi * prediction
        pred_complex = torch.exp(1j * pred_phase_map)

        if use_vortex:
            ny, nx = pred_complex.shape[-2:]
            x = torch.linspace(-nx//2, nx//2, nx, device=prediction.device)
            y = torch.linspace(-ny//2, ny//2, ny, device=prediction.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            vortex_mask = torch.exp(1j * torch.atan2(yy, xx))
            exit_wave_hat = pred_complex * vortex_mask
        else:
            exit_wave_hat = pred_complex

        F_hat = fft.fftshift(fft.fft2(fft.ifftshift(exit_wave_hat)))
        logmag_hat = torch.log1p(torch.abs(F_hat))
        # Normalize the re-simulated magnitude for fair comparison
        logmag_hat = (logmag_hat - torch.min(logmag_hat)) / (torch.max(logmag_hat) - torch.min(logmag_hat))
        
        loss_fourier = self.mse_loss(logmag_hat, input_log_magnitude)
        
        # --- 4. Total Variation Loss (Denoising) ---
        loss_tv = total_variation_loss(prediction)
        
        # --- Final Weighted Sum ---
        total_loss = self.lambda_l1 * loss_l1 + loss_lpips + self.lambda_fourier * loss_fourier + self.lambda_tv * loss_tv
        return total_loss

def train_one_epoch(model, loader, optimizer, loss_fn, device, use_vortex = False):
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets, inputs, use_vortex)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            total_psnr += psnr(targets, predictions).item()
    return total_loss / len(loader), total_psnr / len(loader)


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full-fledged training script for Phase Retrieval.")
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'transformer'], help='Model to train.')
    parser.add_argument('--base', type=int, default=64, help='Base number of feature maps in the first UNet layer')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling instead of transposed convolutions')
    parser.add_argument('--attention', action='store_true', help='Enable attention gates in the UNet skip connections')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate inside UNet convolutional blocks')
    parser.add_argument('--illumination', type=str, required=True, choices=['standard', 'vortex'], help='Illumination type.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--data-path', type=str, default='../data/processed/flickr30k_256x256_oversampled', help='Path to processed data.')
    parser.add_argument('--results-path', type=str, default='../results', help='Path to save models and plots.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode on a small subset of data.')
    parser.add_argument('--tag', type=str, default='', help='A custom tag for the run, used in plot filenames.')
    parser.add_argument('--loss-function', type=str, default='final_four', choices=['mse', 'mae', 'l1_lpips', 'ultimate','final_four'], help='Loss function to use for training.')
    parser.add_argument('--resume', type=str, default=None, help='Path to a model checkpoint to resume training from.')
    args = parser.parse_args()

    # Setup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.debug:
        print("\n" + "="*50)
        print("RUNNING IN DEBUG MODE")
        print("="*50 + "\n")
        args.epochs = 10 
        train_data_path = os.path.join(args.data_path, 'debug')
        val_data_path = os.path.join(args.data_path, 'debug') 
    else:
        train_data_path = os.path.join(args.data_path, 'train')
        val_data_path = os.path.join(args.data_path, 'val')
    print(f"Training {args.model.upper()} with {args.illumination.upper()} illumination on {DEVICE}")
    
    models_dir = os.path.join(args.results_path, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Data Loading
    train_dataset = PhaseRetrievalDataset(train_data_path, args.illumination)
    val_dataset = PhaseRetrievalDataset(val_data_path, args.illumination)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False)

    # Model Initialization
    if args.model == 'unet':
        model = UNet(
            n_channels=1,
            n_classes=1,
            base_ch=args.base,
            bilinear=args.bilinear,
            attention=args.attention,
            dropout=args.dropout
        ).to(DEVICE)
    else:
        model = FourierTransformerUNet(
            img_size=256, patch_size=4,
            embed_dims=[64, 128, 256], depth=[2, 2, 2],
            attn_blocks=[4, 8, 8],
            attn_modes=[12, 12, 16]
        ).to(DEVICE)

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        else:
            print(f"Warning: Checkpoint path not found, starting from scratch: {args.resume}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params / 1e6:.2f} M parameters.")

    # Training
    if args.loss_function == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss_function == 'l1_lpips':
        loss_fn = UltimateLoss(lambda_l1=100.0, lambda_freq=0.0, lambda_tv=0.0, device=DEVICE)
    elif args.loss_function == 'ultimate':
        loss_fn = UltimateLoss(lambda_l1=100.0, lambda_freq=10.0, lambda_tv=1e-4, device=DEVICE)
    elif args.loss_function == 'mae':
        loss_fn = nn.L1Loss()
    elif args.loss_function == 'final_four':
        loss_fn = FinalLoss(lambda_l1=100.0, lambda_fourier=10.0, lambda_tv=1e-4, device=DEVICE)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_loss_history, val_loss_history, val_psnr_history = [], [], []
    best_val_psnr = 0.0
    model_save_path = os.path.join(models_dir, f'best_{args.model}_{args.illumination}_lr{args.lr}_bs{args.batch_size}_{args.tag}.pth')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        is_vortex=True if args.illumination=='vortex' else False
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, use_vortex=is_vortex)
        val_loss, val_psnr = validate(model, val_loader, nn.MSELoss(), DEVICE)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_psnr_history.append(val_psnr)

        print(f"Epoch Summary -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB")
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with PSNR: {best_val_psnr:.2f} dB to {model_save_path}")

    # Post-Training
    print("\nTraining Finished")
    print(f"Best validation PSNR achieved: {best_val_psnr:.2f} dB")
    plot_model_name = args.model
    plot_illum_name = args.illumination
    save_plots(
        train_losses=train_loss_history, 
        val_losses=val_loss_history, 
        val_psnrs=val_psnr_history, 
        results_path=args.results_path, 
        model_name=plot_model_name, 
        illum_name=plot_illum_name, 
        lr=args.lr, 
        batch_size=args.batch_size, 
        tag=args.tag,
        loss_function_name = args.loss_function
    )
    print(f"Training plots saved to {os.path.join(args.results_path, 'plots')}")

