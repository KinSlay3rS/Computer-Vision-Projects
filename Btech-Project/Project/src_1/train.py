#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Avoid GUI backend issues in older torch envs
import matplotlib.pyplot as plt
from unet import UNet
from afno import FourierTransformerUNet


# ============================================================
# Dataset Class (compatible with torch==1.7.1)
# ============================================================

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


# ============================================================
# Utility Functions
# ============================================================

def psnr(target, prediction, max_pixel=1.0):
    mse = torch.mean((target - prediction) ** 2)
    if mse.item() == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def save_plots(train_losses, val_losses, val_psnrs, results_path, model_name, illum_name, lr, batch_size, tag):
    plots_dir = os.path.join(results_path, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    base_filename = f"{model_name}_{illum_name}_lr{lr}_bs{batch_size}"
    
    if tag:
        base_filename += f"_{tag}"

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss vs. Epochs for {base_filename}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
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


# ============================================================
# Training / Validation
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training", ncols=80):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating", ncols=80):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            total_psnr += psnr(targets, predictions)
    return total_loss / len(loader), total_psnr / len(loader)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for Phase Retrieval (PyTorch 1.7.1 compatible).")
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'transformer'], help='Model to train.')
    parser.add_argument('--base', type=int, default=64, help='Base number of feature maps in first UNet layer.')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling.')
    parser.add_argument('--attention', action='store_true', help='Enable attention gates in skip connections.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in UNet blocks.')
    parser.add_argument('--illumination', type=str, required=True, choices=['standard', 'vortex'], help='Illumination type.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--data-path', type=str, default='../data/processed/flickr30k_256x256_augmented', help='Data path.')
    parser.add_argument('--results-path', type=str, default='../results', help='Path to save models and plots.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (small dataset).')
    parser.add_argument('--tag', type=str, default='', help='Tag for filenames.')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.debug:
        print("\n" + "="*50)
        print(" RUNNING IN DEBUG MODE ")
        print("="*50 + "\n")
        args.epochs = 10
        train_data_path = os.path.join(args.data_path, 'debug')
        val_data_path = os.path.join(args.data_path, 'debug')
    else:
        train_data_path = os.path.join(args.data_path, 'train')
        val_data_path = os.path.join(args.data_path, 'val')

    print(f"Training {args.model.upper()} with {args.illumination.upper()} illumination on {DEVICE}")

    models_dir = os.path.join(args.results_path, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Data Loading
    train_dataset = PhaseRetrievalDataset(train_data_path, args.illumination)
    val_dataset = PhaseRetrievalDataset(val_data_path, args.illumination)
    
    # num_workers=0 for older CUDA compatibility
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model Setup
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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params / 1e6:.2f} M parameters.")

    # Optimizer / Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Histories
    train_loss_history, val_loss_history, val_psnr_history = [], [], []
    best_val_psnr = 0.0

    model_save_path = os.path.join(
        models_dir, f'best_{args.model}_{args.illumination}_lr{args.lr}_bs{args.batch_size}_{args.tag}.pth'
    )

    # Training Loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_psnr = validate(model, val_loader, loss_fn, DEVICE)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_psnr_history.append(val_psnr)

        print(f"Epoch Summary -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved ({best_val_psnr:.2f} dB) -> {model_save_path}")

    print("\nTraining Finished")
    print(f"Best validation PSNR: {best_val_psnr:.2f} dB")

    save_plots(
        train_losses=train_loss_history,
        val_losses=val_loss_history,
        val_psnrs=val_psnr_history,
        results_path=args.results_path,
        model_name=args.model,
        illum_name=args.illumination,
        lr=args.lr,
        batch_size=args.batch_size,
        tag=args.tag
    )

    print(f"Training plots saved in {os.path.join(args.results_path, 'plots')}")

