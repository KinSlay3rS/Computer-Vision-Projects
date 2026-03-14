#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# File: train_unrolled.py

import os
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from unrolled_hio import UnrolledHIOPhaseNet

# ---------------- Dataset ----------------
class PhaseRetrievalDatasetOversampled(Dataset):
    """
    - log-magnitude (input to CNN): 1-channel, ∈ [0,1]
    - linear magnitude (input to HIO): 1-channel, ∈ [0,1]
    - target phase = π * amplitude, ∈ [0, π]
    """
    def __init__(self, data_path, illumination_type='standard'):
        self.gt_path = os.path.join(data_path, 'ground_truth')
        self.input_path = os.path.join(data_path, f'input_{illumination_type}')
        self.filenames = [f for f in os.listdir(self.gt_path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base = self.filenames[idx]
        gt = np.load(os.path.join(self.gt_path, base))                 # amplitude ∈ [0,1]
        log_mag = np.load(os.path.join(self.input_path, base))         # log-mag ∈ [0,1]
        lin_mag = np.load(os.path.join(self.input_path, base.replace('.npy', '_lin.npy')))  # linear ∈ [0,1]

        gt_phase = np.pi * gt.astype(np.float32)                       # true phase

        return (
            torch.from_numpy(log_mag).unsqueeze(0).float(),        # [1,H,W]
            torch.from_numpy(lin_mag).unsqueeze(0).float(),        # [1,H,W]
            torch.from_numpy(gt_phase).unsqueeze(0).float()        # [1,H,W]
        )


# ---------------- Metrics ----------------

def psnr_phase(target, prediction, max_val=math.pi):
    mse = torch.mean((target - prediction) ** 2)
    if mse <= 0:
        return float('inf')
    return 20.0 * torch.log10(max_val / torch.sqrt(mse))

def save_plots(train_losses, val_losses, val_psnrs,
               results_path, illum_name, n_steps, beta, tag=""):
    plots_dir = os.path.join(results_path, 'plots_unrolled')
    os.makedirs(plots_dir, exist_ok=True)
    base = f'unrolled_{illum_name}_steps{n_steps}_beta{beta}'
    if tag:
        base += f"_{tag}"

    # Loss
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (phase)')
    plt.legend()
    plt.grid(True)
    plt.title(base + " - Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, base + "_loss.png"))
    plt.close()

    # PSNR
    plt.figure(figsize=(8,4))
    plt.plot(val_psnrs, label='Val PSNR (phase)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.title(base + " - PSNR")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, base + "_psnr.png"))
    plt.close()

# ---------------- Train / Val loops ----------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for log_mag, lin_mag, gt_phase in tqdm(loader, desc="Training"):
        log_mag, lin_mag, gt_phase = log_mag.to(device), lin_mag.to(device), gt_phase.to(device)
        optimizer.zero_grad()
        pred_phase = model(log_mag, lin_mag)
        loss = loss_fn(pred_phase, gt_phase)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for log_mag, lin_mag, gt_phase in tqdm(loader, desc="Validating"):
            log_mag, lin_mag, gt_phase = log_mag.to(device), lin_mag.to(device), gt_phase.to(device)
            pred_phase = model(log_mag, lin_mag)
            loss = loss_fn(pred_phase, gt_phase)
            total_loss += loss.item()
            total_psnr += psnr_phase(gt_phase, pred_phase).item()
    return total_loss / len(loader), total_psnr / len(loader)


# ---------------- Main ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unrolled HIO+CNN phase retrieval model.")
    parser.add_argument('--illumination', type=str, required=True, choices=['standard', 'vortex'],
                        help='Illumination type.')
    parser.add_argument('--data-path', type=str,
                        default='../data/processed/flickr30k_256x256_oversampled',
                        help='Path to processed oversampled data.')
    parser.add_argument('--results-path', type=str, default='../results_unrolled',
                        help='Path to save models and plots.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--steps', type=int, default=8, help='Number of unrolled HIO steps.')
    parser.add_argument('--beta', type=float, default=0.9, help='HIO beta.')
    parser.add_argument('--support-ratio', type=float, default=0.5,
                        help='Centered support size as fraction of image.')
    parser.add_argument('--base-ch', type=int, default=32, help='Base channels in denoiser and init net.')
    parser.add_argument('--debug', action='store_true',
                        help='Use debug split instead of train/val.')
    parser.add_argument('--tag', type=str, default='', help='Extra tag for filenames.')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.debug:
        train_path = os.path.join(args.data_path, 'debug')
        val_path = os.path.join(args.data_path, 'debug')
        print("Running in DEBUG mode on 'debug' split.")
    else:
        train_path = os.path.join(args.data_path, 'train')
        val_path = os.path.join(args.data_path, 'val')

    print(f"Training UNROLLED HIO on {DEVICE} | Illumination: {args.illumination}")

    # Data
    train_dataset = PhaseRetrievalDatasetOversampled(train_path, args.illumination)
    val_dataset   = PhaseRetrievalDatasetOversampled(val_path, args.illumination)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    print("TRAIN SAMPLES:", len(train_dataset))
    print("VAL SAMPLES:", len(val_dataset))


    # Model
    model = UnrolledHIOPhaseNet(
        img_size=256,
        n_steps=args.steps,
        beta=args.beta,
        support_ratio=args.support_ratio,
        base_ch=args.base_ch
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {num_params/1e6:.2f}M")

    # Loss + optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Paths
    os.makedirs(args.results_path, exist_ok=True)
    models_dir = os.path.join(args.results_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_ckpt = os.path.join(
        models_dir,
        f"best_unrolled_{args.illumination}_steps{args.steps}_beta{args.beta}_lr{args.lr}_bs{args.batch_size}{('_'+args.tag) if args.tag else ''}.pth"
    )

    # Train loop
    best_val_psnr = -1e9
    train_hist, val_hist, psnr_hist = [], [], []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_psnr = validate(model, val_loader, loss_fn, DEVICE)

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        psnr_hist.append(val_psnr)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), model_ckpt)
            print(f"Saved new best model to {model_ckpt} (Val PSNR: {best_val_psnr:.2f} dB)")

    save_plots(train_hist, val_hist, psnr_hist,
               args.results_path, args.illumination, args.steps, args.beta, tag=args.tag)

    print("\nTraining finished.")
    print(f"Best validation PSNR (phase): {best_val_psnr:.2f} dB")




