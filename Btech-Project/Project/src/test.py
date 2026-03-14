#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from afno import FourierTransformerUNet
from train import PhaseRetrievalDataset
from skimage.metrics import structural_similarity as ssim


# In[ ]:


def psnr(target, prediction, max_pixel=1.0):
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def save_example_images(inputs, predictions, targets, save_dir, base_idx):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = inputs.shape[0]
    
    fig, axes = plt.subplots(3, batch_size, figsize=(batch_size * 3, 9))
    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    for i in range(batch_size):
        # Input Image
        axes[0, i].imshow(inputs[i, 0], cmap='gray')
        axes[0, i].set_title(f'Input #{base_idx + i}')
        axes[0, i].axis('off')
        
        # Model Prediction
        axes[1, i].imshow(predictions[i, 0], cmap='gray')
        axes[1, i].set_title('Prediction')
        axes[1, i].axis('off')
        
        # Ground Truth
        axes[2, i].imshow(targets[i, 0], cmap='gray')
        axes[2, i].set_title('Ground Truth')
        axes[2, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_{base_idx}_to_{base_idx + batch_size - 1}.png'))
    plt.close()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final evaluation script for Phase Retrieval.")
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'transformer'], help='Model architecture to test.')
    parser.add_argument('--base', type=int, default=64, help='Base number of feature maps in the first UNet layer')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling instead of transposed convolutions')
    parser.add_argument('--attention', action='store_true', help='Enable attention gates in the UNet skip connections')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate inside UNet convolutional blocks')
    parser.add_argument('--illumination', type=str, required=True, choices=['standard', 'vortex'], help='Illumination type model was trained on.')
    parser.add_argument('--model-path', type=str, default=None, help='Explicit path to the saved model weights. If None, constructs a default path.')
    parser.add_argument('--data-path', type=str, default='../data/processed/flickr30k_256x256_augmented', help='Path to processed data.')
    parser.add_argument('--results-path', type=str, default='../results', help='Path to the results folder.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for testing.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode on a small subset of data.')
    parser.add_argument('--save-examples', type=int, default=5, help='Number of example images to save. Set to 0 to disable.')
    parser.add_argument('--tag', type=str, default='', help='A custom tag for the run, used in plot filenames.')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing {args.model.upper()} with {args.illumination.upper()} illumination on {DEVICE}")

    # Model Path
    if args.model_path is None:
        args.model_path = os.path.join(args.results_path, 'models', f'best_{args.model}_{args.illumination}.pth')

    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}")
        exit()
    print(f"Loading model from: {args.model_path}")

    # Data Loading
    if args.debug:
        print("\n" + "="*50)
        print("RUNNING IN DEBUG MODE")
        print("="*50 + "\n")
        test_data_path = os.path.join(args.data_path, 'debug')
    else:
        test_data_path = os.path.join(args.data_path, 'test')
        
    test_dataset = PhaseRetrievalDataset(test_data_path, args.illumination)
    if len(test_dataset) == 0:
        print(f"Error: No data found in the specified test directory: {test_data_path}")
        print("Please check the path and ensure the dataset has been prepared correctly.")
        exit() 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model Initialization and Loading
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
    
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    # Evaluation Loop
    total_mse, total_psnr, total_ssim = 0.0, 0.0, 0.0
    loss_fn = nn.MSELoss()
    examples_saved = 0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            predictions = model(inputs)

            # Calculate metrics
            total_mse += loss_fn(predictions, targets).item()
            preds_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Iterate over each image in the batch for PSNR and SSIM
            for j in range(preds_np.shape[0]):
                total_psnr += psnr(targets_np[j, 0], preds_np[j, 0])
                total_ssim += ssim(targets_np[j, 0], preds_np[j, 0], data_range=1.0)
            
            # Save visual examples
            if examples_saved < args.save_examples:
                save_dir = os.path.join(args.results_path, 'test_examples', f'{args.model}_{args.illumination}_{args.tag}')
                save_example_images(inputs.cpu().numpy(), preds_np, targets_np, save_dir, base_idx=i*args.batch_size)
                examples_saved += inputs.shape[0]

    # Final Report
    num_images = len(test_dataset)
    avg_mse = total_mse / len(test_loader)
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    print("\n" + "="*50)
    print("--- FINAL TEST RESULTS ---")
    print(f"Model: {args.model.upper()} | Illumination: {args.illumination.upper()}")
    print("--------------------------")
    print(f"Average MSE:  {avg_mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*50)
    if args.save_examples > 0:
        print(f"Saved example images to: {os.path.join(args.results_path, 'test_examples')}")

