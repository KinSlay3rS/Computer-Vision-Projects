# Deep Learning Based Phase Retrieval using Vortex Illumination

> B.Tech Project — Department of Physics, Indian Institute of Technology Delhi  
> **Authors:** Aviral Jain & Rahul Singh  
> **Supervisor:** Prof. Kedar Khare

---

## Table of Contents

- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Background](#background)
- [Approach](#approach)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

---

## Motivation

Imaging systems can only record the **intensity** of light — the phase information is lost. Recovering phase would give us additional knowledge about an object such as depth, contrast, and structural detail. Computational (non-interferometric) phase retrieval techniques can recover this information, but are iterative, slow, and computationally demanding. The goal of this project is to recover the phase of an image **faster and with less computational cost**, ideally approaching real-time performance.

---

## Problem Statement

Traditional non-interferometric phase retrieval algorithms such as **Gerchberg–Saxton (GS)** and **Hybrid Input-Output (HIO)** suffer from three key limitations:

- **Twin Image Problem:** Both `g(x)` and `g*(−x)` have identical Fourier magnitudes, creating an inherent ambiguity that the algorithm cannot resolve.
- **Slow Convergence:** The iterative projections stagnate or cycle without guaranteeing convergence to a valid solution.
- **High Computational Cost:** Achieving good recovery requires thousands of iterations, making real-time application infeasible.

---

## Background

### HIO Algorithm
HIO improves on the basic GS algorithm by introducing a **feedback rule** that prevents stagnation outside the support region:

$$g_{n+1}(x) = \begin{cases} g'_n(x), & x \in S \\ g_n(x) - \beta\, g'_n(x), & x \notin S \end{cases}$$

where $\beta \in [0.5, 1]$. Despite this improvement, neither GS nor HIO guarantees convergence, and both remain susceptible to the twin image problem.

### Vortex Illumination
Vortex illumination breaks the **conjugate-inversion symmetry** that causes the twin image problem:

$$\left|\mathcal{F}\{g(x)\,V(x)\}(k)\right| \neq \left|\mathcal{F}\{g^*(−x)\,V(x)\}(k)\right|$$

This eliminates the twin image ambiguity, though the computational cost of iterative recovery remains high.

---

## Approach

This project replaces or augments slow iterative algorithms with deep learning models that can learn the inverse mapping from Fourier magnitude to the original image. Three architectures were explored across two illumination types:

```
Input: log(|FFT of illuminated padded image|)
              │
    ┌─────────┴──────────┐
    │                    │
Standard Illumination   Vortex Illumination
    │                    │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │   Deep Learning    │
    │   Architecture     │
    │  (UNet / ViT /     │
    │  Unrolled HIO)     │
    └─────────┬──────────┘
              │
    Output: Recovered Image (padded)
```

---

## Dataset

| Split      | Size  |
|------------|-------|
| Training   | 8,000 |
| Validation | 1,000 |
| Test       | 1,000 |

**Source:** [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)

### Preprocessing

Two preprocessing pipelines were used:

**Type 1 (Aspect-ratio preserved):**
- Padding only along height or width to preserve aspect ratio
- Gaussian probe applied as amplitude before Fourier transform (realistic simulation)
- Input: `log(|FFT{phase_object}|)`

**Type 2 (Fixed-size padding):**
- Images resized and padded to fit a **128×128** object region within a **256×256** frame
- No Gaussian probe applied
- Simpler and more controlled setup

---

## Models

### 1. UNet
A classic encoder-decoder architecture with skip connections. Trained to directly map the Fourier magnitude to the recovered image.

- **Loss functions explored:** MSE, LPIPS, Complex Loss (MSE in frequency domain + L1 in spatial domain + LPIPS + Total Variation)
- **Illumination variants:** Standard & Vortex

### 2. Vision Transformer (ViT)
Transformer-based architecture applied to the phase retrieval task. Trained under the same conditions as UNet for comparison.

- **Loss functions explored:** MSE, LPIPS, Complex Loss
- **Illumination variants:** Standard & Vortex

### 3. Unrolled HIO (HIO + CNN Denoiser)
A physics-informed architecture that **unrolls the HIO iterative algorithm** into a trainable network. Each stage mimics one HIO step followed by a CNN denoiser, enabling end-to-end learning while respecting the physical constraints of the problem.

- **Architecture size:** ~0.5M parameters
- **Training:** 50 epochs, 12 unrolled iterations
- **Illumination variants:** Standard & Vortex

> **Key Insight:** Network complexity alone cannot fix the ill-posedness of phase retrieval. Embedding physical constraints directly into the architecture (as done in Unrolled HIO) is essential for reliable recovery.

---

## Results

### Mid-Semester (UNet on CelebA — Pilot Study)

| Method           | MAE      | PSNR (dB) |
|------------------|----------|-----------|
| Plane Wave Model | 0.206309 | 12.07     |
| Vortex Model     | 0.066045 | 16.30     |

Vortex illumination significantly outperformed standard illumination in the pilot study. However, the CelebA dataset (faces only) caused the model to output a similar image regardless of the input, motivating a switch to Flickr30k.

---

### UNet & ViT on Flickr30k

Models trained with MSE loss (50 epochs, Type 1 preprocessing):

- UNet models reconstruct general structure but **lack high-frequency detail**
- Transformer models show comparable performance with blocky artifacts
- Vortex illumination consistently produced better reconstructions than standard illumination across both architectures

**HIO Warm-Starting with DL Output:**

| Model                | PSNR (DL) | PSNR (DL→HIO) | PSNR (Rand→HIO) |
|----------------------|-----------|----------------|-----------------|
| UNet-Standard        | 13.646    | 5.980          | 3.939           |
| UNet-Vortex          | 13.837    | 2.615          | 2.606           |
| Transformer-Standard | 10.662    | 6.710          | 3.939           |
| Transformer-Vortex   | 10.770    | 3.568          | 2.606           |

Using a **DL-predicted smart support mask** (thresholding DL output at intensity > 0.1) to initialize HIO gave statistically significant improvement over random initialization (paired t-test p-values as low as 1.1e-4).

---

### Unrolled HIO — Best Results

The Unrolled HIO model, despite having only ~0.5M parameters, dramatically outperformed classic HIO:

**Standard Illumination:**

| Model        | Mean PSNR | Mean SSIM |
|--------------|-----------|-----------|
| Unrolled HIO | **20.36** | **0.771** |
| Classic HIO  | 5.37      | 0.005     |

Paired t-test: T = 223.24, **p = 0.000e+00**

**Vortex Illumination:**

| Model        | Mean PSNR | Mean SSIM |
|--------------|-----------|-----------|
| Unrolled HIO | **16.47** | **0.770** |
| Classic HIO  | 5.66      | 0.033     |

Paired t-test: T = 208.49, **p = 0.000e+00**

The Unrolled HIO (Standard) model achieved convergence quality in **500 DL-initialized HIO iterations** that classic HIO could not match even in **5000 random iterations**.

---

## Conclusion

- Standalone deep learning architectures (UNet, ViT) with complex loss functions can recover coarse image structure but fail to capture fine details due to the inherent ill-posedness of the phase retrieval problem.
- A lightweight **Unrolled HIO (0.5M parameters)** combining physics-based HIO iterations with a CNN denoiser significantly outperforms both classic HIO and standalone DL models.
- DL-generated **smart support masks** provide statistically reliable warm-starts for classic HIO, improving convergence.
- Vortex illumination resolves the twin image ambiguity and consistently improves reconstruction quality over standard plane-wave illumination.

---

## Future Work

- Extend the Unrolled HIO framework to work with **noisy and experimental measurements**
- Explore learnable support estimation integrated directly into the unrolled network
- Apply to **real optical setups** using vortex phase plates
- Investigate **multi-distance** or **multi-wavelength** phase retrieval with DL
- Improve high-frequency detail recovery through adversarial (GAN-based) training

---

## Getting Started

### Installation

```bash
git clone https://github.com/KinSlay3rS/Computer-Vision-Projects.git
cd Computer-Vision-Projects/Btech-Project/Project
pip install -e .
```

### Dataset Preparation

Download the [Flickr30k dataset](https://shannon.cs.illinois.edu/DenotationGraph/) and run the dataset preparation notebook:

```bash
jupyter notebook src/prepare_dataset_flickr.ipynb
# Then run:
jupyter notebook src/create_dataset.ipynb
```

### Training

**UNet / Vision Transformer:**
```bash
python src/train.py
# or submit via HPC:
bash run.sh          # MSE loss, standard illumination
bash run3_lp.sh      # LPIPS loss
bash run4_lp.sh      # Complex loss
```

**Unrolled HIO:**
```bash
python src/train_unrolled.py
# or via HPC job scripts, logs saved to:
# out_unrolled_st.log / out_unrolled_vortex.log
```

### Evaluation

```bash
python src/test.py
# or interactively:
jupyter notebook src/test.ipynb
```

---

## Repository Structure

```
Btech-Project/
└── Project/
    ├── src/                            # Core source code
    │   ├── unet.py                     # UNet architecture
    │   ├── transformer.py              # Vision Transformer (ViT) architecture
    │   ├── afno.py                     # Adaptive Fourier Neural Operator
    │   ├── new_attention.py            # Attention module
    │   ├── unrolled_hio.py             # Unrolled HIO + CNN denoiser
    │   ├── train.py                    # Training script (UNet / ViT)
    │   ├── train_unrolled.py           # Training script (Unrolled HIO)
    │   ├── test.py                     # Evaluation / inference script
    │   ├── prepare_dataset_flickr.ipynb  # Flickr30k dataset preparation
    │   ├── create_dataset.ipynb        # Dataset creation & preprocessing
    │   ├── train.ipynb                 # Training notebook
    │   ├── test.ipynb                  # Testing & visualization notebook
    │   ├── unet.ipynb                  # UNet experiments notebook
    │   ├── transformer.ipynb           # Transformer experiments notebook
    │   └── models.ipynb                # Model comparison notebook
    │
    ├── src_1/                          # Alternate/experimental training scripts
    │   ├── afno.py
    │   └── train.py
    │
    ├── results/                        # UNet & Transformer outputs
    │   ├── models/                     # Saved model checkpoints (.pth)
    │   │   ├── best_unet_standard_*.pth
    │   │   ├── best_unet_vortex_*.pth
    │   │   ├── best_transformer_standard_*.pth
    │   │   └── best_transformer_vortex_*.pth
    │   ├── plots/                      # Training loss & PSNR curves, sample comparisons
    │   └── test_examples/              # Visual comparison outputs on test set
    │
    ├── results_unrolled/               # Unrolled HIO outputs
    │   ├── models/                     # Saved Unrolled HIO checkpoints (.pth)
    │   │   ├── best_unrolled_standard_steps12_*.pth
    │   │   └── best_unrolled_vortex_steps12_*.pth
    │   └── plots_unrolled/             # Loss & PSNR curves for Unrolled HIO
    │
    ├── run.sh                          # SLURM/HPC job scripts
    ├── run2.sh
    ├── run3.sh  / run3_lp.sh           # Runs with LPIPS loss
    ├── run4.sh  / run4_lp.sh
    └── pyproject.toml                  # Package config
```

---

## References

1. Fienup, J. R. (1982). Phase retrieval algorithms: a comparison. *Applied Optics*.
2. Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
3. Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
4. Metzler et al. (2018). prDeep: Robust Phase Retrieval with a Flexible Deep Network. *ICML*.
5. Young & Khare (2023). Vortex Illumination for Phase Retrieval.

---

*Department of Physics, IIT Delhi — B.Tech Project, November 2025*
