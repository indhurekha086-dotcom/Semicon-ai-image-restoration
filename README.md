# SEMICON India Hackathon 2026

## AI-Based Restoration of Degraded Images for Semiconductor Inspection

### Team Nova

Kalaignar Karunanidhi Institute of Technology (KIT), Coimbatore

---

## Problem Statement

This project addresses the KLA problem statement:

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

The goal is to restore degraded semiconductor inspection images affected by noise and reduced spatial resolution while preserving important image structures and details.

---

## Solution

We propose a degradation-robust **Multi-Scale Residual U-Net (MSR-UNet)** restoration model.

The training pipeline uses randomized combinations of:

- Gaussian noise
- Speckle noise
- Downsampling

The model learns a direct mapping from degraded images to clean images.

---

## Architecture

```text
Noisy / Low-Resolution Input
            ↓
     Shallow Features
            ↓
     Multi-Scale Encoder
            ↓
    Residual Bottleneck
            ↓
     Multi-Scale Decoder
            ↓
       Restored Image
