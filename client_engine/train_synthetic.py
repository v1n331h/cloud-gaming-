"""
============================================================
SuperStream AI — Synthetic ESPCN Training Script
============================================================
Generates synthetic training data (random noise + simple
geometric shapes), trains the ESPCN model for 5 quick epochs,
and saves weights to `espcn_fast.pth`.

This gives us a valid weights file to load into the inference
pipeline instantly — no dataset download required.

The model learns a basic identity-like mapping that passes
pixels through without destroying them. Real training on
DIV2K/Set5 would produce actual super-resolution, but this
is sufficient for pipeline integration testing.

Usage:
    python client_engine/train_synthetic.py
    python client_engine/train_synthetic.py --epochs 10 --upscale 2
============================================================
"""

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ── Local imports ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client_engine.model import ESPCN


# ─── Synthetic Data Generation ────────────────────────────────
def generate_synthetic_batch(
    batch_size: int,
    lr_h: int,
    lr_w: int,
    upscale_factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of synthetic LR/HR image pairs.

    Strategy:
      1. Create HR images (batch, 1, lr_h*r, lr_w*r) with random
         gradients, rectangles, and circles to give the model
         some spatial structure to learn from.
      2. Downscale to LR using area-average (simulated by
         F.interpolate with mode='area').
      3. Return (LR, HR) tensors normalized to [0, 1].

    This is NOT meant for real SR quality — it's a quick way to
    produce a valid .pth file with non-random weights.
    """
    hr_h = lr_h * upscale_factor
    hr_w = lr_w * upscale_factor

    hr_images = []
    for _ in range(batch_size):
        # Start with a smooth random gradient
        canvas = np.random.rand(hr_h, hr_w).astype(np.float32) * 0.3

        # Add horizontal gradient
        grad_h = np.linspace(0.0, 0.7, hr_w, dtype=np.float32)
        canvas += grad_h[np.newaxis, :]

        # Add vertical gradient
        grad_v = np.linspace(0.0, 0.5, hr_h, dtype=np.float32)
        canvas += grad_v[:, np.newaxis]

        # Draw random rectangles (simulates edges/textures)
        for _ in range(np.random.randint(3, 8)):
            y1 = np.random.randint(0, hr_h - 10)
            x1 = np.random.randint(0, hr_w - 10)
            y2 = min(y1 + np.random.randint(10, hr_h // 4), hr_h)
            x2 = min(x1 + np.random.randint(10, hr_w // 4), hr_w)
            intensity = np.random.rand() * 0.8 + 0.2
            canvas[y1:y2, x1:x2] = intensity

        # Draw random circles (simulates smooth edges)
        for _ in range(np.random.randint(2, 5)):
            cy = np.random.randint(0, hr_h)
            cx = np.random.randint(0, hr_w)
            radius = np.random.randint(5, min(hr_h, hr_w) // 6)
            yy, xx = np.ogrid[:hr_h, :hr_w]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            canvas[mask] = np.random.rand() * 0.6 + 0.4

        # Clamp to [0, 1]
        canvas = np.clip(canvas, 0.0, 1.0)
        hr_images.append(canvas)

    # Stack into tensors: (B, 1, H, W)
    hr_tensor = torch.from_numpy(np.stack(hr_images)).unsqueeze(1)

    # Downscale HR → LR using area interpolation
    lr_tensor = torch.nn.functional.interpolate(
        hr_tensor, size=(lr_h, lr_w), mode="area"
    )

    return lr_tensor, hr_tensor


# ─── Training Loop ────────────────────────────────────────────
def train(
    upscale_factor: int = 3,
    epochs: int = 5,
    batch_size: int = 16,
    lr_h: int = 64,
    lr_w: int = 64,
    learning_rate: float = 1e-3,
    output_path: str = "espcn_fast.pth",
):
    """Train ESPCN on synthetic data and save weights."""

    print("=" * 60)
    print("⚡ SuperStream AI — Synthetic ESPCN Trainer")
    print("=" * 60)

    # ── Device selection ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 Training on: CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Training on: Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Training on: CPU")

    # ── Create model ──
    model = ESPCN(upscale_factor=upscale_factor, num_channels=1)
    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 ESPCN | {upscale_factor}× | {total_params:,} params")
    print(f"📐 LR size: {lr_w}×{lr_h} → HR size: {lr_w*upscale_factor}×{lr_h*upscale_factor}")
    print(f"📦 Batch size: {batch_size} | Epochs: {epochs}")
    print(f"📈 Learning rate: {learning_rate}")
    print("─" * 60)

    # ── Loss + optimizer ──
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ── Training loop ──
    total_start = time.perf_counter()
    batches_per_epoch = 10  # 10 batches of synthetic data per epoch

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_start = time.perf_counter()

        for batch_idx in range(batches_per_epoch):
            # Generate fresh synthetic data each batch
            lr_batch, hr_batch = generate_synthetic_batch(
                batch_size, lr_h, lr_w, upscale_factor
            )
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            # Forward pass
            sr_batch = model(lr_batch)

            # Ensure SR output matches HR dimensions
            if sr_batch.shape != hr_batch.shape:
                hr_batch = torch.nn.functional.interpolate(
                    hr_batch, size=sr_batch.shape[2:], mode="bilinear",
                    align_corners=False
                )

            loss = criterion(sr_batch, hr_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / batches_per_epoch
        epoch_ms = (time.perf_counter() - epoch_start) * 1000

        print(
            f"  Epoch [{epoch}/{epochs}] │ "
            f"Loss: {avg_loss:.6f} │ "
            f"Time: {epoch_ms:.0f}ms"
        )

    total_time = time.perf_counter() - total_start
    print("─" * 60)
    print(f"✅ Training complete in {total_time:.2f}s")

    # ── Save weights ──
    # Resolve output path relative to this script's directory
    output_file = Path(__file__).resolve().parent / output_path
    torch.save(model.state_dict(), output_file)
    size_kb = output_file.stat().st_size / 1024
    print(f"💾 Saved weights: {output_file}")
    print(f"   File size: {size_kb:.1f} KB")
    print("=" * 60)

    return str(output_file)


# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ESPCN on synthetic data for pipeline testing"
    )
    parser.add_argument("--upscale", type=int, default=3, choices=[2, 3],
                        help="Upscale factor (default: 3)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--output", type=str, default="espcn_fast.pth",
                        help="Output weights filename (default: espcn_fast.pth)")
    args = parser.parse_args()

    train(
        upscale_factor=args.upscale,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_path=args.output,
    )
