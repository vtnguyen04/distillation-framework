import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

def plot_results(csv_file: str, save_dir: str):
    """
    Plots training results from a CSV file.
    Expects columns like 'epoch', 'train_loss', 'val_loss', 'val_accuracy'.
    Generates 'results.png'.
    """
    try:
        df = pd.read_csv(csv_file)
        # Strip spaces from column names
        df.columns = df.columns.str.strip()

        save_path = Path(save_dir) / "results.png"

        # Determine subplots based on available metrics
        cols = [c for c in df.columns if c != 'epoch']
        n_cols = len(cols)

        if n_cols == 0:
            return

        fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 6, 5))
        if n_cols == 1: axs = [axs]

        for i, col in enumerate(cols):
            ax = axs[i]
            ax.plot(df['epoch'], df[col], marker='.', label=col)
            ax.set_title(col)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"📈 Results plotted to {save_path}")

    except Exception as e:
        print(f"⚠️ Failed to plot results: {e}")

def plot_images(images: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor,
                classes: List[str], save_path: str, max_images: int = 16):
    """
    Plots a mosaic of images with Ground Truth vs Prediction.
    """
    try:
        # Denormalize roughly for visualization (assuming ImageNet/CIFAR stds)
        # Just clamping to 0-1 range is often "good enough" for quick preview if roughly 0.5 mean
        images = images.cpu()

        # Approximate denorm (x * std + mean) - using CIFAR stats
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)

        batch_size = min(images.size(0), max_images)
        rows = int(np.sqrt(batch_size))
        cols = int(np.ceil(batch_size / rows))

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        axs = axs.flatten()

        for i in range(batch_size):
            img = images[i].permute(1, 2, 0).numpy()
            target_idx = targets[i].item()
            pred_idx = preds[i].item()

            label_true = classes[target_idx] if classes else str(target_idx)
            label_pred = classes[pred_idx] if classes else str(pred_idx)

            color = 'green' if target_idx == pred_idx else 'red'

            ax = axs[i]
            ax.imshow(img)
            ax.set_title(f"T: {label_true}\nP: {label_pred}", color=color, fontsize=10)
            ax.axis('off')

        # Hide empty plots
        for i in range(batch_size, len(axs)):
            axs[i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"🖼️ Inference preview saved to {save_path}")

    except Exception as e:
        print(f"⚠️ Failed to plot images: {e}")
