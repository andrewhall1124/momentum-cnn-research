from models import Returns1DCNN
from torch import optim
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import datetime as dt
from datasets import DailyReturnsDataset
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss_curves(loss_history: dict, save_path: str = 'loss_curves.png',
                     show_batch_losses: bool = True, moving_avg_window: int = 50):
    """
    Plot and save training and validation loss curves.

    Args:
        loss_history: Dictionary containing loss data with keys:
            - 'train_losses': Per-batch training losses
            - 'val_losses': Per-batch validation losses
            - 'epoch_train_losses': Average training loss per epoch
            - 'epoch_val_losses': Average validation loss per epoch
        save_path: Path to save the plot image
        show_batch_losses: If True, show per-batch losses with moving average
        moving_avg_window: Window size for moving average smoothing
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Per-batch losses with moving average
    if show_batch_losses and loss_history['train_losses']:
        ax = axes[0]
        train_losses = loss_history['train_losses']
        val_losses = loss_history['val_losses']

        # Plot raw batch losses (semi-transparent)
        ax.plot(train_losses, alpha=0.3, color='blue', linewidth=0.5, label='Train (raw)')
        ax.plot(range(len(train_losses), len(train_losses) + len(val_losses)),
                val_losses, alpha=0.3, color='orange', linewidth=0.5, label='Val (raw)')

        # Calculate and plot moving averages
        if len(train_losses) >= moving_avg_window:
            train_ma = np.convolve(train_losses, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            ax.plot(range(moving_avg_window-1, len(train_losses)), train_ma,
                   color='blue', linewidth=2, label=f'Train (MA-{moving_avg_window})')

        if len(val_losses) >= moving_avg_window:
            val_ma = np.convolve(val_losses, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            ax.plot(range(len(train_losses) + moving_avg_window-1, len(train_losses) + len(val_losses)),
                   val_ma, color='orange', linewidth=2, label=f'Val (MA-{moving_avg_window})')

        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Per-Batch Losses (with Moving Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No batch losses to display',
                    ha='center', va='center', transform=axes[0].transAxes)

    # Plot 2: Per-epoch average losses
    ax = axes[1]
    if loss_history['epoch_train_losses']:
        epochs = range(1, len(loss_history['epoch_train_losses']) + 1)
        ax.plot(epochs, loss_history['epoch_train_losses'],
               marker='o', linewidth=2, markersize=8, label='Train', color='blue')
        ax.plot(epochs, loss_history['epoch_val_losses'],
               marker='s', linewidth=2, markersize=8, label='Validation', color='orange')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss')
        ax.set_title('Per-Epoch Average Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value annotations
        for i, (train_loss, val_loss) in enumerate(zip(loss_history['epoch_train_losses'],
                                                       loss_history['epoch_val_losses']), 1):
            ax.annotate(f'{train_loss:.4f}', (i, train_loss),
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            ax.annotate(f'{val_loss:.4f}', (i, val_loss),
                       textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No epoch losses to display',
               ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curves saved to {save_path}")
    plt.close()

def train(date_: dt.date,
          save_dir: str = 'checkpoints',
          batch_size: int = 64,
          num_workers: int = 4,
          max_epochs: int = 10,
          learning_rate: float = 1e-4,
          early_stopping_patience: int = 3):
    """
    Train the Returns1DCNN model.

    Args:
        date_: Target date for training
        save_dir: Directory to save checkpoints
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Number of epochs to wait for improvement before stopping
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")
    print(f"Max epochs: {max_epochs}, Learning rate: {learning_rate}")
    print(f"Early stopping patience: {early_stopping_patience}")

    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    train_start, train_end = date_ - dt.timedelta(days=5 * 365), date_ - dt.timedelta(days=91)
    val_start, val_end = date_ - dt.timedelta(days=90), date_

    print(f"Loading training data: {train_start} to {train_end}")
    train_dataset = DailyReturnsDataset().load(start=train_start, end=train_end)
    print(f"Loading validation data: {val_start} to {val_end}")
    val_dataset = DailyReturnsDataset().load(start=val_start, end=val_end)
                         
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffling improves training
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )

    model = Returns1DCNN().to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    # Track all losses for plotting
    train_losses = []  # Per-batch training losses
    val_losses = []    # Per-batch validation losses
    epoch_train_losses = []  # Average training loss per epoch
    epoch_val_losses = []    # Average validation loss per epoch

    print(f"\nStarting training for up to {max_epochs} epochs...")
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss
            train_batches += 1
            train_losses.append(batch_loss)  # Track per-batch loss
            pbar.set_postfix({'loss': f'{batch_loss:.6f}'})

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        epoch_train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]  ")
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                batch_loss = loss.item()
                val_loss += batch_loss
                val_batches += 1
                val_losses.append(batch_loss)  # Track per-batch loss
                pbar.set_postfix({'loss': f'{batch_loss:.6f}'})

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        epoch_val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save best model and check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            checkpoint_path = f"{save_dir}/best_model_{date_.strftime('%Y%m%d')}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"✓ Saved new best model to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  Best validation loss: {best_val_loss:.6f} (at epoch {best_epoch})")
                break

    if epochs_without_improvement < early_stopping_patience:
        print(f"\nTraining complete! Finished all {max_epochs} epochs")
    print(f"Best validation loss: {best_val_loss:.6f} (at epoch {best_epoch})")

    # Return model and all loss history
    loss_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_train_losses': epoch_train_losses,
        'epoch_val_losses': epoch_val_losses
    }

    return model, best_val_loss, loss_history
        

if __name__ == '__main__':
    date_ = dt.date(2024, 1, 1)
    model, best_val_loss, loss_history = train(date_)

    # Generate and save loss curves
    os.makedirs("charts", exist_ok=True)
    plot_loss_curves(
        loss_history,
        save_path=f'charts/loss_curves_{date_.strftime("%Y%m%d")}.png',
        show_batch_losses=True,
        moving_avg_window=50
    )