from pathlib import Path
import typer
from loguru import logger
import gc
from tqdm import tqdm

from image_super_resolution.config import MODELS_DIR, TRAIN_PATH, VALID_PATH, PROJ_ROOT
import torch
import os
from cnn import LaplacianPyramidAttentionNetwork
from image_super_resolution.dataset import get_loader, get_filenames_list, get_parameters
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# Initialize Typer application
app = typer.Typer()

# Custom loss function based on DSSIM (Structural Similarity)
class DSSIMLoss(nn.Module):
    def __init__(self, data_range: float, device: torch.device) -> None:
        """
        Initializes the DSSIM Loss with the specified data range and device.
        
        Args:
            data_range (float): The data range for computing SSIM (usually 255 for images).
            device (torch.device): The device (CPU or GPU) to which the loss function is moved.
        """
        super(DSSIMLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the DSSIM loss between the predicted and target images.
        
        Args:
            preds (torch.Tensor): The predicted super-resolved images.
            target (torch.Tensor): The ground-truth target images.
        
        Returns:
            torch.Tensor: The computed DSSIM loss (higher SSIM means lower loss).
        """
        return (1 - self.ssim(preds, target)) / 2

    def compute_ssim(self, preds: torch.Tensor, target: torch.Tensor) -> float:
        """
        Computes SSIM value between predicted and target images.
        
        Args:
            preds (torch.Tensor): The predicted super-resolved images.
            target (torch.Tensor): The ground-truth target images.
        
        Returns:
            float: The computed SSIM score.
        """
        return self.ssim(preds, target).item()
    

def device_clear_memory(device: torch.device):
    """Free up memory resources by clearing caches."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "cpu":
        torch.mps.empty_cache()

# Function to save model checkpoint
def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: Path) -> None:
    """
    Saves the model and optimizer states along with the current epoch.
    
    Args:
        model (nn.Module): The model to be saved.
        optimizer (optim.Optimizer): The optimizer to be saved.
        epoch (int): The current epoch number.
        path (Path): The path to save the checkpoint file.
    
    Returns:
        None
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, path)

# Function to load model checkpoint
def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: Path) -> int:
    """
    Loads a model checkpoint if it exists, returns the starting epoch.
    
    Args:
        model (nn.Module): The model to load the checkpoint into.
        optimizer (optim.Optimizer): The optimizer to load the checkpoint into.
        path (Path): The path to the checkpoint file.
    
    Returns:
        int: The starting epoch for training.
    """
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    return start_epoch

# Learning rate lambda function based on epoch
def lr_lambda(epoch: int, threshold: int, high: float, low: float) -> float:
    """
    Adjusts the learning rate based on the epoch.
    
    Args:
        epoch (int): The current epoch.
        threshold (int): The epoch threshold where the learning rate changes.
        high (float): The high learning rate before the threshold.
        low (float): The low learning rate after the threshold.
    
    Returns:
        float: The learning rate based on the epoch.
    """
    return high if epoch < threshold else low

# Training function for the model
@app.command()
def train_model(
    config_path: Path = typer.Option(
        PROJ_ROOT / "image_super_resolution" / "modeling" / "parameters.yml",
        help="Path to the configuration file."
    ),
    save_path: Path = typer.Option(
        MODELS_DIR / "model.pth",
        help="Path to save the trained model."
    )
) -> None:
    """
    Trains the model on the training dataset and validates on the validation dataset.
    
    Args:
        config_path (Path): Path to the configuration file.
        save_path (Path): Path where the trained model should be saved.
    
    Returns:
        None
    """
    hyp = get_parameters(config_path)  # Load model parameters from config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set the device to GPU or CPU

    # Initialize model with configuration parameters
    model = LaplacianPyramidAttentionNetwork(
        in_channels=hyp['model']['in_channels'],
        out_channels=hyp['model']['out_channels']
    ).to(device)

    # Optimizer setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(hyp['training']['learning_rate']),
        betas=tuple(hyp['training']['betas']),
        eps=float(hyp['training']['epsilon'])
    )

    # Loss function setup using DSSIM
    dssim = DSSIMLoss(data_range=hyp['training']['data_range'], device=device)

    # Learning rate scheduler
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_lambda(
            epoch,
            hyp['training']['lr_lambda_epoch_threshold'],
            hyp['training']['lr_lambda_high'],
            hyp['training']['lr_lambda_low']
        )
    )

    # Load dataset paths and create data loaders
    dataset_paths = get_filenames_list()
    train_loader = get_loader(hyp, TRAIN_PATH, dataset_paths['train'])
    valid_loader = get_loader(hyp, VALID_PATH, dataset_paths['valid'])

    num_epochs = hyp['training']['num_epochs']
    patience = hyp['training']['patience']
    start_epoch = load_checkpoint(model, optimizer, save_path)  # Load checkpoint and get start epoch
    best_ssim = 0.0  # Best validation SSIM
    patience_counter = 0  # Counter to stop training if no improvement

    logger.info("Starting model training...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_ssim = 0.0
        
        # Training loop
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
            for batch in train_loader:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)

                optimizer.zero_grad()
                sr_images = model(images)
                loss = dssim(sr_images, targets)
                ssim_score = dssim.compute_ssim(sr_images, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_ssim += ssim_score * images.size(0)

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # Average loss and SSIM for training
        train_loss = train_loss / len(train_loader.dataset)
        train_ssim = train_ssim / len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_ssim = 0.0

        with tqdm(total=len(valid_loader), desc=f"Validating Epoch {epoch + 1}", unit="batch") as pbar:
            with torch.no_grad():
                for batch in valid_loader:
                    val_images, val_targets = batch
                    val_images, val_targets = val_images.to(device), val_targets.to(device)

                    sr_images = model(val_images)
                    loss = dssim(sr_images, val_targets)
                    val_loss += loss.item() * val_images.size(0)
                    val_ssim += dssim.compute_ssim(sr_images, val_targets) * val_images.size(0)

                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

        # Average loss and SSIM for validation
        val_loss = val_loss / len(valid_loader.dataset)
        val_ssim = val_ssim / len(valid_loader.dataset)

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train SSIM: {train_ssim:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val SSIM: {val_ssim:.4f}")

        # Save model if validation SSIM improves
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, save_path)
            logger.info(f"Validation SSIM improved to {val_ssim:.4f}. Model saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation SSIM for {patience_counter} epochs.")

        # Early stopping if no improvement for 'patience' epochs
        if patience_counter >= patience:
            logger.info(f"Patience of {patience} epochs reached. Stopping training.")
            break

        scheduler.step()  # Update the learning rate

        device_clear_memory(device)

    logger.success("Modeling training complete.")
    device_clear_memory(device)

# Entry point for the script (runs the CLI app)
if __name__ == "__main__":
    app()  # Run the Typer app