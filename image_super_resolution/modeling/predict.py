import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from pathlib import Path
from image_super_resolution.dataset import get_loader, get_parameters, get_filenames_list
from loguru import logger

from image_super_resolution.config import PROJ_ROOT, TRAIN_PATH, MODELS_DIR
from image_super_resolution.dataset import get_loader  # Assuming you have a function to get the test loader
from cnn import LaplacianPyramidAttentionNetwork  # Assuming the model is defined here

def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    return model

# Typer CLI app
import typer
app = typer.Typer()

@app.command()
def main(
    config_path: Path = typer.Option(
        PROJ_ROOT / "image_super_resolution" / "modeling" / "parameters.yml",
        help="Path to the configuration file."
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "model.pth",
        help="Path to save the trained model."
    )
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyp = get_parameters(config_path)

    model = LaplacianPyramidAttentionNetwork(
        in_channels=hyp['model']['in_channels'],
        out_channels=hyp['model']['out_channels']
    ).to(device)
    model = load_model_checkpoint(model, model_path, device)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    model.eval()

    dataset_paths = get_filenames_list()
    test_loader = get_loader(hyp, TRAIN_PATH, dataset_paths['test'])

    total_ssim = 0.0
    num_elements = 0

    with tqdm(total=len(test_loader), desc="Evaluating on Test Set", unit="batch") as pbar:
        for images, targets in test_loader:
            with torch.no_grad():
                predicted_images = model(images.to(device))

                ssim_value = ssim_metric(predicted_images.to(device), targets.to(device))
                total_ssim += ssim_value.item()
                num_elements += 1

                pbar.set_postfix(ssim=ssim_value.item())
                pbar.update(1)


    avg_ssim = total_ssim / num_elements
    logger.info(f"Average SSIM on Test Set: {avg_ssim:.4f}")

if __name__ == "__main__":
    app()