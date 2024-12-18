from pathlib import Path
from typing import List, Tuple, Dict, Any
import os
import yaml
from PIL import Image

import random
import typer
from loguru import logger
from tqdm import tqdm

import torch
import json
from collections import defaultdict
from torch import Tensor
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from image_super_resolution.config import TRAIN_PATH, VALID_PATH, PROCESSED_DATA_DIR, NUM_WORKERS

# Define transformation for images: converting to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Define types for better readability
ConfigType = Dict[str, Any]
FilenamesType = List[str]

# Function to get the list of filenames for train, validation, and test datasets
def get_filenames_list(
    train_path: Path = TRAIN_PATH, valid_path: Path = VALID_PATH, json_path: Path = PROCESSED_DATA_DIR / "dataset_paths.json"
) -> defaultdict:
    # If the dataset paths JSON already exists, load it
    if json_path.exists():
        logger.info(f"Loading dataset paths from {json_path}...")
        with open(json_path, 'r') as f:
            dataset_paths = defaultdict(list, json.load(f))  # Load paths into defaultdict
    else:
        logger.info(f"Generating dataset paths and saving to {json_path}...")
        
        # List the filenames in the train directory
        filenames = os.listdir(train_path)
        
        # Split the filenames into train and test sets
        train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, random_state=42)
        
        # List the filenames in the valid directory
        valid_filenames = os.listdir(valid_path)
        
        # Create a defaultdict to store paths categorized by train, valid, and test
        dataset_paths = defaultdict(list)
        
        # Add the full paths of the train, valid, and test images to the respective lists
        dataset_paths["train"] = [str(train_path / filename) for filename in train_filenames]
        dataset_paths["valid"] = [str(valid_path / filename) for filename in valid_filenames]
        dataset_paths["test"] = [str(train_path / filename) for filename in test_filenames]
        
        # Save the dataset paths to a JSON file
        with open(json_path, 'w') as f:
            json.dump(dataset_paths, f, indent=4)

    return dataset_paths  # Return the dictionary with the dataset paths

# Function to load parameters from a YAML configuration file
def get_parameters(yml_path: Path) -> ConfigType:
    with open(yml_path, "r") as file:
        config = yaml.safe_load(file)  # Load YAML config file
    return config

# Dataset class for creating patches from the images
class PatchDataset(Dataset):
    def __init__(
        self, 
        img_dir: Path,  # Path to the image directory
        filenames: FilenamesType,  # List of filenames for the dataset
        crop_size: Tuple[int, int],  # Size of the cropped patches
        scale_factor: int,  # Factor for downsampling the images
        num_patches: int,  # Number of patches to generate per image
        transform: transforms.Compose = None  # Optional transformations to apply
    ):
        # Initialize the dataset with the necessary parameters
        self.img_dir = img_dir
        self.filenames = filenames
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.num_patches = num_patches
        self.transform = transform
        
        # Calculate the total number of patches in the dataset
        self.total_patches = len(filenames) * num_patches

    def __len__(self) -> int:
        # Return the total number of patches in the dataset
        return self.total_patches

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Get a specific patch by index
        image_idx = idx // self.num_patches  # Determine which image this patch belongs to
        patch_idx = idx % self.num_patches  # Get the specific patch number
        
        # Load the image
        img_path = os.path.join(self.img_dir, self.filenames[image_idx])
        img = Image.open(img_path)
        width, height = img.size
        
        # Randomly crop the image to the desired patch size
        left = random.randint(0, width - self.crop_size[0])
        top = random.randint(0, height - self.crop_size[1])
        right = left + self.crop_size[0]
        bottom = top + self.crop_size[1]
        
        img_cropped = img.crop((left, top, right, bottom))  # Crop the image
        
        # Downscale the cropped image based on the scale factor
        new_size = (self.crop_size[0] // self.scale_factor, self.crop_size[1] // self.scale_factor)
        img_downscaled = img_cropped.resize(new_size, Image.BICUBIC)
        
        # Apply transformations (if any)
        if self.transform:
            img_cropped = self.transform(img_cropped)
            img_downscaled = self.transform(img_downscaled)
        
        return img_downscaled, img_cropped  # Return the downscaled and cropped image patches

    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        # Function to combine the batch of patches into a single batch
        downscaled_patches, cropped_patches = zip(*batch)  # Unzip the list of patches
        downscaled_patches = torch.stack(downscaled_patches, dim=0)  # Stack the downscaled patches
        cropped_patches = torch.stack(cropped_patches, dim=0)  # Stack the cropped patches
    
        return downscaled_patches, cropped_patches  # Return the batched patches

# Function to create a DataLoader for the PatchDataset
def get_loader(
    config: ConfigType,  # Configuration parameters
    path: Path,  # Path to the image directory
    filenames: FilenamesType  # List of filenames to use
) -> DataLoader:
    # Create the PatchDataset
    dataset = PatchDataset(
        img_dir=path,
        filenames=filenames,
        crop_size=(config["dataset"]["image_size"], config["dataset"]["image_size"]),
        scale_factor=config["dataset"]["image_scale"],
        num_patches=config["dataset"]["num_patches"],
        transform=transform
    )

    # Return the DataLoader
    return DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],  # Set the batch size from the config
        num_workers=NUM_WORKERS,  # Set the number of workers for parallel data loading
        collate_fn=PatchDataset.collate_fn,  # Use the custom collate function
        shuffle=True  # Shuffle the data
    )