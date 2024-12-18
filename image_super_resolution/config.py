from pathlib import Path
import os

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
#logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

TRAIN_PATH = RAW_DATA_DIR / "div2k-dataset" / "versions/1" / "DIV2K_train_HR" / "DIV2K_train_HR"
VALID_PATH = RAW_DATA_DIR / "div2k-dataset" / "versions/1" / "DIV2K_valid_HR" / "DIV2K_valid_HR"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NUM_WORKERS = os.cpu_count()