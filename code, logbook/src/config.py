"""
config.py

Centralized configuration file for the Kinematic IMU Analysis Pipeline.
All paths and core parameters are defined here to ensure modularity
and ease of reconfiguration across the entire project.
"""
from pathlib import Path

# ==========================================
# 1. DIRECTORY PATHS
# ==========================================
# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"
SRC_DIR = BASE_DIR / "src"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PREPARED_DATA_DIR = DATA_DIR / "prepared"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROCESSED_DATA_DIR / "experiments"
STAGE2_DATA_DIR = PROCESSED_DATA_DIR / "stage2"

# Report subdirectories
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure essential output directories exist
for directory in [PREPARED_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENTS_DIR, STAGE2_DATA_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. PIPELINE PARAMETERS
# ==========================================
# Preprocessing Parameters
SMOOTHING_WINDOW_S = "0.5s"
GAP_THRESHOLD_S = 0.300

# Segmentation Parameters
WINDOW_PRE_S = 2.0
WINDOW_POST_S = 2.0
NEGATIVE_STRIDE_S = 5.0
NEGATIVE_EXCLUSION_S = 10.0
USABLE_THRESHOLD = 0.8