"""
Configuration settings for MMEE Embedding Viewer.
All constants and configuration parameters are centralized here.
"""

from pathlib import Path

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Go up to project root
APP_DIR = PROJECT_ROOT / "app"
DATA_ROOT = PROJECT_ROOT / "data"

# ============================================================================
# MODEL DEFAULTS
# ============================================================================
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"
DEFAULT_BATCH_SIZE = 64

AVAILABLE_MODELS = ["ViT-B-32", "ViT-L-14", "ViT-L-14-336"]
AVAILABLE_WEIGHTS = ["openai", "laion2b_s34b_b79k", "laion2b_s32b_b82k"]

# ============================================================================
# PROJECTION SETTINGS
# ============================================================================
PROJECTION_METHODS = ["PCA", "tSNE", "UMAP"]
DEFAULT_METHOD = "PCA"
DEFAULT_RANDOM_STATE = 42

# tSNE parameters
DEFAULT_TSNE_PERPLEXITY = 30
TSNE_PERPLEXITY_RANGE = (5, 60)

# UMAP parameters
DEFAULT_UMAP_NEIGHBORS = 30
UMAP_NEIGHBORS_RANGE = (5, 100)
DEFAULT_UMAP_MIN_DIST = 0.1

# ============================================================================
# DETECTION SETTINGS
# ============================================================================
OUTLIER_METHODS = [
    "Isolation Forest",
    "kNN Distance (Quantile)",
    "LOF (Local Outlier Factor)",
    "DBSCAN (noise)"
]

DEFAULT_CONTAMINATION = 0.03
DEFAULT_KNN_K = 10
DEFAULT_KNN_QUANTILE = 0.98
DEFAULT_LOF_K = 20
DEFAULT_DBSCAN_EPS = 0.8
DEFAULT_DBSCAN_MIN_SAMPLES = 10

# ============================================================================
# VALIDATION & EVALUATION
# ============================================================================
DEFAULT_TARGET_RECALL = 0.70
DEFAULT_MIN_PRECISION = 0.20
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_CV_FOLDS = 5

# ============================================================================
# UI SETTINGS
# ============================================================================
PAGE_TITLE = "MMEE • Embedding Viewer"
PAGE_LAYOUT = "wide"

# Plot settings
DEFAULT_OUTLIER_SIZE = 10
DEFAULT_OUTLIER_WIDTH = 2
OUTLIER_SIZE_RANGE = (6, 20)
OUTLIER_WIDTH_RANGE = (1, 5)

# Sampling defaults
DEFAULT_MAX_CLASSES = 30
DEFAULT_SAMPLES_PER_CLASS = 30
MAX_SAMPLES_UPPER_LIMIT = 500

# ============================================================================
# FILE EXTENSIONS
# ============================================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ============================================================================
# COLOR PALETTE
# ============================================================================
def generate_class_colors(class_names: list) -> dict:
    """
    Generate HSL colors for class names.
    
    Args:
        class_names: List of class name strings
        
    Returns:
        Dictionary mapping class names to HSL color strings
    """
    sorted_names = sorted(class_names)
    n = max(1, len(sorted_names))
    return {
        name: f"hsl({int(360 * i / n)}, 70%, 50%)"
        for i, name in enumerate(sorted_names)
    }