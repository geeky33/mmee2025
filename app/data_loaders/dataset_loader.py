"""
Dataset discovery and management.
Handles finding and organizing datasets under the data/ directory.
"""

from pathlib import Path
from typing import Dict, List
try:
    # Prefer absolute import when running as a script or when package root is set
    from app.config.settings import IMAGE_EXTENSIONS
except ImportError:
    try:
        # Fallback to relative import when executed as a proper package module
        from ..config.settings import IMAGE_EXTENSIONS
    except ImportError:
        # Final fallback: sensible default extensions to keep functionality working
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}


def discover_datasets(data_root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Discover all valid datasets under the data root directory.
    A valid dataset must have: images/ folder, captions.json, labels.json
    
    Args:
        data_root: Path to the data directory
        
    Returns:
        Dictionary mapping dataset names to their component paths
        {
            "dataset_name": {
                "root": Path,
                "images": Path,
                "captions": Path,
                "labels": Path
            }
        }
    """
    datasets = {}
    
    if not data_root.exists():
        return datasets
    
    for dataset_dir in sorted(data_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        images_dir = dataset_dir / "images"
        captions_file = dataset_dir / "captions.json"
        labels_file = dataset_dir / "labels.json"
        
        # Check if all required components exist
        if images_dir.exists() and captions_file.exists() and labels_file.exists():
            datasets[dataset_dir.name] = {
                "root": dataset_dir,
                "images": images_dir,
                "captions": captions_file,
                "labels": labels_file,
            }
    
    return datasets


def scan_images(images_dir: Path) -> List[Path]:
    """
    Recursively scan a directory for image files.
    
    Args:
        images_dir: Path to the images directory
        
    Returns:
        Sorted list of image file paths
    """
    image_paths = [
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def get_noise_files(captions_file: Path) -> tuple:
    """
    Get paths to noise-related files if they exist.
    
    Args:
        captions_file: Path to the original captions.json
        
    Returns:
        Tuple of (captions_with_noise.json path, bad_caption_gt.json path)
    """
    captions_noise = captions_file.with_name("captions_with_noise.json")
    bad_gt = captions_file.with_name("bad_caption_gt.json")
    return captions_noise, bad_gt   