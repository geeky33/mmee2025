"""
Caption loading and processing utilities.
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def read_captions(captions_path: Path) -> Dict[str, List[str]]:
    """
    Read captions from a JSON file.
    Handles both pandas-readable and standard JSON formats.
    
    Args:
        captions_path: Path to captions.json file
        
    Returns:
        Dictionary mapping image identifiers to lists of caption strings
    """
    if not captions_path.exists():
        return {}
    
    try:
        # Try pandas JSON reading first
        obj = pd.read_json(captions_path, typ="series")
        captions_dict = {}
        for key, value in obj.items():
            if isinstance(value, (list, tuple)):
                captions_dict[str(key)] = list(value)
            elif pd.notna(value):
                captions_dict[str(key)] = [str(value)]
            else:
                captions_dict[str(key)] = []
        return captions_dict
        
    except ValueError:
        # Fall back to standard JSON reading
        with open(captions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        captions_dict = {}
        for key, value in data.items():
            if isinstance(value, list):
                captions_dict[str(key)] = [str(x) for x in value]
            elif value is None:
                captions_dict[str(key)] = []
            else:
                captions_dict[str(key)] = [str(value)]
        
        return captions_dict


def align_captions_to_images(
    image_paths: List[Path],
    captions_map: Dict[str, List[str]],
    images_root: Path
) -> List[List[str]]:
    """
    Align captions to a list of image paths.
    Tries multiple key formats: relative path, filename, stem.
    
    Args:
        image_paths: List of image Path objects
        captions_map: Dictionary from read_captions()
        images_root: Root directory for computing relative paths
        
    Returns:
        List of caption lists, one per image (empty list if no captions found)
    """
    aligned_captions = []
    
    for img_path in image_paths:
        # Try different key formats
        try:
            rel_path = str(img_path.relative_to(images_root))
        except Exception:
            rel_path = None
        
        # Check in order: relative path, filename, stem
        for key in (rel_path, img_path.name, img_path.stem):
            if key and key in captions_map:
                # Filter out empty captions
                caps = [c for c in captions_map[key] if c]
                aligned_captions.append(caps)
                break
        else:
            # No captions found for this image
            aligned_captions.append([])
    
    return aligned_captions


def load_bad_caption_ground_truth(bad_gt_path: Path) -> Dict[str, List[int]]:
    """
    Load ground truth for bad captions.
    
    Args:
        bad_gt_path: Path to bad_caption_gt.json
        
    Returns:
        Dictionary mapping image identifiers to lists of 0/1 flags
    """
    if not bad_gt_path.exists():
        return {}
    
    try:
        with open(bad_gt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}