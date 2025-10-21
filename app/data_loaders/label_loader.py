"""
Label loading and processing utilities.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def derive_class_from_folder(image_path: Path, images_root: Path) -> str:
    """
    Derive class name from folder structure or filename.
    
    Args:
        image_path: Path to the image file
        images_root: Root images directory
        
    Returns:
        Inferred class name string
    """
    try:
        rel_path = image_path.relative_to(images_root)
        # Use first folder in path as class name
        if rel_path.parts and len(rel_path.parts) > 1:
            return rel_path.parts[0]
    except Exception:
        pass
    
    # Fall back to first part of filename before underscore
    return image_path.stem.split("_")[0]


def read_labels(
    labels_path: Path,
    image_paths: List[Path],
    images_root: Path
) -> pd.DataFrame:
    """
    Read labels from labels.json and create a DataFrame.
    Handles both list and dict formats in the JSON file.
    
    Args:
        labels_path: Path to labels.json file
        image_paths: List of all image paths in the dataset
        images_root: Root directory for images
        
    Returns:
        DataFrame with columns: image_path, image_relpath, class_name, class_id
    """
    records = []
    
    # Create lookup dictionaries for fast access
    by_stem = {p.stem: p for p in image_paths}
    try:
        by_rel = {str(p.relative_to(images_root)): p for p in image_paths}
    except Exception:
        by_rel = {}
    
    used_paths = set()
    
    # Load labels file if it exists
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List format: each item is a dict with image info
            records.extend(_process_list_format(
                data, by_rel, by_stem, images_root, used_paths
            ))
        
        elif isinstance(data, dict):
            # Dict format: keys are image identifiers
            records.extend(_process_dict_format(
                data, by_rel, by_stem, images_root, used_paths
            ))
    
    # Fill in unlabeled images with folder-derived classes
    for img_path in image_paths:
        if img_path not in used_paths:
            class_name = derive_class_from_folder(img_path, images_root)
            records.append({
                "image_path": str(img_path),
                "image_relpath": _get_relpath(img_path, images_root),
                "class_name": class_name,
                "class_id": None,
            })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    if df.empty:
        raise RuntimeError("No labeled images found.")
    
    # Assign integer class IDs if missing
    if df["class_id"].isna().any():
        name_to_id = {
            name: idx + 1
            for idx, name in enumerate(sorted(df["class_name"].unique()))
        }
        df["class_id"] = df["class_name"].map(name_to_id)
    
    df["class_id"] = df["class_id"].astype(int)
    return df


def _process_list_format(
    data: List[Dict],
    by_rel: Dict,
    by_stem: Dict,
    images_root: Path,
    used_paths: set
) -> List[Dict[str, Any]]:
    """Process labels in list format."""
    records = []
    
    for row in data:
        # Try different key names for image identifier
        key = (
            row.get("image_relpath") or
            row.get("image_path") or
            row.get("path") or
            row.get("file")
        )
        if not key:
            continue
        
        # Find matching image path
        img_path = by_rel.get(key) or by_stem.get(Path(key).stem)
        if not img_path:
            continue
        
        class_id = row.get("class_id")
        class_name = row.get("class_name") or derive_class_from_folder(
            img_path, images_root
        )
        
        records.append({
            "image_path": str(img_path),
            "image_relpath": _get_relpath(img_path, images_root),
            "class_name": class_name,
            "class_id": int(class_id) if class_id is not None else None,
        })
        used_paths.add(img_path)
    
    return records


def _process_dict_format(
    data: Dict,
    by_rel: Dict,
    by_stem: Dict,
    images_root: Path,
    used_paths: set
) -> List[Dict[str, Any]]:
    """Process labels in dict format."""
    records = []
    
    for key, meta in data.items():
        # Find matching image path
        img_path = by_rel.get(key) or by_stem.get(Path(key).stem)
        if not img_path:
            continue
        
        class_id = None
        
        if isinstance(meta, list):
            # List of class names
            classes = [str(x) for x in meta if str(x).strip()]
            class_name = classes[0] if classes else "unlabeled"
        elif isinstance(meta, dict):
            # Dict with class_id and/or class_name
            class_id = meta.get("class_id")
            class_name = meta.get("class_name") or "unlabeled"
        else:
            # Single value
            class_name = str(meta) if meta else "unlabeled"
        
        records.append({
            "image_path": str(img_path),
            "image_relpath": _get_relpath(img_path, images_root),
            "class_name": class_name,
            "class_id": int(class_id) if class_id is not None else None,
        })
        used_paths.add(img_path)
    
    return records


def _get_relpath(img_path: Path, images_root: Path) -> str:
    """Get relative path string, falling back to name if not relative."""
    try:
        if img_path.is_relative_to(images_root):
            return str(img_path.relative_to(images_root))
    except Exception:
        pass
    return img_path.name