"""
Data validation and train/validation split utilities.
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Tuple


def frozen_validation_split(
    df: pd.DataFrame,
    frac: float,
    key_cols: Tuple[str, ...] = ("image_relpath", "caption_short"),
    seed_str: str = "MMEE_v1",
) -> np.ndarray:
    """
    Create a deterministic validation split by hashing key columns.
    This ensures the same rows are always selected for validation,
    even across different runs.
    
    Args:
        df: DataFrame to split
        frac: Fraction of data to use for validation (0.0 to 1.0)
        key_cols: Tuple of column names to use for hashing
        seed_str: Seed string for hash reproducibility
        
    Returns:
        Boolean mask array where True indicates validation samples
    """
    n = len(df)
    
    if frac <= 0:
        return np.zeros(n, dtype=bool)
    if frac >= 1:
        return np.ones(n, dtype=bool)

    def row_key(i: int) -> str:
        """Build a stable per-row key from specified columns."""
        parts = [seed_str]
        for col in key_cols:
            if col in df.columns:
                parts.append(str(df.iloc[i][col]))
            else:
                parts.append("")
        return "|".join(parts)

    # Hash each row and convert to 32-bit integers
    hash_values = np.array([
        int(hashlib.sha1(row_key(i).encode("utf-8")).hexdigest()[:8], 16)
        for i in range(n)
    ], dtype=np.uint32)

    # Map hashes to [0, 1) range and threshold by fraction
    random_values = (hash_values % 100_000_003) / 100_000_003.0
    return random_values < float(frac)


def validate_embedding_shapes(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    n_images: int,
    n_texts_expected: int
) -> bool:
    """
    Validate that embedding shapes match expected dimensions.
    
    Args:
        img_emb: Image embeddings array
        txt_emb: Text embeddings array
        n_images: Expected number of images
        n_texts_expected: Expected number of text embeddings
        
    Returns:
        True if shapes are valid, False otherwise
    """
    if img_emb.shape[0] != n_images:
        return False
    if txt_emb.shape[0] != n_texts_expected:
        return False
    return True


def sample_dataframe_per_class(
    df: pd.DataFrame,
    samples_per_class: int,
    class_column: str = "class_name",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample a fixed number of rows per class.
    
    Args:
        df: Input DataFrame
        samples_per_class: Number of samples to take per class (0 = all)
        class_column: Name of the class column
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    if samples_per_class <= 0:
        return df
    
    return df.groupby(class_column, group_keys=False).apply(
        lambda g: g.sample(min(len(g), samples_per_class), random_state=random_state)
    ).reset_index(drop=True)