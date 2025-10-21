"""
General utility functions for the MMEE application.
Includes normalization, similarity calculations, and hashing.
"""

import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2 normalize rows of a matrix.
    
    Args:
        X: Input array of shape (N, D)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        L2-normalized array of shape (N, D)
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (X / norms).astype(np.float32, copy=False)


def rowwise_cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute row-wise cosine similarity between two matrices.
    
    Args:
        A: First matrix of shape (N, D)
        B: Second matrix of shape (N, D)
        
    Returns:
        Array of cosine similarities of shape (N,)
    """
    A_norm = l2_normalize(A)
    B_norm = l2_normalize(B)
    return (A_norm * B_norm).sum(axis=1).astype(np.float32, copy=False)


def signature_for_embeddings(
    df: pd.DataFrame,
    dataset: str,
    model: str,
    weights: str,
    agg_tag: str
) -> str:
    """
    Generate a unique signature hash for embedding configuration.
    Used for caching and invalidation.
    
    Args:
        df: DataFrame with image paths
        dataset: Dataset name
        model: Model name
        weights: Pretrained weights name
        agg_tag: Aggregation tag (e.g., 'avg', 'first', 'percap_5')
        
    Returns:
        SHA1 hex digest string
    """
    payload = {
        "dataset": dataset,
        "model": model,
        "weights": weights,
        "agg": agg_tag,
        "paths": (
            df["image_relpath"].tolist()
            if "image_relpath" in df.columns
            else df["image_path"].tolist()
        ),
    }
    json_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(json_str.encode("utf-8")).hexdigest()


def ensure_2d_array(X: np.ndarray) -> np.ndarray:
    """
    Ensure input is a 2D array.
    
    Args:
        X: Input array
        
    Returns:
        2D array of shape (N, D)
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    return X


def joint_weighted_embeddings(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Blend image and text embeddings with weighted average.
    
    Args:
        img_emb: Image embeddings of shape (N, D)
        txt_emb: Text embeddings of shape (N, D)
        alpha: Weight for image embeddings (0 to 1)
               alpha=1.0 means only image, alpha=0.0 means only text
        
    Returns:
        Joint embeddings of shape (N, D), L2-normalized
    """
    I = l2_normalize(img_emb)
    T = l2_normalize(txt_emb)
    J = alpha * I + (1.0 - alpha) * T
    J = l2_normalize(J)
    return J.astype(np.float32, copy=False)


def shorten_text(text: str, max_length: int = 120) -> str:
    """
    Shorten text to maximum length with ellipsis.
    
    Args:
        text: Input text string
        max_length: Maximum length before truncation
        
    Returns:
        Shortened text with '…' if truncated
    """
    if len(text) > max_length:
        return text[:max_length] + "…"
    return text