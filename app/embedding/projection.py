"""
Dimensionality reduction and projection utilities.
Supports PCA, t-SNE, and UMAP projections.
"""

import numpy as np
import streamlit as st
from typing import Tuple, Optional
from scipy.linalg import orthogonal_procrustes
from importlib import import_module

# Try to import projection functions from utils.clip_utils
_clip = import_module("utils.clip_utils")

pca_project = getattr(_clip, "pca_project", None)
tsne_project = getattr(_clip, "tsne_project", None)
umap_project = getattr(_clip, "umap_project", None)

# Provide fallback implementations if not in utils
if pca_project is None:
    from sklearn.decomposition import PCA as _PCA
    def pca_project(X, n_components=2):
        """Fallback PCA projection."""
        return _PCA(n_components=min(n_components, X.shape[1])).fit_transform(
            X.astype(np.float32, copy=False)
        ).astype(np.float32, copy=False)

if tsne_project is None:
    from sklearn.manifold import TSNE as _TSNE
    def tsne_project(X, n_components=2, perplexity=30, random_state=42):
        """Fallback t-SNE projection."""
        return _TSNE(
            n_components=n_components,
            perplexity=int(perplexity),
            random_state=int(random_state),
            init="pca",
            metric="cosine"
        ).fit_transform(X.astype(np.float32, copy=False)).astype(np.float32, copy=False)

if umap_project is None:
    try:
        import umap
    except ImportError:
        umap = None
    
    def umap_project(X, n_neighbors=30, min_dist=0.1, random_state=42):
        """Fallback UMAP projection."""
        if umap is None:
            raise RuntimeError("umap-learn not installed. Install with: pip install umap-learn")
        
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric="cosine",
            random_state=int(random_state)
        )
        reducer_3d = umap.UMAP(
            n_components=3,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric="cosine",
            random_state=int(random_state)
        )
        return reducer_2d.fit_transform(X), reducer_3d.fit_transform(X)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize rows of a matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (X / norms).astype(np.float32, copy=False)


@st.cache_data(show_spinner=False)
def project_2d(
    X: np.ndarray,
    method: str,
    random_state: int,
    tsne_perplexity: int,
    umap_neighbors: int,
    umap_min_dist: float
) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D.
    
    Args:
        X: Input embeddings of shape (N, D)
        method: Projection method ('PCA', 'tSNE', or 'UMAP')
        random_state: Random seed
        tsne_perplexity: t-SNE perplexity parameter
        umap_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter
        
    Returns:
        2D coordinates of shape (N, 2)
    """
    method_lower = method.lower()
    
    if method_lower == "pca":
        return pca_project(X, n_components=2)
    
    elif method_lower == "tsne":
        return tsne_project(
            X,
            n_components=2,
            perplexity=int(tsne_perplexity),
            random_state=int(random_state)
        )
    
    elif method_lower == "umap":
        coords_2d, _ = umap_project(
            X,
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
            random_state=int(random_state)
        )
        return coords_2d
    
    else:
        raise ValueError(f"Unknown projection method: {method}")


def project_images_and_text(
    img_embeddings: np.ndarray,
    txt_embeddings: np.ndarray,
    method: str,
    random_state: int,
    tsne_perplexity: int,
    umap_neighbors: int,
    umap_min_dist: float,
    per_caption: bool = False,
    caption_img_idx: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project image and text embeddings to 2D, aligning them via Procrustes.
    
    Args:
        img_embeddings: Image embeddings (N_img, D)
        txt_embeddings: Text embeddings (N_txt, D)
        method: Projection method
        random_state: Random seed
        tsne_perplexity: t-SNE perplexity
        umap_neighbors: UMAP neighbors
        umap_min_dist: UMAP min distance
        per_caption: If True, use per-caption alignment
        caption_img_idx: Maps each caption to its parent image index
        
    Returns:
        Tuple of (image_coords_2d, text_coords_2d)
    """
    # Case 1: Per-image alignment (image and text have same count)
    if not per_caption and img_embeddings.shape[0] == txt_embeddings.shape[0]:
        return _project_with_procrustes_alignment(
            img_embeddings, txt_embeddings, method, random_state,
            tsne_perplexity, umap_neighbors, umap_min_dist
        )
    
    # Case 2: Per-caption alignment
    if per_caption:
        assert caption_img_idx is not None, "caption_img_idx required for per-caption mode"
        assert len(caption_img_idx) == txt_embeddings.shape[0], \
            "caption_img_idx must map each caption to an image index"
        
        return _project_with_per_caption_alignment(
            img_embeddings, txt_embeddings, caption_img_idx,
            method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist
        )
    
    # Case 3: No alignment, just stack and project
    I_norm = l2_normalize(img_embeddings)
    T_norm = l2_normalize(txt_embeddings)
    X_combined = np.vstack([I_norm, T_norm])
    Y_combined = project_2d(
        X_combined, method, random_state,
        tsne_perplexity, umap_neighbors, umap_min_dist
    )
    n_images = img_embeddings.shape[0]
    return Y_combined[:n_images], Y_combined[n_images:]


def _project_with_procrustes_alignment(
    I: np.ndarray,
    T: np.ndarray,
    method: str,
    random_state: int,
    tsne_perplexity: int,
    umap_neighbors: int,
    umap_min_dist: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align text to image embeddings using Procrustes analysis before projection.
    """
    # Normalize
    I_norm = l2_normalize(I)
    T_norm = l2_normalize(T)
    
    # Center
    I_centered = I_norm - I_norm.mean(axis=0, keepdims=True)
    T_centered = T_norm - T_norm.mean(axis=0, keepdims=True)
    
    # Find optimal rotation and scaling
    R, scale = orthogonal_procrustes(T_centered, I_centered)
    
    # Align text to image space
    T_aligned = (T_centered @ R) * scale + I_norm.mean(axis=0, keepdims=True)
    
    # Stack and project
    X_combined = np.vstack([I_norm, T_aligned])
    Y_combined = project_2d(
        X_combined, method, random_state,
        tsne_perplexity, umap_neighbors, umap_min_dist
    )
    
    n = I.shape[0]
    return Y_combined[:n], Y_combined[n:]


def _project_with_per_caption_alignment(
    I: np.ndarray,
    T: np.ndarray,
    caption_img_idx: list,
    method: str,
    random_state: int,
    tsne_perplexity: int,
    umap_neighbors: int,
    umap_min_dist: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align text to their parent images using Procrustes, for per-caption mode.
    """
    I_norm = l2_normalize(I)
    T_norm = l2_normalize(T)
    
    # Repeat parent image embeddings for each caption
    caption_idx_array = np.asarray(caption_img_idx, dtype=int)
    I_repeated = I_norm[caption_idx_array, :]
    
    # Center
    I_centered = I_repeated - I_repeated.mean(axis=0, keepdims=True)
    T_centered = T_norm - T_norm.mean(axis=0, keepdims=True)
    
    # Procrustes alignment
    R, scale = orthogonal_procrustes(T_centered, I_centered)
    T_aligned = (T_centered @ R) * scale + I_repeated.mean(axis=0, keepdims=True)
    
    # Stack original images with aligned text
    X_combined = np.vstack([I_norm, T_aligned])
    Y_combined = project_2d(
        X_combined, method, random_state,
        tsne_perplexity, umap_neighbors, umap_min_dist
    )
    
    n = I.shape[0]
    return Y_combined[:n], Y_combined[n:]