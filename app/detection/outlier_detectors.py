"""
Outlier detection algorithms.
All methods work on a per-class basis for better precision.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN


def ensure_2d(X: np.ndarray) -> np.ndarray:
    """Ensure input is 2D array."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    return X


def isolation_forest_single_class(
    X: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 300,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Isolation Forest on a single class.
    
    Args:
        X: Feature matrix of shape (N, D)
        contamination: Expected proportion of outliers
        n_estimators: Number of trees in the forest
        random_state: Random seed
        
    Returns:
        Tuple of (labels, scores) where:
            - labels: Binary array (1 = outlier, 0 = inlier)
            - scores: Anomaly scores (higher = more anomalous)
    """
    X = ensure_2d(X)
    
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=float(contamination),
        random_state=int(random_state),
        n_jobs=-1,
        max_samples="auto"
    )
    model.fit(X)
    
    # Score samples: negate so higher = more anomalous
    scores = -model.score_samples(X).astype(np.float32)
    
    # Predict: -1 = outlier, 1 = inlier
    predictions = model.predict(X)
    labels = (predictions == -1).astype(np.int8)
    
    return labels, scores


def lof_single_class(
    X: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Local Outlier Factor on a single class.
    
    Args:
        X: Feature matrix of shape (N, D)
        n_neighbors: Number of neighbors to consider
        contamination: Expected proportion of outliers
        
    Returns:
        Tuple of (labels, scores)
    """
    X = ensure_2d(X)
    n_neighbors_eff = min(n_neighbors, max(2, len(X) - 1))
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors_eff,
        contamination=float(contamination),
        novelty=False,
        n_jobs=-1,
        metric="minkowski"
    )
    
    # Predict: -1 = outlier, 1 = inlier
    predictions = lof.fit_predict(X)
    labels = (predictions == -1).astype(np.int8)
    
    # Negative outlier factor: negate so higher = more anomalous
    scores = (-lof.negative_outlier_factor_).astype(np.float32)
    
    return labels, scores


def knn_quantile_single_class(
    X: np.ndarray,
    k: int = 10,
    quantile: float = 0.98
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using k-NN distance and quantile threshold.
    
    Args:
        X: Feature matrix of shape (N, D)
        k: Number of nearest neighbors
        quantile: Quantile threshold (e.g., 0.98 = flag top 2%)
        
    Returns:
        Tuple of (labels, scores)
    """
    X = ensure_2d(X)
    k_eff = min(k, max(1, len(X) - 1))
    
    # Fit k-NN and get distances
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1, metric="minkowski")
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Distance to k-th nearest neighbor (exclude self at index 0)
    kth_distances = distances[:, -1].astype(np.float32)
    
    # Threshold at quantile
    threshold = np.quantile(kth_distances, float(quantile)) if len(kth_distances) else np.inf
    labels = (kth_distances >= threshold).astype(np.int8)
    
    return labels, kth_distances


def dbscan_single_class(
    X: np.ndarray,
    eps: float = None,
    min_samples: int = 10,
    k_for_eps: int = 10,
    quantile_for_eps: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using DBSCAN (noise points).
    
    Args:
        X: Feature matrix of shape (N, D)
        eps: DBSCAN epsilon parameter (None = auto-compute)
        min_samples: Minimum samples for core point
        k_for_eps: k for auto-computing eps
        quantile_for_eps: Quantile for auto-computing eps
        
    Returns:
        Tuple of (labels, scores)
    """
    X = ensure_2d(X)
    n = len(X)
    
    if n <= 1:
        return np.zeros(n, np.int8), np.zeros(n, np.float32)
    
    # Auto-compute eps if not provided
    if eps is None:
        k_eff = min(k_for_eps, max(1, n - 1))
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        kth_distances = distances[:, -1]
        eps = float(np.quantile(kth_distances, float(quantile_for_eps)))
    
    # Run DBSCAN
    min_samples_eff = min(int(min_samples), max(2, n))
    clustering = DBSCAN(eps=float(eps), min_samples=min_samples_eff, n_jobs=-1)
    clustering.fit(X)
    
    # Label -1 = noise (outlier), others = clusters
    labels = (clustering.labels_ == -1).astype(np.int8)
    
    # Compute scores as k-NN distances for consistency
    try:
        k_eff = min(10, max(1, n - 1))
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        scores = distances[:, -1].astype(np.float32)
    except Exception:
        scores = labels.astype(np.float32)
    
    return labels, scores


def run_classwise_detector(
    X: np.ndarray,
    class_labels: pd.Series,
    method_name: str,
    contamination: float,
    params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run outlier detector per class and combine results.
    
    Args:
        X: Feature matrix of shape (N, D)
        class_labels: Series of class names/labels
        method_name: Detection method name
        contamination: Expected proportion of outliers
        params: Dictionary of method-specific parameters
        
    Returns:
        Tuple of:
            - labels_matrix: (N, C) binary matrix of per-class outlier labels
            - scores_matrix: (N, C) matrix of per-class anomaly scores
    """
    classes = class_labels.unique().tolist()
    n_samples = X.shape[0]
    
    labels_per_class = []
    scores_per_class = []
    
    for cls in classes:
        # Get samples for this class
        class_mask = (class_labels == cls).values
        
        if class_mask.sum() < 5:
            # Too few samples: mark all as inliers
            labels_local = np.zeros(class_mask.sum(), np.int8)
            scores_local = np.zeros(class_mask.sum(), np.float32)
        else:
            X_class = X[class_mask]
            
            # Run appropriate detector
            if method_name == "Isolation Forest":
                labels_local, scores_local = isolation_forest_single_class(
                    X_class, contamination=contamination
                )
            elif method_name == "kNN Distance (Quantile)":
                labels_local, scores_local = knn_quantile_single_class(
                    X_class,
                    k=int(params.get("knn_k", 10)),
                    quantile=float(params.get("knn_q", 0.98))
                )
            elif method_name == "LOF (Local Outlier Factor)":
                labels_local, scores_local = lof_single_class(
                    X_class,
                    n_neighbors=int(params.get("lof_k", 20)),
                    contamination=contamination
                )
            elif method_name == "DBSCAN (noise)":
                eps_val = None if params.get("db_eps_auto", False) else float(params.get("db_eps", 0.8))
                labels_local, scores_local = dbscan_single_class(
                    X_class,
                    eps=eps_val,
                    min_samples=int(params.get("db_min", 10))
                )
            else:
                raise ValueError(f"Unknown detection method: {method_name}")
            
            # Normalize scores per class for comparability
            if len(scores_local) > 1:
                mean_score = float(np.mean(scores_local))
                std_score = float(np.std(scores_local)) or 1.0
                scores_local = (scores_local - mean_score) / std_score
        
        # Scatter back to full array
        labels_full = np.zeros(n_samples, dtype=np.int8)
        scores_full = np.zeros(n_samples, dtype=np.float32)
        labels_full[class_mask] = labels_local
        scores_full[class_mask] = scores_local
        
        labels_per_class.append(labels_full)
        scores_per_class.append(scores_full)
    
    # Stack into matrices: (N, C)
    labels_matrix = np.stack(labels_per_class, axis=1)
    scores_matrix = np.stack(scores_per_class, axis=1)
    
    return labels_matrix, scores_matrix


def combine_classwise_predictions(
    labels_matrix: np.ndarray,
    scores_matrix: np.ndarray,
    mode: str = "union"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine per-class predictions into final predictions.
    
    Args:
        labels_matrix: (N, C) binary matrix
        scores_matrix: (N, C) score matrix
        mode: 'union' (recall mode) or 'intersection' (precision mode)
        
    Returns:
        Tuple of (final_labels, final_scores) of shape (N,)
    """
    if mode.lower().startswith("union"):
        # Union: flag if ANY class detector flags it
        final_labels = labels_matrix.max(axis=1).astype(np.int8)
        final_scores = scores_matrix.max(axis=1).astype(np.float32)
    else:
        # Intersection: flag only if ALL class detectors flag it
        final_labels = labels_matrix.min(axis=1).astype(np.int8)
        final_scores = scores_matrix.mean(axis=1).astype(np.float32)
    
    return final_labels, final_scores