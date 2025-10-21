"""
Evaluation metrics and reporting for outlier detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive detection metrics.
    
    Args:
        y_true: Ground truth labels (0 = clean, 1 = bad)
        y_pred: Predicted labels (0 = clean, 1 = outlier)
        
    Returns:
        Dictionary with precision, recall, F1, TP, FP, FN, TN
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def get_confusion_buckets(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Get indices for each confusion matrix bucket.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with keys: 'tp', 'fp', 'fn', 'tn' and index arrays
    """
    return {
        "tp": np.where((y_true == 1) & (y_pred == 1))[0],
        "fp": np.where((y_true == 0) & (y_pred == 1))[0],
        "fn": np.where((y_true == 1) & (y_pred == 0))[0],
        "tn": np.where((y_true == 0) & (y_pred == 0))[0],
    }


def create_bucket_dataframe(
    df: pd.DataFrame,
    indices: np.ndarray,
    bucket_name: str,
    score_column: str = "anomaly_score"
) -> pd.DataFrame:
    """
    Create a DataFrame for a specific confusion matrix bucket.
    
    Args:
        df: Source DataFrame
        indices: Row indices for this bucket
        bucket_name: Name of bucket ('TP', 'FP', 'FN', 'TN')
        score_column: Column name for the anomaly score
        
    Returns:
        DataFrame with bucket column and sorted by score
    """
    if len(indices) == 0:
        return pd.DataFrame(columns=[
            "bucket", "score", "class_name", "image_relpath",
            "caption_short", "cosine_sim"
        ])
    
    subset = df.iloc[indices].copy()
    subset["bucket"] = bucket_name
    subset["score"] = subset[score_column] if score_column in subset.columns else 0.0
    subset = subset.sort_values("score", ascending=False)
    
    return subset[[
        "bucket", "score", "class_name", "image_relpath",
        "caption_short", "cosine_sim"
    ]]


def compute_per_class_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    class_column: str = "class_name"
) -> pd.DataFrame:
    """
    Compute per-class precision-recall metrics.
    
    Args:
        df: DataFrame with class labels
        y_true: Ground truth labels
        scores: Predicted scores
        class_column: Name of class column
        
    Returns:
        DataFrame with per-class AP scores
    """
    results = []
    
    for cls in df[class_column].unique():
        class_mask = (df[class_column] == cls).values
        y_true_class = y_true[class_mask]
        scores_class = scores[class_mask]
        
        # Skip if no variation in labels
        if len(np.unique(y_true_class)) < 2:
            continue
        
        # Compute average precision
        ap = average_precision_score(y_true_class, scores_class)
        
        results.append({
            "class_name": cls,
            "average_precision": float(ap),
            "n_samples": int(class_mask.sum()),
            "n_positives": int(y_true_class.sum()),
        })
    
    return pd.DataFrame(results)


def compute_pr_roc_curves(
    y_true: np.ndarray,
    scores: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute precision-recall and ROC curves.
    
    Args:
        y_true: Ground truth labels
        scores: Predicted scores
        
    Returns:
        Dictionary with curve data:
            - 'precision', 'recall', 'pr_thresholds'
            - 'fpr', 'tpr', 'roc_thresholds'
            - 'ap', 'auc'
    """
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "ap": float(ap),
        "auc": float(roc_auc),
    }


def create_export_dataframes(
    df: pd.DataFrame,
    buckets: Dict[str, np.ndarray],
    colors: Dict[str, str]
) -> pd.DataFrame:
    """
    Create exportable DataFrames for all confusion buckets.
    
    Args:
        df: Source DataFrame
        buckets: Dictionary from get_confusion_buckets()
        colors: Dictionary mapping class names to colors
        
    Returns:
        Combined DataFrame with all buckets
    """
    export_dfs = []
    
    for bucket_name, indices in buckets.items():
        if len(indices) == 0:
            continue
        
        subset = df.iloc[indices].copy()
        subset["bucket"] = bucket_name.upper()
        subset["class_color"] = subset["class_name"].map(colors).fillna("")
        
        export_dfs.append(subset[[
            "bucket", "class_name", "class_id", "image_relpath",
            "caption_short", "cosine_sim", "anomaly_score",
            "x", "y", "class_color"
        ]])
    
    if export_dfs:
        return pd.concat(export_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def find_top_n_missed_per_class(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_column: str,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Find top-N missed bad captions per class (false negatives).
    
    Args:
        df: DataFrame with detections
        y_true: Ground truth labels
        y_pred: Predicted labels
        score_column: Column with anomaly scores
        top_n: Number of top misses per class
        
    Returns:
        DataFrame with top missed captions per class
    """
    # Filter to false negatives
    fn_mask = (y_true == 1) & (y_pred == 0)
    df_fn = df[fn_mask].copy()
    
    if df_fn.empty:
        return pd.DataFrame(columns=["class_name", "score", "image_relpath", "caption_short"])
    
    results = []
    
    for cls, group in df_fn.groupby("class_name"):
        top_missed = group.nlargest(top_n, score_column)
        
        for _, row in top_missed.iterrows():
            results.append({
                "class_name": cls,
                "score": float(row[score_column]),
                "image_relpath": row["image_relpath"],
                "caption_short": row["caption_short"],
            })
    
    return pd.DataFrame(results)