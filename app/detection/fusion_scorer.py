"""
Feature fusion scorer using Logistic Regression with cross-validation.
Combines multiple features (cosine similarity, residual, odd-one-out, etc.)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_feature_table(
    df_txt: pd.DataFrame,
    img_embeddings: np.ndarray,
    txt_embeddings: np.ndarray,
    per_caption: bool,
    caption_img_idx: Optional[List[int]]
) -> np.ndarray:
    """
    Build feature table for fusion scorer.
    
    Features:
        1. Cosine similarity (image-text)
        2. L2 residual norm
        3. Odd-one-out score (deviation from mean)
        4. Caption length
        5. Detector anomaly score
    
    Args:
        df_txt: Text DataFrame with columns like cosine_sim, caption_short, anomaly_score
        img_embeddings: Image embeddings (N_img, D)
        txt_embeddings: Text embeddings (N_txt, D)
        per_caption: Whether in per-caption mode
        caption_img_idx: Maps each caption to its parent image index
        
    Returns:
        Feature matrix of shape (N_txt, 5)
    """
    # Feature 1: Cosine similarity (already in DataFrame)
    cosine_sim = df_txt["cosine_sim"].astype(float).fillna(0.0).to_numpy()
    
    # Feature 2 & 3: Residual and odd-one-out
    if per_caption and caption_img_idx is not None and len(caption_img_idx) == txt_embeddings.shape[0]:
        # Repeat parent image embeddings for each caption
        img_repeated = img_embeddings[np.asarray(caption_img_idx, dtype=int), :]
        residual = np.linalg.norm(img_repeated - txt_embeddings, axis=1).astype(np.float32)
        
        # Odd-one-out: how much each caption deviates from its siblings
        by_image = defaultdict(list)
        for j, img_idx in enumerate(caption_img_idx):
            by_image[img_idx].append(cosine_sim[j])
        
        odd_one_out = np.zeros_like(cosine_sim, dtype=np.float32)
        position = 0
        for img_idx, sibling_sims in by_image.items():
            sibling_array = np.asarray(sibling_sims, dtype=np.float32)
            mean_sim = float(np.mean(sibling_array))
            std_sim = float(np.std(sibling_array)) + 1e-6
            
            # Z-score for each sibling
            n_siblings = len(sibling_sims)
            z_scores = np.abs((sibling_array - mean_sim) / std_sim)
            odd_one_out[position:position + n_siblings] = z_scores
            position += n_siblings
    else:
        # Per-image mode: simple residual and odd-one-out
        residual = np.linalg.norm(img_embeddings - txt_embeddings, axis=1).astype(np.float32)
        
        mean_cos = np.mean(cosine_sim)
        std_cos = np.std(cosine_sim) + 1e-6
        odd_one_out = np.abs((cosine_sim - mean_cos) / std_cos).astype(np.float32)
    
    # Feature 4: Caption length
    caption_length = df_txt["caption_short"].fillna("").map(len).astype(np.float32).to_numpy()
    
    # Feature 5: Detector anomaly score
    detector_score = (
        df_txt["anomaly_score"].astype(float).to_numpy()
        if "anomaly_score" in df_txt.columns
        else np.zeros_like(cosine_sim)
    )
    
    # Stack all features
    feature_matrix = np.stack([
        cosine_sim,
        residual,
        odd_one_out,
        caption_length,
        detector_score
    ], axis=1).astype(np.float32)
    
    return feature_matrix


def train_fusion_scorer(
    X_all: np.ndarray,
    y_all: np.ndarray,
    val_mask: np.ndarray,
    target_recall: float = 0.70,
    cv_folds: int = 5
) -> Dict:
    """
    Train fusion scorer with cross-validation on validation set.
    
    Args:
        X_all: Feature matrix for all samples (N, 5)
        y_all: Ground truth labels (N,)
        val_mask: Boolean mask for validation samples
        target_recall: Target recall for threshold calibration
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with:
            - 'threshold': Calibrated threshold
            - 'ap_val': Average precision on validation set
            - 'pipeline': Fitted sklearn pipeline
            - 'scores_all': Predicted probabilities for all samples
            - 'predictions_all': Binary predictions for all samples
    """
    # Create pipeline: standardize then logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])
    
    # Cross-validation on validation set to get unbiased probability estimates
    if val_mask.any():
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        probs_val = cross_val_predict(
            pipeline,
            X_all[val_mask],
            y_all[val_mask],
            cv=skf,
            method="predict_proba"
        )[:, 1]  # Probability of positive class
        
        # Calibrate threshold to achieve target recall
        precision, recall, thresholds = precision_recall_curve(
            y_all[val_mask],
            probs_val
        )
        
        # Find threshold that achieves at least target_recall
        valid_indices = np.where(recall >= float(target_recall))[0]
        if len(valid_indices) == 0:
            threshold_star = 0.0  # No threshold achieves target recall
        else:
            # Among valid thresholds, pick the one with highest precision
            best_idx = valid_indices[np.argmax(precision[valid_indices])]
            threshold_star = float(thresholds[min(best_idx, len(thresholds) - 1)])
        
        # Compute average precision on validation set
        ap_val = float(average_precision_score(y_all[val_mask], probs_val))
    else:
        # No validation set: use default threshold
        threshold_star = 0.5
        ap_val = 0.0
    
    # Train on ALL data (both train and validation)
    pipeline.fit(X_all, y_all)
    
    # Score all samples
    scores_all = pipeline.predict_proba(X_all)[:, 1].astype(np.float32)
    predictions_all = (scores_all >= threshold_star).astype(np.int8)
    
    return {
        "threshold": threshold_star,
        "ap_val": ap_val,
        "pipeline": pipeline,
        "scores_all": scores_all,
        "predictions_all": predictions_all,
    }


def apply_fusion_scorer(
    df_txt: pd.DataFrame,
    img_embeddings: np.ndarray,
    txt_embeddings: np.ndarray,
    per_caption: bool,
    caption_img_idx: Optional[List[int]],
    val_mask: np.ndarray,
    target_recall: float = 0.70,
    cv_folds: int = 5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete fusion scorer workflow: build features, train, and apply.
    
    Args:
        df_txt: Text DataFrame
        img_embeddings: Image embeddings
        txt_embeddings: Text embeddings
        per_caption: Per-caption mode flag
        caption_img_idx: Caption-to-image index mapping
        val_mask: Validation split mask
        target_recall: Target recall for calibration
        cv_folds: CV folds
        
    Returns:
        Tuple of:
            - Updated df_txt with fusion_score and fusion_pred columns
            - Info dictionary with threshold and metrics
    """
    # Check if ground truth is available
    if "is_bad" not in df_txt.columns:
        return df_txt, {}
    
    # Build feature table
    X_all = build_feature_table(
        df_txt, img_embeddings, txt_embeddings,
        per_caption, caption_img_idx
    )
    
    # Get ground truth labels
    y_all = df_txt["is_bad"].astype(int).to_numpy()
    
    # Check if we have any positive examples
    if y_all.sum() == 0:
        # No positive examples: skip fusion
        return df_txt, {}
    
    # Train fusion scorer
    result = train_fusion_scorer(
        X_all, y_all, val_mask,
        target_recall=target_recall,
        cv_folds=cv_folds
    )
    
    # Add predictions to DataFrame
    df_txt = df_txt.copy()
    df_txt["fusion_score"] = result["scores_all"]
    df_txt["fusion_pred"] = result["predictions_all"]
    
    # Return info
    info = {
        "thr": result["threshold"],
        "ap": result["ap_val"]
    }
    
    return df_txt, info