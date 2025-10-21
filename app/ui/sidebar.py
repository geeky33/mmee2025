"""
Sidebar UI components and controls.
All sidebar widgets are defined here for better organization.
"""

import streamlit as st
from pathlib import Path
from typing import Tuple, List
from ..config import settings


def render_dataset_selector(datasets: dict) -> str:
    """
    Render dataset selection dropdown.
    
    Args:
        datasets: Dictionary of available datasets
        
    Returns:
        Selected dataset name
    """
    st.header("Dataset")
    return st.selectbox(
        "Choose dataset",
        list(datasets.keys()),
        index=0,
        key="dataset_selector"
    )


def render_paths_display(images_dir: Path, captions_file: Path, labels_file: Path):
    """
    Display current dataset paths.
    
    Args:
        images_dir: Path to images directory
        captions_file: Path to captions JSON
        labels_file: Path to labels JSON
    """
    st.header("Paths")
    st.code(str(images_dir), language="bash")
    st.code(str(captions_file), language="bash")
    st.code(str(labels_file), language="bash")


def render_captions_source_selector(captions_noise_exists: bool) -> str:
    """
    Render captions source selector (Original vs With noise).
    
    Args:
        captions_noise_exists: Whether captions_with_noise.json exists
        
    Returns:
        Selected caption source
    """
    st.header("Captions source")
    options = ["Original"]
    if captions_noise_exists:
        options.append("With noise")
    
    return st.radio(
        "Select captions file",
        options,
        index=0,
        help="Use tools/add_bad_captions.py to generate captions_with_noise.json + bad_caption_gt.json"
    )


def render_model_controls() -> Tuple[str, str, int]:
    """
    Render model selection controls.
    
    Returns:
        Tuple of (model_name, pretrained_weights, batch_size)
    """
    st.header("Model")
    
    model_name = st.selectbox(
        "CLIP model",
        settings.AVAILABLE_MODELS,
        index=0
    )
    
    pretrained = st.selectbox(
        "Weights",
        settings.AVAILABLE_WEIGHTS,
        index=0
    )
    
    batch_size = st.slider(
        "Image batch size",
        8, 128, settings.DEFAULT_BATCH_SIZE, step=8
    )
    
    return model_name, pretrained, batch_size


def render_projection_controls() -> dict:
    """
    Render projection method controls.
    
    Returns:
        Dictionary with projection parameters
    """
    st.header("Projection")
    
    method = st.selectbox(
        "Method",
        settings.PROJECTION_METHODS,
        index=0
    )
    
    random_state = st.number_input(
        "Random state",
        value=settings.DEFAULT_RANDOM_STATE,
        step=1
    )
    
    tsne_perplexity = st.slider(
        "t-SNE perplexity",
        *settings.TSNE_PERPLEXITY_RANGE,
        settings.DEFAULT_TSNE_PERPLEXITY
    )
    
    umap_neighbors = st.slider(
        "UMAP n_neighbors",
        *settings.UMAP_NEIGHBORS_RANGE,
        settings.DEFAULT_UMAP_NEIGHBORS
    )
    
    umap_min_dist = st.slider(
        "UMAP min_dist",
        0.0, 1.0,
        settings.DEFAULT_UMAP_MIN_DIST
    )
    
    return {
        "method": method,
        "random_state": random_state,
        "tsne_perplexity": tsne_perplexity,
        "umap_neighbors": umap_neighbors,
        "umap_min_dist": umap_min_dist,
    }


def render_caption_controls() -> dict:
    """
    Render caption granularity controls.
    
    Returns:
        Dictionary with caption parameters
    """
    st.header("Captions")
    
    caption_mode = st.radio(
        "Caption granularity",
        ["Aggregate per image", "Per-caption points"],
        index=0
    )
    
    per_caption = (caption_mode == "Per-caption points")
    
    if per_caption:
        caps_limit = st.number_input(
            "Max captions/image (0 = all)",
            min_value=0,
            value=0,
            step=1,
            help="Embed the first K captions per image. Set 0 to use all captions."
        )
        text_agg = "first"  # Not used in per-caption mode
    else:
        caps_limit = 0
        text_agg = st.selectbox(
            "Text aggregation",
            ["average", "first"],
            index=0
        )
    
    return {
        "per_caption": per_caption,
        "caps_limit": caps_limit,
        "text_agg": text_agg,
    }


def render_joint_outlier_controls(method: str) -> dict:
    """
    Render joint embedding and outlier detection space controls.
    
    Args:
        method: Projection method name
        
    Returns:
        Dictionary with joint/outlier parameters
    """
    st.header("Joint & Outliers")
    
    alpha_joint = st.slider(
        "Joint blend α (image ↔ text)",
        0.0, 1.0, 0.5, step=0.05,
        help="α=1.0 means only image, α=0.0 means only text"
    )
    
    method_space = st.selectbox(
        "Detection space",
        ["Raw joint (512D)", f"{method} 2D: Image", f"{method} 2D: Text"],
        index=0
    )
    
    return {
        "alpha_joint": alpha_joint,
        "method_space": method_space,
    }


def render_outlier_method_controls() -> dict:
    """
    Render outlier detection method and parameter controls.
    
    Returns:
        Dictionary with outlier detection parameters
    """
    st.header("Outlier method (per-class)")
    
    out_method = st.selectbox(
        "Choose method",
        settings.OUTLIER_METHODS,
        index=0
    )
    
    with st.container(border=True):
        st.caption("Method parameters (applied per class)")
        
        contamination = st.slider(
            "Contamination / share of outliers",
            0.0, 0.3,
            settings.DEFAULT_CONTAMINATION,
            step=0.005
        )
        
        # Method-specific parameters
        params = {"contamination": contamination}
        
        if out_method == "kNN Distance (Quantile)":
            params["knn_k"] = st.slider("k (neighbors)", 2, 50, settings.DEFAULT_KNN_K)
            params["knn_q"] = st.slider("Quantile", 0.80, 0.999, settings.DEFAULT_KNN_QUANTILE)
        
        if out_method == "LOF (Local Outlier Factor)":
            params["lof_k"] = st.slider("n_neighbors", 5, 100, settings.DEFAULT_LOF_K)
        
        if out_method == "DBSCAN (noise)":
            params["db_eps_auto"] = st.checkbox("Auto eps from class kNN distances", value=True)
            params["db_eps"] = st.slider("eps (ignored if auto)", 0.05, 5.0, settings.DEFAULT_DBSCAN_EPS)
            params["db_min"] = st.slider("min_samples", 3, 100, settings.DEFAULT_DBSCAN_MIN_SAMPLES)
    
    return {
        "method": out_method,
        "params": params,
    }


def render_combine_mode_control() -> str:
    """
    Render per-class combine mode control.
    
    Returns:
        Selected combine mode
    """
    st.header("Per-class combine")
    return st.radio(
        "Merge multiple class detectors into a single prediction",
        ["Union (recall mode)", "Intersection (precision mode)"],
        index=0
    )


def render_validation_controls() -> dict:
    """
    Render validation and target controls.
    
    Returns:
        Dictionary with validation parameters
    """
    st.header("Validation / Targets")
    
    target_recall = st.slider(
        "Target recall (calibration)",
        0.10, 0.95,
        settings.DEFAULT_TARGET_RECALL,
        0.05
    )
    
    min_precision = st.slider(
        "Min precision (display goal)",
        0.05, 0.80,
        settings.DEFAULT_MIN_PRECISION,
        0.05
    )
    
    valid_frac = st.slider(
        "Validation split (frozen)",
        0.05, 0.50,
        settings.DEFAULT_VALIDATION_FRACTION,
        0.05
    )
    
    return {
        "target_recall": target_recall,
        "min_precision": min_precision,
        "valid_frac": valid_frac,
    }


def render_fusion_controls() -> dict:
    """
    Render fusion scorer controls.
    
    Returns:
        Dictionary with fusion parameters
    """
    st.header("Fusion scorer")
    
    use_fusion = st.checkbox(
        "Enable feature-fusion scorer (LR + CV)",
        value=True
    )
    
    cv_folds = st.slider(
        "CV folds",
        3, 10,
        settings.DEFAULT_CV_FOLDS
    )
    
    return {
        "use_fusion": use_fusion,
        "cv_folds": cv_folds,
    }


def render_display_controls() -> dict:
    """
    Render display and visualization controls.
    
    Returns:
        Dictionary with display parameters
    """
    st.header("Apply / Display")
    
    run_detection = st.checkbox("Run detection", value=False)
    show_clean_only = st.checkbox("Show only clean data in plots", value=False)
    
    outlier_size = st.slider(
        "Outlier marker size",
        *settings.OUTLIER_SIZE_RANGE,
        settings.DEFAULT_OUTLIER_SIZE,
        help="Size of the X markers"
    )
    
    outlier_width = st.slider(
        "Outlier stroke width",
        *settings.OUTLIER_WIDTH_RANGE,
        settings.DEFAULT_OUTLIER_WIDTH
    )
    
    return {
        "run_detection": run_detection,
        "show_clean_only": show_clean_only,
        "outlier_size": outlier_size,
        "outlier_width": outlier_width,
    }


def render_evaluation_controls() -> dict:
    """
    Render evaluation display controls.
    
    Returns:
        Dictionary with evaluation display parameters
    """
    st.header("Evaluation (vs GT)")
    
    show_eval = st.checkbox(
        "Show authenticity metrics & confusion matrix",
        value=True
    )
    
    overlay_auth = st.checkbox(
        "Overlay authenticity on plots",
        value=True
    )
    
    topN_sanity = st.slider(
        "Sanity tables: top-N by score",
        3, 50, 10
    )
    
    return {
        "show_eval": show_eval,
        "overlay_auth": overlay_auth,
        "topN_sanity": topN_sanity,
    }


def render_subset_sampling_controls(
    all_classes: List[str],
    dataset_name: str,
    max_per_class_in_data: int
) -> dict:
    """
    Render subset and sampling controls.
    
    Args:
        all_classes: List of all available class names
        dataset_name: Current dataset name (for unique keys)
        max_per_class_in_data: Maximum samples per class in dataset
        
    Returns:
        Dictionary with subset/sampling parameters
    """
    st.header("Subset & Sampling")
    
    n_classes = len(all_classes)
    
    if n_classes <= 1:
        st.info(f"Only one class detected for [{dataset_name}].")
        max_classes_to_show = 1
        default_classes = all_classes
    else:
        max_classes_to_show = st.slider(
            "Max classes to include",
            min_value=1,
            max_value=n_classes,
            value=min(settings.DEFAULT_MAX_CLASSES, n_classes),
            key=f"max_classes_{dataset_name}"
        )
        default_classes = all_classes[:max_classes_to_show]
    
    chosen_classes = st.multiselect(
        "Choose classes (optional)",
        all_classes,
        default=default_classes,
        key=f"chosen_classes_{dataset_name}"
    )
    
    # Determine upper limit for samples per class
    upper_samples = max(1, min(settings.MAX_SAMPLES_UPPER_LIMIT, max_per_class_in_data))
    default_samples = min(settings.DEFAULT_SAMPLES_PER_CLASS, upper_samples)
    
    samples_per_class = st.slider(
        "Samples per class (0 = all)",
        min_value=0,
        max_value=upper_samples,
        value=default_samples,
        step=1,
        key=f"samples_per_class_{dataset_name}"
    )
    
    return {
        "chosen_classes": chosen_classes if chosen_classes else default_classes,
        "samples_per_class": samples_per_class,
    }