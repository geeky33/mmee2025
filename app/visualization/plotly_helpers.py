"""
High-level Plotly chart builders for the MMEE application.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
from .trace_builders import (
    build_outlier_traces_by_class,
    build_authenticity_traces,
    build_connection_lines,
    build_base_scatter_trace,
)


def create_scatter_with_outliers(
    df: pd.DataFrame,
    colors: Dict[str, str],
    title: str,
    outlier_size: int = 10,
    outlier_width: int = 2,
    authenticity_overlay: bool = False
) -> go.Figure:
    """
    Create scatter plot with optional outlier or authenticity overlay.
    
    Args:
        df: DataFrame with x, y, class_name, anomaly columns
        colors: Dictionary mapping class names to colors
        title: Plot title
        outlier_size: Size of outlier markers
        outlier_width: Width of outlier borders
        authenticity_overlay: If True, show TP/FP/FN instead of outliers
        
    Returns:
        Plotly Figure
    """
    # Separate clean and outlier points
    has_anomaly = "anomaly" in df.columns
    
    if has_anomaly:
        df_clean = df[~df["anomaly"].astype(bool)]
    else:
        df_clean = df
    
    # Create base scatter plot with clean points
    hover_cols = [
        c for c in ["class_name", "class_id", "image_relpath", "caption_short", "cosine_sim"]
        if c in df.columns
    ]
    
    fig = px.scatter(
        df_clean,
        x="x",
        y="y",
        color="class_name",
        hover_data=hover_cols,
        title=title,
        color_discrete_map=colors,
        opacity=0.85,
        render_mode="webgl"
    )
    
    # Update base marker style
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    
    # Add overlays
    if has_anomaly:
        if authenticity_overlay and "is_bad" in df.columns:
            # Add authenticity traces (TP, FP, FN)
            for trace in build_authenticity_traces(df, outlier_size, outlier_width):
                fig.add_trace(trace)
        else:
            # Add outlier traces by class
            for trace in build_outlier_traces_by_class(
                df, colors, outlier_size, outlier_width, name_prefix="Outlier"
            ):
                fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    
    return fig


def create_joint_plot(
    df_img: pd.DataFrame,
    df_txt: pd.DataFrame,
    colors: Dict[str, str],
    title: str,
    per_caption: bool,
    show_clean_only: bool = False,
    outlier_size: int = 10,
    outlier_width: int = 2,
    authenticity_overlay: bool = False
) -> go.Figure:
    """
    Create joint plot showing both images and text with connection lines.
    
    Args:
        df_img: Image DataFrame
        df_txt: Text DataFrame
        colors: Class color mapping
        title: Plot title
        per_caption: Per-caption mode flag
        show_clean_only: Show only clean points
        outlier_size: Outlier marker size
        outlier_width: Outlier marker width
        authenticity_overlay: Show authenticity overlay
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Add connection lines first (so they appear below points)
    for trace in build_connection_lines(df_img, df_txt, per_caption, show_clean_only):
        fig.add_trace(trace)
    
    # Add base markers for images and text
    fig.add_trace(build_base_scatter_trace(df_img, "Image"))
    
    text_name = "Text (per-caption)" if per_caption else "Text"
    fig.add_trace(build_base_scatter_trace(df_txt, text_name))
    
    # Add overlays
    if authenticity_overlay and "is_bad" in df_txt.columns:
        for trace in build_authenticity_traces(df_txt, outlier_size, outlier_width):
            fig.add_trace(trace)
    else:
        for trace in build_outlier_traces_by_class(
            df_img, colors, outlier_size, outlier_width, "Outlier"
        ):
            fig.add_trace(trace)
        for trace in build_outlier_traces_by_class(
            df_txt, colors, outlier_size, outlier_width, "Outlier"
        ):
            fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig


def create_confusion_matrix_heatmap(
    tn: int,
    fp: int,
    fn: int,
    tp: int
) -> go.Figure:
    """
    Create confusion matrix heatmap.
    
    Args:
        tn, fp, fn, tp: Confusion matrix values
        
    Returns:
        Plotly Figure
    """
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred 0 (clean)", "Pred 1 (outlier)"],
        y=["True 0 (clean)", "True 1 (bad cap)"],
        text=cm.astype(str),
        texttemplate="%{text}",
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Count: %{z}<extra></extra>",
        colorscale="Blues"
    ))
    
    fig.update_layout(
        title="Confusion Matrix (Per-caption detection vs Ground Truth)",
        xaxis_title="Prediction",
        yaxis_title="Ground Truth",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360
    )
    
    return fig


def create_pr_curve(
    per_class_data: list,
    title: str = "PR curves (validation)"
) -> go.Figure:
    """
    Create precision-recall curves for multiple classes.
    
    Args:
        per_class_data: List of tuples (class_name, AP, points_array)
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    for class_name, ap, points in per_class_data:
        fig.add_trace(go.Scatter(
            x=points[:, 0],  # Recall
            y=points[:, 1],  # Precision
            mode="lines",
            name=f"{class_name} (AP={ap:.2f})"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig


def create_roc_curve(
    per_class_data: list,
    title: str = "ROC curves (validation)"
) -> go.Figure:
    """
    Create ROC curves for multiple classes.
    
    Args:
        per_class_data: List of tuples (class_name, AUC, points_array)
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    for class_name, auc_score, points in per_class_data:
        fig.add_trace(go.Scatter(
            x=points[:, 0],  # FPR
            y=points[:, 1],  # TPR
            mode="lines",
            name=f"{class_name} (AUC={auc_score:.2f})"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="FPR",
        yaxis_title="TPR",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig


# Import numpy for confusion matrix
import numpy as np