"""
Build custom Plotly traces for different visualization needs.
Handles outlier markers, authenticity overlays, and connection lines.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List


def build_outlier_traces_by_class(
    df: pd.DataFrame,
    colors: dict,
    marker_size: int = 10,
    marker_width: int = 2,
    name_prefix: str = "Outlier"
) -> List[go.Scatter]:
    """
    Build Plotly traces for outliers, grouped by class.
    
    Args:
        df: DataFrame with columns: x, y, anomaly, class_name, etc.
        colors: Dictionary mapping class names to colors
        marker_size: Size of outlier markers
        marker_width: Width of outlier marker borders
        name_prefix: Prefix for trace names
        
    Returns:
        List of Plotly Scatter traces
    """
    # Filter to outliers only
    outliers = df[df["anomaly"].astype(bool)].copy()
    
    if outliers.empty:
        return []
    
    traces = []
    
    for class_name, group in outliers.groupby("class_name", sort=False):
        class_color = colors.get(class_name, "gray")
        
        # Build custom data for hover
        custom_data = np.stack([
            group["class_name"].astype(str).values,
            group["image_relpath"].astype(str).values,
            group["caption_short"].fillna("").astype(str).values,
            group["cosine_sim"].astype(float).values,
            group["anomaly_score"].astype(float).values
        ], axis=1)
        
        hover_template = (
            "class=%{customdata[0]}<br>"
            "🖼 %{customdata[1]}<br>"
            "📝 %{customdata[2]}<br>"
            "cosine=%{customdata[3]:.3f}<br>"
            "score=%{customdata[4]:.3f}<extra></extra>"
        )
        
        traces.append(go.Scatter(
            x=group["x"],
            y=group["y"],
            mode="markers",
            name=f"{name_prefix} · {class_name}",
            marker=dict(
                symbol="x",
                size=marker_size,
                color=class_color,
                line=dict(width=marker_width, color=class_color)
            ),
            customdata=custom_data,
            hovertemplate=hover_template,
            showlegend=True,
        ))
    
    return traces


def build_authenticity_traces(
    df: pd.DataFrame,
    marker_size: int = 10,
    marker_width: int = 2
) -> List[go.Scatter]:
    """
    Build traces showing true positives, false positives, and false negatives.
    Used for authenticity evaluation overlay.
    
    Args:
        df: DataFrame with columns: x, y, is_bad, anomaly, etc.
        marker_size: Marker size
        marker_width: Marker border width
        
    Returns:
        List of Plotly Scatter traces
    """
    traces = []
    
    if "is_bad" not in df.columns or "anomaly" not in df.columns:
        return traces
    
    # True Positives (detected bad captions)
    tp_mask = (df["is_bad"].astype(int) == 1) & (df["anomaly"].astype(bool))
    tp_df = df[tp_mask]
    
    if not tp_df.empty:
        custom_data = np.stack([
            tp_df["image_relpath"].astype(str).values,
            tp_df["caption_short"].fillna("").astype(str).values,
            tp_df["cosine_sim"].astype(float).values,
            tp_df["anomaly_score"].astype(float).values
        ], axis=1)
        
        hover_template = (
            "🖼 %{customdata[0]}<br>"
            "📝 %{customdata[1]}<br>"
            "cosine=%{customdata[2]:.3f}<br>"
            "score=%{customdata[3]:.3f}<extra>True outlier</extra>"
        )
        
        traces.append(go.Scatter(
            x=tp_df["x"],
            y=tp_df["y"],
            mode="markers",
            name="True outliers",
            marker=dict(
                symbol="x",
                size=marker_size,
                color="red",
                line=dict(width=marker_width, color="red")
            ),
            customdata=custom_data,
            hovertemplate=hover_template,
            showlegend=True,
        ))
    
    # False Positives (clean flagged as bad)
    fp_mask = (df["is_bad"].astype(int) == 0) & (df["anomaly"].astype(bool))
    fp_df = df[fp_mask]
    
    if not fp_df.empty:
        custom_data = np.stack([
            fp_df["image_relpath"].astype(str).values,
            fp_df["caption_short"].fillna("").astype(str).values,
            fp_df["cosine_sim"].astype(float).values,
            fp_df["anomaly_score"].astype(float).values
        ], axis=1)
        
        hover_template = (
            "🖼 %{customdata[0]}<br>"
            "📝 %{customdata[1]}<br>"
            "cosine=%{customdata[2]:.3f}<br>"
            "score=%{customdata[3]:.3f}<extra>False positive</extra>"
        )
        
        traces.append(go.Scatter(
            x=fp_df["x"],
            y=fp_df["y"],
            mode="markers",
            name="False positives",
            marker=dict(
                symbol="x",
                size=marker_size,
                color="white",
                line=dict(width=marker_width, color="white")
            ),
            customdata=custom_data,
            hovertemplate=hover_template,
            showlegend=True,
        ))
    
    # False Negatives (bad captions missed)
    fn_mask = (df["is_bad"].astype(int) == 1) & (~df["anomaly"].astype(bool))
    fn_df = df[fn_mask]
    
    if not fn_df.empty:
        custom_data = np.stack([
            fn_df["image_relpath"].astype(str).values,
            fn_df["caption_short"].fillna("").astype(str).values,
            fn_df["cosine_sim"].astype(float).values
        ], axis=1)
        
        hover_template = (
            "🖼 %{customdata[0]}<br>"
            "📝 %{customdata[1]}<br>"
            "cosine=%{customdata[2]:.3f}<extra>False negative</extra>"
        )
        
        traces.append(go.Scatter(
            x=fn_df["x"],
            y=fn_df["y"],
            mode="markers",
            name="Missed bad captions",
            marker=dict(
                symbol="diamond-open",
                size=marker_size - 1,
                line=dict(width=marker_width, color="orange")
            ),
            customdata=custom_data,
            hovertemplate=hover_template,
            showlegend=True,
        ))
    
    return traces


def build_connection_lines(
    df_img: pd.DataFrame,
    df_txt: pd.DataFrame,
    per_caption: bool,
    show_clean_only: bool = False
) -> List[go.Scatter]:
    """
    Build connection lines between images and their captions/text.
    
    Args:
        df_img: Image DataFrame with x, y, image_relpath
        df_txt: Text DataFrame with x, y, image_relpath, anomaly
        per_caption: Whether in per-caption mode
        show_clean_only: Whether to show only clean connections
        
    Returns:
        List of line traces
    """
    traces = []
    
    # Create mapping from image_relpath to index
    rel_to_index = {
        rel: idx for idx, rel in enumerate(df_img["image_relpath"])
    }
    
    # Filter text DataFrame if needed
    if show_clean_only and "anomaly" in df_txt.columns:
        df_txt_filtered = df_txt[df_txt["anomaly"] == False]
    else:
        df_txt_filtered = df_txt
    
    if per_caption:
        # Per-caption: connect each caption to its parent image
        for j in range(len(df_txt_filtered)):
            rel_path = df_txt_filtered.iloc[j]["image_relpath"]
            img_idx = rel_to_index.get(rel_path)
            
            if img_idx is None:
                continue
            
            traces.append(go.Scatter(
                x=[df_img.iloc[img_idx]["x"], df_txt_filtered.iloc[j]["x"]],
                y=[df_img.iloc[img_idx]["y"], df_txt_filtered.iloc[j]["y"]],
                mode="lines",
                line=dict(width=0.5, color="gray"),
                showlegend=False,
                hoverinfo="skip"
            ))
    else:
        # Per-image: one-to-one connections
        for i in range(min(len(df_img), len(df_txt_filtered))):
            if pd.isna(df_txt_filtered.iloc[i]["x"]):
                continue
            
            traces.append(go.Scatter(
                x=[df_img.iloc[i]["x"], df_txt_filtered.iloc[i]["x"]],
                y=[df_img.iloc[i]["y"], df_txt_filtered.iloc[i]["y"]],
                mode="lines",
                line=dict(width=0.5, color="gray"),
                showlegend=False,
                hoverinfo="skip"
            ))
    
    return traces


def build_base_scatter_trace(
    df: pd.DataFrame,
    name: str,
    marker_size: int = 6
) -> go.Scatter:
    """
    Build a base scatter trace for images or text.
    
    Args:
        df: DataFrame with x, y, class_name, image_relpath, caption_short, cosine_sim
        name: Trace name
        marker_size: Marker size
        
    Returns:
        Plotly Scatter trace
    """
    custom_data = np.stack([
        df["class_name"].astype(str).values,
        df["image_relpath"].astype(str).values,
        df["caption_short"].fillna("").astype(str).values,
        df["cosine_sim"].astype(float).values
    ], axis=1)
    
    hover_template = (
        "class_name=%{customdata[0]}<br>"
        "x=%{x:.6f}<br>y=%{y:.6f}<br>"
        "image_relpath=%{customdata[1]}<br>"
        "caption_short=%{customdata[2]}<br>"
        "cosine_sim=%{customdata[3]:.3f}<extra></extra>"
    )
    
    return go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        name=name,
        marker=dict(size=marker_size),
        customdata=custom_data,
        hovertemplate=hover_template
    )