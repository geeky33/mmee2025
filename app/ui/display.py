"""
Display components for the main application area.
Includes KPIs, status messages, and data tables.
"""

import streamlit as st
import pandas as pd
from typing import Optional


def display_kpis(
    n_images: int,
    n_positives: Optional[int],
    n_flagged: int
):
    """
    Display key performance indicators in columns.
    
    Args:
        n_images: Total number of images
        n_positives: Number of positive examples (or None)
        n_flagged: Number of flagged outliers
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total image samples", n_images)
    
    with col2:
        st.metric(
            "GT positives in selection",
            n_positives if n_positives is not None else "—"
        )
    
    with col3:
        st.metric("Outliers flagged", n_flagged)


def display_detection_results(
    metrics: dict,
    method_name: str,
    positives_count: int,
    caption_mode_note: str
):
    """
    Display detection results and metrics.
    
    Args:
        metrics: Dictionary with precision, recall, F1, TP, FP, FN, TN
        method_name: Name of detection method used
        positives_count: Number of positive examples in current selection
        caption_mode_note: Note about caption mode (e.g., "all captions" or "first 5 captions")
    """
    metrics_text = (
        f"Precision={metrics['precision']:.3f} • "
        f"Recall={metrics['recall']:.3f} • "
        f"F1={metrics['f1']:.3f} • "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}"
    )
    
    st.info(f"Operating point: **{method_name}**")
    st.info(
        f"Positives in current subset: **{positives_count}** "
        f"(after class/sample filters and {caption_mode_note})"
    )
    st.success(f"Detection vs GT (per-caption): {metrics_text}")


def display_ranked_table(
    df: pd.DataFrame,
    title: str,
    top_n: int = 10
):
    """
    Display a ranked table with a title.
    
    Args:
        df: DataFrame to display
        title: Section title
        top_n: Number of top rows to show
    """
    st.subheader(title)
    if df.empty:
        st.caption("No data to display.")
    else:
        st.dataframe(df.head(top_n), use_container_width=True)


def display_download_buttons(
    df_all: pd.DataFrame,
    df_tp: pd.DataFrame,
    df_fp: pd.DataFrame,
    df_fn: pd.DataFrame,
    df_tn: pd.DataFrame
):
    """
    Display CSV download buttons for detection results.
    
    Args:
        df_all: Combined DataFrame with all buckets
        df_tp, df_fp, df_fn, df_tn: Individual bucket DataFrames
    """
    with st.expander("📥 Download detections (CSV)"):
        st.download_button(
            "Download ALL buckets (TP/FP/FN/TN)",
            data=df_all.to_csv(index=False),
            file_name="detections_all_buckets.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.download_button(
            "TP.csv",
            df_tp.to_csv(index=False),
            "tp.csv",
            "text/csv"
        )
        col2.download_button(
            "FP.csv",
            df_fp.to_csv(index=False),
            "fp.csv",
            "text/csv"
        )
        col3.download_button(
            "FN.csv",
            df_fn.to_csv(index=False),
            "fn.csv",
            "text/csv"
        )
        col4.download_button(
            "TN.csv",
            df_tn.to_csv(index=False),
            "tn.csv",
            "text/csv"
        )


def display_embedding_buttons() -> tuple:
    """
    Display buttons for computing embeddings.
    
    Returns:
        Tuple of (compute_images, compute_text) boolean flags
    """
    col1, col2 = st.columns(2)
    
    with col1:
        compute_images = st.button(
            "🖼️ Compute IMAGE embeddings",
            use_container_width=True
        )
    
    with col2:
        compute_text = st.button(
            "📝 Compute TEXT embeddings",
            use_container_width=True
        )
    
    return compute_images, compute_text


def display_status_messages(
    df_length: int,
    n_classes: int,
    dataset_name: str,
    n_captions_available: int,
    n_total_images: int,
    caption_note: str = ""
):
    """
    Display status messages about loaded data.
    
    Args:
        df_length: Number of images in filtered DataFrame
        n_classes: Number of unique classes
        dataset_name: Name of current dataset
        n_captions_available: Number of images with captions
        n_total_images: Total number of images (before filtering)
        caption_note: Additional note about caption mode
    """
    st.success(
        f"[{dataset_name}] Using {df_length} images across {n_classes} classes."
    )
    
    st.caption(
        f"Captions available for {n_captions_available}/{n_total_images} selected images."
    )
    
    if caption_note:
        st.info(caption_note)