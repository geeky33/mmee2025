"""
MMEE Embedding Viewer - Main Application
Clean, modular entry point that orchestrates all components.

Usage:
    streamlit run app/main.py
"""

import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

# Suppress optional warnings
warnings.filterwarnings("ignore", message="QuickGELU mismatch.*")

# Ensure project root is on path
PROJ_DIR = Path(__file__).resolve().parents[1]
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

# Import all modules
from app.config import settings
from app.data_loaders import (
    discover_datasets,
    scan_images,
    get_noise_files,
    read_captions,
    align_captions_to_images,
    load_bad_caption_ground_truth,
    read_labels,
)
from app.embedding import (
    compute_image_embeddings_cached,
    compute_text_embeddings_cached,
    prepare_per_caption_payload,
    get_aggregation_tag,
    project_images_and_text,
    cosine_similarity,
)
from app.detection import (
    run_classwise_detector,
    combine_classwise_predictions,
    apply_fusion_scorer,
    compute_detection_metrics,
    get_confusion_buckets,
    create_bucket_dataframe,
    compute_pr_roc_curves,
    create_export_dataframes,
    find_top_n_missed_per_class,
)
from app.visualization import (
    create_scatter_with_outliers,
    create_joint_plot,
    create_confusion_matrix_heatmap,
    create_pr_curve,
    create_roc_curve,
)
from app.ui import (
    render_dataset_selector,
    render_paths_display,
    render_captions_source_selector,
    render_model_controls,
    render_projection_controls,
    render_caption_controls,
    render_joint_outlier_controls,
    render_outlier_method_controls,
    render_combine_mode_control,
    render_validation_controls,
    render_fusion_controls,
    render_display_controls,
    render_evaluation_controls,
    render_subset_sampling_controls,
    display_kpis,
    display_detection_results,
    display_ranked_table,
    display_download_buttons,
    display_embedding_buttons,
    display_status_messages,
)
from app.utils import (
    signature_for_embeddings,
    rowwise_cosine_similarity,
    joint_weighted_embeddings,
    shorten_text,
    frozen_validation_split,
    validate_embedding_shapes,
    sample_dataframe_per_class,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    layout=settings.PAGE_LAYOUT
)

st.title(" • MULTI-MODAL EMBEDDING EXPLORER")

# ============================================================================
# DATA ROOT RESOLUTION (robust to inner/outer repo layout)
# ============================================================================
# Try the configured path, then common fallbacks relative to this file
DATA_ROOT = Path(settings.DATA_ROOT) if getattr(settings, "DATA_ROOT", None) else PROJ_DIR / "data"
candidates = [
    DATA_ROOT,
    PROJ_DIR / "data",          # .../embedding/data
    PROJ_DIR.parent / "data",   # .../data (if running inside .../embedding/embedding)
]
for p in candidates:
    if p.exists():
        settings.DATA_ROOT = p
        break

st.sidebar.caption(f"📁 Using DATA_ROOT = {settings.DATA_ROOT}")

# ============================================================================
# DATASET DISCOVERY & SELECTION
# ============================================================================
DATASETS = discover_datasets(settings.DATA_ROOT)

if not DATASETS:
    st.error(
        f"No datasets found under {settings.DATA_ROOT}. "
        f"Expected subfolders with images/, captions.json, labels.json."
    )
    st.stop()

# Render sidebar controls
with st.sidebar:
    dataset_name = render_dataset_selector(DATASETS)

# Reset embeddings if dataset changed
prev_ds = st.session_state.get("dataset_name")
if prev_ds != dataset_name:
    st.session_state["dataset_name"] = dataset_name
    for key in ("IMG_EMB", "TXT_EMB", "EMB_SIG", "EMB_PATHS"):
        st.session_state.pop(key, None)

# Get dataset paths
IMAGES_DIR = DATASETS[dataset_name]["images"]
CAPTIONS_JSON = DATASETS[dataset_name]["captions"]
LABELS_JSON = DATASETS[dataset_name]["labels"]

# ============================================================================
# CAPTIONS SOURCE SELECTION
# ============================================================================
CAPTIONS_NOISE_JSON, BAD_GT_JSON = get_noise_files(CAPTIONS_JSON)

with st.sidebar:
    captions_choice = render_captions_source_selector(
        CAPTIONS_NOISE_JSON.exists()
    )

# Determine active captions file
ACTIVE_CAPTIONS_JSON = (
    CAPTIONS_NOISE_JSON
    if (captions_choice == "With noise" and CAPTIONS_NOISE_JSON.exists())
    else CAPTIONS_JSON
)

# Load bad caption ground truth if available
BAD_GT = {}
if captions_choice == "With noise" and BAD_GT_JSON.exists():
    try:
        BAD_GT = load_bad_caption_ground_truth(BAD_GT_JSON)
        st.sidebar.caption("Ground-truth for injected bad captions: found")
    except Exception as e:
        st.sidebar.warning(f"Failed to load bad_caption_gt.json: {e}")
else:
    st.sidebar.caption("Ground-truth for bad captions: not active (Original captions).")

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    render_paths_display(IMAGES_DIR, ACTIVE_CAPTIONS_JSON, LABELS_JSON)
    
    model_name, pretrained, batch_img = render_model_controls()
    projection_params = render_projection_controls()
    caption_params = render_caption_controls()
    joint_params = render_joint_outlier_controls(projection_params["method"])
    outlier_params = render_outlier_method_controls()
    combine_mode = render_combine_mode_control()
    validation_params = render_validation_controls()
    fusion_params = render_fusion_controls()
    display_params = render_display_controls()
    eval_params = render_evaluation_controls()

# ============================================================================
# LOAD DATASET
# ============================================================================
with st.status("Loading dataset (images, labels, captions)…", expanded=False):
    IMG_PATHS = scan_images(IMAGES_DIR)
    if not IMG_PATHS:
        st.error("No images found under data/images.")
        st.stop()
    
    CAP_MAP = read_captions(ACTIVE_CAPTIONS_JSON)
    DF = read_labels(LABELS_JSON, IMG_PATHS, IMAGES_DIR)

# ============================================================================
# SUBSET & SAMPLING CONTROLS
# ============================================================================
all_classes = sorted(DF["class_name"].unique().tolist())
max_per_class = int(DF["class_name"].value_counts().max()) if not DF.empty else 1

with st.sidebar:
    subset_params = render_subset_sampling_controls(
        all_classes, dataset_name, max_per_class
    )

# Apply class filtering
DF = DF[DF["class_name"].isin(subset_params["chosen_classes"])].reset_index(drop=True)

# Apply per-class sampling
if subset_params["samples_per_class"] > 0:
    DF = sample_dataframe_per_class(DF, subset_params["samples_per_class"])

# Display status
display_status_messages(
    len(DF),
    DF["class_name"].nunique(),
    dataset_name,
    0,  # Will update below
    len(DF)
)

# ============================================================================
# ALIGN CAPTIONS TO SELECTED IMAGES
# ============================================================================
ALL_CAPS = align_captions_to_images(
    [Path(p) for p in DF["image_path"].tolist()],
    CAP_MAP,
    IMAGES_DIR
)

caps_available = sum(1 for c in ALL_CAPS if c)
st.caption(f"Captions available for {caps_available}/{len(ALL_CAPS)} selected images.")

# ============================================================================
# PREPARE CAPTION PAYLOAD (PER-CAPTION OR AGGREGATE)
# ============================================================================
per_caption = caption_params["per_caption"]
caps_limit = caption_params["caps_limit"]
text_agg = caption_params["text_agg"]

if per_caption:
    cap_payload, cap_texts, cap_img_idx = prepare_per_caption_payload(
        ALL_CAPS, caps_limit
    )
    n_txt_expected = len(cap_payload)
    agg_tag = get_aggregation_tag(per_caption, caps_limit, text_agg)
    
    sel_note = "all captions per image" if caps_limit == 0 else f"first {caps_limit} captions/image"
    st.info(f"GT positives computed over current subset and {sel_note}.")
else:
    cap_payload = ALL_CAPS
    cap_texts = []
    cap_img_idx = []
    n_txt_expected = len(ALL_CAPS)
    agg_tag = get_aggregation_tag(per_caption, caps_limit, text_agg)

# ============================================================================
# INVALIDATE EMBEDDINGS IF SUBSET CHANGED
# ============================================================================
cur_paths = tuple(DF["image_path"].tolist())
prev_paths = st.session_state.get("EMB_PATHS")

if prev_paths != cur_paths:
    for key in ("IMG_EMB", "TXT_EMB", "EMB_SIG"):
        st.session_state.pop(key, None)
    st.session_state["EMB_PATHS"] = cur_paths

# Generate current signature
CUR_SIG = signature_for_embeddings(DF, dataset_name, model_name, pretrained, agg_tag)

# ============================================================================
# COMPUTE EMBEDDINGS
# ============================================================================
compute_images, compute_text = display_embedding_buttons()

if compute_images:
    with st.spinner("Encoding images with CLIP…"):
        IMG = compute_image_embeddings_cached(
            DF["image_path"].tolist(),
            model_name,
            pretrained,
            batch_img,
            CUR_SIG
        )
        st.session_state.IMG_EMB = IMG
        st.session_state.EMB_SIG = CUR_SIG
        st.success(f"Image embeddings: {IMG.shape}")

if compute_text:
    with st.spinner("Encoding captions with CLIP…"):
        agg = text_agg if not per_caption else "first"
        TXT = compute_text_embeddings_cached(
            cap_payload,
            model_name,
            pretrained,
            agg,
            CUR_SIG
        )
        st.session_state.TXT_EMB = TXT
        st.session_state.EMB_SIG = CUR_SIG
        st.success(f"Text embeddings: {TXT.shape}")

# ============================================================================
# VALIDATE EMBEDDINGS
# ============================================================================
if "IMG_EMB" not in st.session_state or "TXT_EMB" not in st.session_state:
    st.warning("Compute both IMAGE and TEXT embeddings to proceed.")
    st.stop()

if st.session_state.get("EMB_SIG") != CUR_SIG:
    for key in ("IMG_EMB", "TXT_EMB", "EMB_SIG"):
        st.session_state.pop(key, None)
    st.warning("Settings changed. Please recompute IMAGE and TEXT embeddings.")
    st.stop()

IMG = st.session_state.IMG_EMB
TXT = st.session_state.TXT_EMB
n_images = len(DF)

if not validate_embedding_shapes(IMG, TXT, n_images, n_txt_expected):
    for key in ("IMG_EMB", "TXT_EMB", "EMB_SIG"):
        st.session_state.pop(key, None)
    st.warning("Subset/caption mode changed. Please recompute IMAGE and TEXT embeddings.")
    st.stop()

# ============================================================================
# COMPUTE PROJECTIONS
# ============================================================================
P_IMG, P_TXT = project_images_and_text(
    IMG,
    TXT,
    projection_params["method"],
    projection_params["random_state"],
    projection_params["tsne_perplexity"],
    projection_params["umap_neighbors"],
    projection_params["umap_min_dist"],
    per_caption=per_caption,
    caption_img_idx=cap_img_idx if per_caption else None
)

# ============================================================================
# COMPUTE JOINT EMBEDDINGS & COSINE SIMILARITIES
# ============================================================================
alpha_joint = joint_params["alpha_joint"]

if per_caption:
    IMG_rep = (
        IMG[np.asarray(cap_img_idx, dtype=int), :]
        if len(cap_img_idx) > 0
        else np.zeros((0, IMG.shape[1]), dtype=np.float32)
    )
    J = joint_weighted_embeddings(IMG_rep, TXT, alpha=alpha_joint)
    cos_pairs_caps = (
        rowwise_cosine_similarity(IMG_rep, TXT)
        if len(IMG_rep)
        else np.zeros((0,), dtype=np.float32)
    )
    
    # Aggregate cosine per image
    sim_by_img = defaultdict(list)
    for j, i in enumerate(cap_img_idx):
        sim_by_img[i].append(float(cos_pairs_caps[j]))
    cos_per_image = np.full(n_images, np.nan, dtype=np.float32)
    for i, vals in sim_by_img.items():
        cos_per_image[i] = float(np.mean(vals))
else:
    J = joint_weighted_embeddings(IMG, TXT, alpha=alpha_joint)
    cos_pairs_caps = rowwise_cosine_similarity(IMG, TXT)
    cos_per_image = cos_pairs_caps.copy()

# ============================================================================
# BUILD DATAFRAMES FOR PLOTTING
# ============================================================================
colors = settings.generate_class_colors(DF["class_name"].unique())

# Image DataFrame
df_img2 = pd.DataFrame({
    "x": P_IMG[:, 0],
    "y": P_IMG[:, 1],
    "class_name": DF["class_name"].values,
    "class_id": DF["class_id"].values,
    "image_relpath": DF["image_relpath"].values,
    "caption_short": [""] * n_images,
    "cosine_sim": cos_per_image,
    "anomaly": False,
    "anomaly_score": 0.0,
})

# Text DataFrame
if per_caption:
    parent_class = DF["class_name"].values
    parent_id = DF["class_id"].values
    parent_relpath = DF["image_relpath"].values
    cap_short = [shorten_text(c) for c in cap_texts]
    
    df_txt2 = pd.DataFrame({
        "x": P_TXT[:, 0],
        "y": P_TXT[:, 1],
        "class_name": [parent_class[i] for i in cap_img_idx],
        "class_id": [int(parent_id[i]) for i in cap_img_idx],
        "image_relpath": [parent_relpath[i] for i in cap_img_idx],
        "caption_short": cap_short,
        "cosine_sim": cos_pairs_caps,
        "anomaly": False,
        "anomaly_score": 0.0,
    })
else:
    first_caps = [caps[0] if caps else "" for caps in cap_payload]
    cap_short = [shorten_text(c) for c in first_caps]
    
    df_txt2 = pd.DataFrame({
        "x": P_TXT[:, 0],
        "y": P_TXT[:, 1],
        "class_name": DF["class_name"].values,
        "class_id": DF["class_id"].values,
        "image_relpath": DF["image_relpath"].values,
        "caption_short": cap_short,
        "cosine_sim": cos_pairs_caps,
        "anomaly": False,
        "anomaly_score": 0.0,
    })

# ============================================================================
# ATTACH GROUND TRUTH (is_bad) FOR PER-CAPTION MODE
# ============================================================================
if per_caption:
    counter = defaultdict(int)
    aligned_is_bad = []
    rels = df_txt2["image_relpath"].tolist()
    
    for rel in rels:
        k = counter[rel]
        seq = BAD_GT.get(rel) or BAD_GT.get(Path(rel).name)
        flag = int(seq[k]) if (seq and k < len(seq)) else 0
        aligned_is_bad.append(flag)
        counter[rel] += 1
    
    df_txt2["is_bad"] = aligned_is_bad
else:
    df_txt2["is_bad"] = 0

# ============================================================================
# FROZEN VALIDATION SPLIT
# ============================================================================
VAL_MASK = (
    frozen_validation_split(df_txt2, validation_params["valid_frac"])
    if per_caption
    else np.zeros(len(df_txt2), dtype=bool)
)
TRAIN_MASK = ~VAL_MASK

# ============================================================================
# OUTLIER DETECTION
# ============================================================================
OUT_LABELS = None
OUT_SCORES = None
target_df = None
used_method = None

if display_params["run_detection"]:
    # Choose detection space
    method_space = joint_params["method_space"]
    
    if method_space.startswith("Raw"):
        X_det = J
        target_df = (
            df_txt2
            if per_caption or (TXT.shape[0] == df_txt2.shape[0])
            else df_img2
        )
    elif "Image" in method_space:
        X_det = P_IMG
        target_df = df_img2
    else:
        X_det = P_TXT
        target_df = df_txt2
    
    # Run class-wise detection
    Lmat, Smat = run_classwise_detector(
        X_det,
        target_df["class_name"],
        outlier_params["method"],
        outlier_params["params"]["contamination"],
        outlier_params["params"]
    )
    
    # Combine per-class predictions
    OUT_LABELS, OUT_SCORES = combine_classwise_predictions(Lmat, Smat, combine_mode)
    
    # Update target DataFrame
    target_df["anomaly"] = (OUT_LABELS == 1)
    target_df["anomaly_score"] = OUT_SCORES
    used_method = outlier_params["method"]

# ============================================================================
# DISPLAY KPIs
# ============================================================================
pos_in_sel = (
    int(df_txt2["is_bad"].sum())
    if per_caption and ("is_bad" in df_txt2.columns)
    else None
)
flagged = (
    int(target_df["anomaly"].sum())
    if (display_params["run_detection"] and target_df is not None)
    else 0
)

display_kpis(len(df_img2), pos_in_sel, flagged)

if per_caption and ("is_bad" in df_txt2.columns):
    caption_mode_note = "all captions" if caps_limit == 0 else f"first {caps_limit} captions"
    st.caption(f"GT positives computed over current subset and {caption_mode_note}/image.")

# ============================================================================
# FUSION SCORER (OPTIONAL)
# ============================================================================
fusion_info = None

if (
    display_params["run_detection"]
    and fusion_params["use_fusion"]
    and per_caption
    and ("is_bad" in df_txt2.columns)
):
    df_txt2, fusion_info = apply_fusion_scorer(
        df_txt2,
        IMG,
        TXT,
        per_caption,
        cap_img_idx if per_caption else None,
        VAL_MASK,
        target_recall=validation_params["target_recall"],
        cv_folds=fusion_params["cv_folds"]
    )

# ============================================================================
# EVALUATION METRICS
# ============================================================================
metrics_text = None
tp_idx = fp_idx = fn_idx = tn_idx = np.array([], dtype=int)

if (
    eval_params["show_eval"]
    and display_params["run_detection"]
    and per_caption
    and ("is_bad" in df_txt2.columns)
):
    y_true = df_txt2["is_bad"].astype(int).values
    
    if fusion_params["use_fusion"] and ("fusion_pred" in df_txt2.columns):
        y_pred = df_txt2["fusion_pred"].astype(int).values
        score_for_rank = df_txt2["fusion_score"].astype(float).values
        method_note = f"{used_method} + Fusion(LR)"
    else:
        y_pred = df_txt2["anomaly"].astype(int).values
        score_for_rank = df_txt2["anomaly_score"].astype(float).values
        method_note = used_method
    
    # Compute metrics
    metrics = compute_detection_metrics(y_true, y_pred)
    
    # Display results
    caption_mode_note = "all" if caps_limit == 0 else f"first {caps_limit}"
    display_detection_results(
        metrics,
        method_note,
        int(y_true.sum()),
        caption_mode_note
    )
    
    # Confusion matrix
    cm_fig = create_confusion_matrix_heatmap(
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    )
    st.plotly_chart(cm_fig, use_container_width=True, theme="streamlit")
    
    # Get confusion buckets
    buckets = get_confusion_buckets(y_true, y_pred)
    tp_idx = buckets["tp"]
    fp_idx = buckets["fp"]
    fn_idx = buckets["fn"]
    tn_idx = buckets["tn"]
    
    # PR/ROC curves per class (validation only)
    if VAL_MASK.any():
        st.subheader("Per-class PR/ROC (validation)")
        col1, col2 = st.columns(2)
        
        with col1:
            pr_data = []
            for cls, g in df_txt2[VAL_MASK].groupby("class_name"):
                yt = g["is_bad"].astype(int).to_numpy()
                ps = (
                    g["fusion_score"]
                    if (fusion_params["use_fusion"] and "fusion_score" in g.columns)
                    else g["anomaly_score"]
                ).astype(float).to_numpy()
                
                if len(np.unique(yt)) < 2:
                    continue
                
                curves = compute_pr_roc_curves(yt, ps)
                pr_data.append((
                    cls,
                    curves["ap"],
                    np.stack([curves["recall"], curves["precision"]], axis=1)
                ))
            
            if pr_data:
                st.plotly_chart(
                    create_pr_curve(pr_data),
                    use_container_width=True,
                    theme="streamlit"
                )
        
        with col2:
            roc_data = []
            for cls, g in df_txt2[VAL_MASK].groupby("class_name"):
                yt = g["is_bad"].astype(int).to_numpy()
                ps = (
                    g["fusion_score"]
                    if (fusion_params["use_fusion"] and "fusion_score" in g.columns)
                    else g["anomaly_score"]
                ).astype(float).to_numpy()
                
                if len(np.unique(yt)) < 2:
                    continue
                
                curves = compute_pr_roc_curves(yt, ps)
                roc_data.append((
                    cls,
                    curves["auc"],
                    np.stack([curves["fpr"], curves["tpr"]], axis=1)
                ))
            
            if roc_data:
                st.plotly_chart(
                    create_roc_curve(roc_data),
                    use_container_width=True,
                    theme="streamlit"
                )
    
    # Ranked tables
    score_col = (
        "fusion_score"
        if (fusion_params["use_fusion"] and "fusion_score" in df_txt2.columns)
        else "anomaly_score"
    )
    
    df_tp = create_bucket_dataframe(df_txt2, tp_idx, "TP", score_col)
    df_fp = create_bucket_dataframe(df_txt2, fp_idx, "FP", score_col)
    df_fn = create_bucket_dataframe(df_txt2, fn_idx, "FN", score_col)
    
    display_ranked_table(df_tp, "Ranked detections: True Positives", eval_params["topN_sanity"])
    display_ranked_table(df_fp, "Ranked detections: False Positives", eval_params["topN_sanity"])
    display_ranked_table(df_fn, "Ranked detections: False Negatives", eval_params["topN_sanity"])
    
    # Top missed per class
    df_missed = find_top_n_missed_per_class(
        df_txt2,
        y_true,
        y_pred,
        score_col,
        top_n=min(5, eval_params["topN_sanity"])
    )
    
    if not df_missed.empty:
        display_ranked_table(
            df_missed,
            "Per-class Top-N missed bad captions",
            eval_params["topN_sanity"]
        )
    else:
        st.caption("No missed bad captions under current selection.")
    
    # CSV exports
    df_tn = create_bucket_dataframe(df_txt2, tn_idx, "TN", score_col)
    df_all_csv = create_export_dataframes(df_txt2, buckets, colors)
    
    display_download_buttons(df_all_csv, df_tp, df_fp, df_fn, df_tn)

# ============================================================================
# SIDE-BY-SIDE PLOTS
# ============================================================================
left, right = st.columns(2, gap="large")

clean_suffix = " • CLEAN" if display_params["show_clean_only"] else ""

with left:
    st.subheader(
        f"[{dataset_name}] Image embeddings • {projection_params['method']}{clean_suffix}"
    )
    
    df_img_plot = (
        df_img2
        if not display_params["show_clean_only"]
        else df_img2[df_img2["anomaly"] == False]
    )
    fig_img = create_scatter_with_outliers(
        df_img_plot,
        colors,
        f"Image • {projection_params['method']}",
        display_params["outlier_size"],
        display_params["outlier_width"],
        authenticity_overlay=False
    )
    st.plotly_chart(fig_img, use_container_width=True, theme="streamlit")

with right:
    title_txt = "Text (per-caption)" if per_caption else "Text (per-image)"
    st.subheader(
        f"[{dataset_name}] {title_txt} embeddings • {projection_params['method']}{clean_suffix}"
    )
    
    df_txt_plot = (
        df_txt2
        if not display_params["show_clean_only"]
        else df_txt2[df_txt2["anomaly"] == False]
    )
    authenticity = (
        eval_params["overlay_auth"]
        and per_caption
        and ("is_bad" in df_txt2.columns)
    )
    
    fig_txt = create_scatter_with_outliers(
        df_txt_plot,
        colors,
        f"Text • {projection_params['method']}",
        display_params["outlier_size"],
        display_params["outlier_width"],
        authenticity_overlay=authenticity
    )
    st.plotly_chart(fig_txt, use_container_width=True, theme="streamlit")

# ============================================================================
# JOINT PLOT
# ============================================================================
st.subheader(
    f"[{dataset_name}] Joint {projection_params['method']} (shared axes){clean_suffix}"
)

fig_joint = create_joint_plot(
    df_img2,
    df_txt2,
    colors,
    f"Joint • {projection_params['method']}",
    per_caption,
    display_params["show_clean_only"],
    display_params["outlier_size"],
    display_params["outlier_width"],
    eval_params["overlay_auth"] and per_caption and ("is_bad" in df_txt2.columns)
)
st.plotly_chart(fig_joint, use_container_width=True, theme="streamlit")

# ============================================================================
# SINGLE-VIEW: JOINT-BLEND PROJECTION (OPTIONAL)
# ============================================================================
with st.expander("Single-view: Joint-blend projection (uses α above)"):
    from app.embedding.projection import project_2d
    
    PJ = project_2d(
        J,
        projection_params["method"],
        projection_params["random_state"],
        projection_params["tsne_perplexity"],
        projection_params["umap_neighbors"],
        projection_params["umap_min_dist"]
    )
    
    # Build DataFrame for joint projection
    if per_caption:
        classes_j = [DF.iloc[i]["class_name"] for i in cap_img_idx]
        class_id_j = [int(DF.iloc[i]["class_id"]) for i in cap_img_idx]
        rels_j = [DF.iloc[i]["image_relpath"] for i in cap_img_idx]
        cap_short_j = [shorten_text(c) for c in cap_texts]
        cos_sim_j = cos_pairs_caps
        
        if target_df is not None and "anomaly" in target_df.columns:
            anom_vals = target_df["anomaly"].astype(bool).values
            score_vals = target_df["anomaly_score"].astype(float).values
        else:
            anom_vals = np.zeros(len(PJ), bool)
            score_vals = np.zeros(len(PJ), float)
    else:
        classes_j = DF["class_name"].values
        class_id_j = DF["class_id"].values
        rels_j = DF["image_relpath"].values
        first_caps = [caps[0] if caps else "" for caps in cap_payload]
        cap_short_j = [shorten_text(c) for c in first_caps]
        cos_sim_j = cos_pairs_caps
        
        if target_df is not None and "anomaly" in target_df.columns:
            anom_vals = target_df["anomaly"].astype(bool).values
            score_vals = target_df["anomaly_score"].astype(float).values
        else:
            anom_vals = np.zeros(len(PJ), bool)
            score_vals = np.zeros(len(PJ), float)
    
    df_joint = pd.DataFrame({
        "x": PJ[:, 0],
        "y": PJ[:, 1],
        "class_name": classes_j,
        "class_id": class_id_j,
        "image_relpath": rels_j,
        "caption_short": cap_short_j,
        "cosine_sim": cos_sim_j,
        "anomaly": anom_vals,
        "anomaly_score": score_vals,
    })
    
    fig_joint_blend = create_scatter_with_outliers(
        df_joint,
        colors,
        f"Joint-blend • {projection_params['method']}",
        display_params["outlier_size"],
        display_params["outlier_width"],
        authenticity_overlay=False
    )
    st.plotly_chart(fig_joint_blend, use_container_width=True, theme="streamlit")

# ============================================================================
# COSINE SIMILARITY SANITY CHECK
# ============================================================================
st.divider()
st.markdown("### 🔎 Optional: Cosine similarity sanity check (CLIP space)")

idx = st.slider("Pick an image index", 0, len(IMG) - 1, 0)

if st.button("Top-5 closest texts for this image (cosine)"):
    sims_row = cosine_similarity(IMG[idx:idx + 1], TXT)[0]
    topk = np.argsort(-sims_row)[:5]
    
    if per_caption:
        rels = [DF.iloc[cap_img_idx[i]]["image_relpath"] for i in topk]
        caps_show = [cap_texts[i] for i in topk]
        classes = [DF.iloc[cap_img_idx[i]]["class_name"] for i in topk]
    else:
        rels = DF["image_relpath"].values[topk]
        caps_show = [cap_payload[i][0] if cap_payload[i] else "" for i in topk]
        classes = DF["class_name"].values[topk]
    
    df_sim = pd.DataFrame({
        "rank": np.arange(1, len(topk) + 1),
        "image_relpath": rels,
        "class_name": classes,
        "caption": caps_show,
        "cosine_sim": sims_row[topk],
    })
    st.dataframe(df_sim, use_container_width=True)

# ============================================================================
# FOOTER NOTE
# ============================================================================
st.caption(
    "Per-caption mode shows one point per caption for Text and Joint; "
    "evaluation compares detector outliers vs bad_caption_gt.json (authenticity)."
)
