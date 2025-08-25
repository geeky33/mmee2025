# app/main.py
# MMEE CUB-200 Embedding Viewer (uses utils/clip_utils)
# Streamlit app: projections (PCA / t-SNE / UMAP) for IMAGE and TEXT embeddings
# + Outlier detection (multiple methods), removal, and clean-view UI
#
# Usage:
#   streamlit run app/main.py

from pathlib import Path
from typing import List, Dict, Any
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.linalg import orthogonal_procrustes
from importlib import import_module

# -- ensure project root (parent of app/) is on sys.path
PROJ_DIR = Path(__file__).resolve().parents[1]
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

# 🔗 Import helpers (with graceful fallbacks if some names are missing)
_clip = import_module("utils.clip_utils")

# always grab the core functions from utils
_compute_image_embeddings = _clip.compute_image_embeddings
_compute_text_embeddings  = _clip.compute_text_embeddings
cosine_similarity         = _clip.cosine_similarity
joint_weighted_embeddings = getattr(_clip, "joint_weighted_embeddings", None)

# try to import projection + iforest helpers from utils
pca_project    = getattr(_clip, "pca_project", None)
tsne_project   = getattr(_clip, "tsne_project", None)
umap_project   = getattr(_clip, "umap_project", None)
iforest_detect = getattr(_clip, "iforest_detect", None)
iforest_on_raw = getattr(_clip, "iforest_on_raw", None)

# provide local fallbacks if missing (so the app still runs)
if pca_project is None:
    from sklearn.decomposition import PCA as _PCA
    def pca_project(X, n_components=2):
        return _PCA(n_components=min(n_components, X.shape[1])).fit_transform(
            X.astype(np.float32, copy=False)
        ).astype(np.float32, copy=False)

if tsne_project is None:
    try:
        from sklearn.manifold import TSNE as _TSNE
    except Exception:
        _TSNE = None
    def tsne_project(X, n_components=2, perplexity=30, random_state=42):
        if _TSNE is None:
            raise RuntimeError("scikit-learn TSNE not available")
        return _TSNE(
            n_components=n_components, perplexity=int(perplexity),
            random_state=int(random_state), init="pca", metric="cosine"
        ).fit_transform(X.astype(np.float32, copy=False)).astype(np.float32, copy=False)

if umap_project is None:
    try:
        import umap
    except Exception:
        umap = None
    def umap_project(X, n_neighbors=30, min_dist=0.1, random_state=42):
        if umap is None:
            raise RuntimeError("umap-learn not installed")
        r2 = umap.UMAP(n_components=2, n_neighbors=int(n_neighbors), min_dist=float(min_dist),
                       metric="cosine", random_state=int(random_state))
        r3 = umap.UMAP(n_components=3, n_neighbors=int(n_neighbors), min_dist=float(min_dist),
                       metric="cosine", random_state=int(random_state))
        return r2.fit_transform(X), r3.fit_transform(X)

if iforest_detect is None or iforest_on_raw is None:
    try:
        from sklearn.ensemble import IsolationForest as _IF
    except Exception:
        _IF = None
    def iforest_detect(X, contamination=0.05, n_estimators=300,
                       max_samples="auto", random_state=42, return_model=False):
        if _IF is None:
            raise RuntimeError("scikit-learn IsolationForest not available")
        X = np.asarray(X, dtype=np.float32, order="C")
        model = _IF(n_estimators=n_estimators, contamination=contamination,
                    max_samples=max_samples, random_state=random_state, n_jobs=-1)
        model.fit(X)
        scores = (-model.score_samples(X)).astype(np.float32, copy=False)  # higher = more anomalous
        labels = (model.predict(X) == -1).astype(np.int8, copy=False)
        return (labels, scores, model) if return_model else (labels, scores)
    def iforest_on_raw(E, **kwargs):
        return iforest_detect(E, **kwargs)

# ensure joint blending exists even if utils was older
if joint_weighted_embeddings is None:
    def joint_weighted_embeddings(img, txt, alpha=0.5):
        I = img / np.maximum(np.linalg.norm(img, axis=1, keepdims=True), 1e-12)
        T = txt / np.maximum(np.linalg.norm(txt, axis=1, keepdims=True), 1e-12)
        J = alpha * I + (1.0 - alpha) * T
        J /= np.maximum(np.linalg.norm(J, axis=1, keepdims=True), 1e-12)
        return J.astype(np.float32, copy=False)

# -----------------------------
# Paths / Project layout
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJ_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
CAPTIONS_JSON = DATA_DIR / "captions.json"
LABELS_JSON = DATA_DIR / "labels.json"

# Optional on-disk caches dir
CACHE_DIR = PROJ_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Data loading (images, captions, labels)
# -----------------------------

def scan_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])

def read_captions(captions_path: Path) -> Dict[str, List[str]]:
    if not captions_path.exists():
        return {}
    try:
        obj = pd.read_json(captions_path, typ="series")
        return {
            str(k): list(v) if isinstance(v, (list, tuple))
            else ([str(v)] if pd.notna(v) else [])
            for k, v in obj.items()
        }
    except ValueError:
        import json
        with open(captions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, List[str]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[str(k)] = [str(x) for x in v]
            elif v is None:
                out[str(k)] = []
            else:
                out[str(k)] = [str(v)]
        return out

def _derive_class_from_folder(image_path: Path, images_root: Path) -> str:
    try:
        rel = image_path.relative_to(images_root)
        if rel.parts and len(rel.parts) > 1:
            return rel.parts[0]
    except Exception:
        pass
    return image_path.stem.split("_")[0]

def read_labels(labels_path: Path, image_paths: List[Path]) -> pd.DataFrame:
    """Read labels.json (dict or list) or infer from folder structure."""
    records: List[Dict[str, Any]] = []

    by_stem = {p.stem: p for p in image_paths}
    try:
        by_rel = {str(p.relative_to(IMAGES_DIR)): p for p in image_paths}
    except Exception:
        by_rel = {}

    used = set()
    if labels_path.exists():
        import json
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for row in data:
                key = row.get("image_relpath") or row.get("image_path") or row.get("path") or row.get("file")
                if not key:
                    continue
                p = by_rel.get(key) or by_stem.get(Path(key).stem)
                if not p:
                    continue
                class_id = row.get("class_id")
                class_name = row.get("class_name") or _derive_class_from_folder(p, IMAGES_DIR)
                records.append({
                    "image_path": str(p),
                    "image_relpath": str(p.relative_to(IMAGES_DIR)) if p.is_relative_to(IMAGES_DIR) else p.name,
                    "class_name": class_name,
                    "class_id": int(class_id) if class_id is not None else None,
                })
                used.add(p)

        elif isinstance(data, dict):
            for key, meta in data.items():
                p = by_rel.get(key) or by_stem.get(Path(key).stem)
                if not p:
                    continue
                if isinstance(meta, dict):
                    class_id = meta.get("class_id")
                    class_name = meta.get("class_name") or _derive_class_from_folder(p, IMAGES_DIR)
                else:
                    class_id = None
                    class_name = str(meta) if meta else _derive_class_from_folder(p, IMAGES_DIR)
                records.append({
                    "image_path": str(p),
                    "image_relpath": str(p.relative_to(IMAGES_DIR)) if p.is_relative_to(IMAGES_DIR) else p.name,
                    "class_name": class_name,
                    "class_id": int(class_id) if class_id is not None else None,
                })
                used.add(p)

    for p in image_paths:
        if p in used:
            continue
        class_name = _derive_class_from_folder(p, IMAGES_DIR)
        records.append({
            "image_path": str(p),
            "image_relpath": str(p.relative_to(IMAGES_DIR)) if p.is_relative_to(IMAGES_DIR) else p.name,
            "class_name": class_name,
            "class_id": None,
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No labeled images found under data/images.")

    if df["class_id"].isna().any():
        name2id = {name: i + 1 for i, name in enumerate(sorted(df["class_name"].unique()))}
        df["class_id"] = df["class_name"].map(name2id)
    df["class_id"] = df["class_id"].astype(int)
    return df

def captions_for_images(image_paths: List[Path], captions_map: Dict[str, List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    for p in image_paths:
        try:
            rel = str(p.relative_to(IMAGES_DIR))
        except Exception:
            rel = None
        for key in (rel, p.name, p.stem):
            if key and key in captions_map:
                caps = [c for c in captions_map[key] if c]
                out.append(caps)
                break
        else:
            out.append([])
    return out

# -----------------------------
# Projection helpers (via utils)
# -----------------------------

@st.cache_data(show_spinner=False)
def project_2d(X: np.ndarray, method: str, random_state: int,
               tsne_perplexity: int, umap_neighbors: int, umap_min_dist: float) -> np.ndarray:
    m = method.lower()
    if m == "pca":
        return pca_project(X, n_components=2)
    if m == "tsne":
        return tsne_project(
            X, n_components=2, perplexity=int(tsne_perplexity),
            random_state=int(random_state)
        )
    if m == "umap":
        coords_2d, _ = umap_project(
            X, n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
            random_state=int(random_state),
        )
        return coords_2d
    raise ValueError(f"Unknown method: {method}")

def _l2_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (X / n).astype(np.float32, copy=False)

def joint_project_2d(IMG: np.ndarray, TXT: np.ndarray, method: str, random_state: int,
                     tsne_perplexity: int, umap_neighbors: int, umap_min_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """Project images+texts together on a shared 2D basis and split."""
    # 1) L2-normalize
    I = _l2_normalize_np(IMG)
    T = _l2_normalize_np(TXT)
    # 2) Orthogonal Procrustes (align text manifold to image manifold)
    Ic = I - I.mean(0, keepdims=True)
    Tc = T - T.mean(0, keepdims=True)
    R, s = orthogonal_procrustes(Tc, Ic)   # solve min ||Tc R - Ic||_F
    T_aligned = (Tc @ R) * s + I.mean(0, keepdims=True)
    # 3) Stack and project on a common 2D basis
    X = np.vstack([I, T_aligned])
    Y = project_2d(X, method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist)
    n = IMG.shape[0]
    return Y[:n], Y[n:]

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="MMEE • CUB-200 Embedding Viewer", layout="wide")
st.title("🕊️ MMEE • CUB-200 Embedding Viewer (data/* + utils/clip_utils)")

with st.sidebar:
    st.header("Data")
    st.code(str(IMAGES_DIR), language="bash")
    st.code(str(CAPTIONS_JSON), language="bash")
    st.code(str(LABELS_JSON), language="bash")

    st.header("Model")
    model_name = st.selectbox("CLIP model", ["ViT-B-32", "ViT-L-14", "ViT-L-14-336"], index=0)
    pretrained = st.selectbox("Weights", ["openai", "laion2b_s34b_b79k", "laion2b_s32b_b82k"], index=0)
    batch_img = st.slider("Image batch size", 8, 128, 64, step=8)
    text_agg = st.selectbox("Text aggregation", ["average", "first"], index=0)

    st.header("Projection")
    method = st.selectbox("Method", ["PCA", "tSNE", "UMAP"], index=0)
    random_state = st.number_input("Random state", value=42, step=1)
    tsne_perplexity = st.slider("t-SNE perplexity", 5, 60, 30)
    umap_neighbors = st.slider("UMAP n_neighbors", 5, 100, 30)
    umap_min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)

    st.header("Joint & Outliers")
    alpha_joint = st.slider("Joint blend α (image ↔ text)", 0.0, 1.0, 0.5, step=0.05)

    method_space = st.selectbox(
        "Detection space",
        ["Raw joint (512D)", f"{method} 2D: Image", f"{method} 2D: Text"],
        index=0
    )

    out_method = st.selectbox(
        "Outlier method",
        [
            "Isolation Forest",
            "kNN Distance (Quantile)",
            "LOF (Local Outlier Factor)",
            "DBSCAN (noise)",
            "Mahalanobis (robust)",
            "PCA Reconstruction Error",
            "One-Class SVM",
        ],
        index=0
    )

    with st.container(border=True):
        st.caption("Method parameters")
        contamination = st.slider("Contamination / ν", 0.0, 0.2, 0.03, step=0.005)
        if out_method == "kNN Distance (Quantile)":
            knn_k = st.slider("k (neighbors)", 2, 50, 10)
            knn_q = st.slider("Quantile", 0.80, 0.999, 0.98)
        if out_method == "LOF (Local Outlier Factor)":
            lof_k = st.slider("n_neighbors", 5, 100, 20)
        if out_method == "DBSCAN (noise)":
            db_eps = st.slider("eps", 0.05, 5.0, 0.8)
            db_min = st.slider("min_samples", 3, 100, 10)
        if out_method == "Mahalanobis (robust)":
            robust = st.checkbox("Robust (MinCovDet)", value=True)
        if out_method == "PCA Reconstruction Error":
            pca_nc = st.text_input("n_components (int or fraction)", value="0.9")
        if out_method == "One-Class SVM":
            ocsvm_kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
            ocsvm_gamma = st.selectbox("gamma", ["scale", "auto"], index=0)

    st.header("Apply / Display")
    run_detection   = st.checkbox("Run detection", value=False)
    remove_outliers = st.checkbox("Remove outliers (show clean data)", value=False)
    show_clean_only = st.checkbox("Show only clean data in plots", value=False)

# Load dataset
with st.status("Loading dataset (images, labels, captions)…", expanded=False):
    IMG_PATHS = scan_images(IMAGES_DIR)
    if not IMG_PATHS:
        st.error("No images found under data/images.")
        st.stop()
    CAP_MAP = read_captions(CAPTIONS_JSON)
    DF = read_labels(LABELS_JSON, IMG_PATHS)

st.success(f"Indexed {len(DF)} images across {DF['class_name'].nunique()} classes.")

# Class filter & sampling
all_classes = sorted(DF["class_name"].unique().tolist())
with st.sidebar:
    chosen = st.multiselect("Choose classes (optional)", all_classes, default=all_classes[:min(len(all_classes), 30)])

max_classes_to_show = st.session_state.get("max_classes_to_show", len(all_classes))
if chosen:
    DF = DF[DF["class_name"].isin(chosen)].reset_index(drop=True)
else:
    DF = DF[DF["class_name"].isin(all_classes[:min(len(all_classes), 30)])].reset_index(drop=True)

samples_per_class = st.session_state.get("samples_per_class", 30)
if samples_per_class > 0:
    DF = DF.groupby("class_name", group_keys=False).apply(
        lambda g: g.sample(min(len(g), samples_per_class), random_state=42)
    ).reset_index(drop=True)

st.write(f"Using **{len(DF)}** images across **{DF['class_name'].nunique()}** classes.")

# Align captions per selected image order
def captions_for_images(image_paths: List[Path], captions_map: Dict[str, List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    for p in image_paths:
        try:
            rel = str(p.relative_to(IMAGES_DIR))
        except Exception:
            rel = None
        for key in (rel, p.name, p.stem):
            if key and key in captions_map:
                caps = [c for c in captions_map[key] if c]
                out.append(caps)
                break
        else:
            out.append([])
    return out

ALL_CAPS = captions_for_images([Path(p) for p in DF["image_path"].tolist()], CAP_MAP)
HAVE = sum(1 for c in ALL_CAPS if c)
st.caption(f"Captions available for {HAVE}/{len(ALL_CAPS)} images.")

# ⚡️ Cached wrappers around utils.clip_utils
@st.cache_data(show_spinner=False)
def _img_emb_cached(paths: List[str], model: str, weights: str, bs: int) -> np.ndarray:
    return _compute_image_embeddings(paths, model_name=model, pretrained=weights, batch_size=bs)

@st.cache_data(show_spinner=False)
def _txt_emb_cached(caps: List[List[str]], model: str, weights: str, agg: str) -> np.ndarray:
    return _compute_text_embeddings(caps, model_name=model, pretrained=weights, aggregate=agg)

# Buttons
colA, colB = st.columns(2)
with colA:
    if st.button("🖼️ Compute IMAGE embeddings", use_container_width=True):
        with st.spinner("Encoding images with CLIP…"):
            IMG = _img_emb_cached(DF["image_path"].tolist(), model_name, pretrained, batch_img)
            st.session_state.IMG_EMB = IMG
            st.success(f"Image embeddings: {IMG.shape}")
with colB:
    if st.button("📝 Compute TEXT embeddings", use_container_width=True):
        with st.spinner("Encoding captions with CLIP…"):
            TXT = _txt_emb_cached(ALL_CAPS, model_name, pretrained, text_agg)
            st.session_state.TXT_EMB = TXT
            st.success(f"Text embeddings: {TXT.shape}")

if "IMG_EMB" not in st.session_state or "TXT_EMB" not in st.session_state:
    st.warning("Compute both IMAGE and TEXT embeddings to proceed.")
    st.stop()

IMG = st.session_state.IMG_EMB
TXT = st.session_state.TXT_EMB

# Joint projection (shared axes for side-by-side view)
with st.spinner(f"Projecting jointly with {method}…"):
    P_IMG, P_TXT = joint_project_2d(
        IMG, TXT, method, int(random_state), int(tsne_perplexity),
        int(umap_neighbors), float(umap_min_dist)
    )

# Joint-blended embedding in raw CLIP space (512D)
J = joint_weighted_embeddings(IMG, TXT, alpha=alpha_joint)

# Prepare frames
class_names_sorted = sorted(DF["class_name"].unique())
colors = {
    name: f"hsl({int(360*i/max(1,len(class_names_sorted)))},70%,50%)"
    for i, name in enumerate(class_names_sorted)
}

df_img2 = pd.DataFrame({
    "x": P_IMG[:, 0],
    "y": P_IMG[:, 1],
    "class_name": DF["class_name"].values,
    "class_id": DF["class_id"].values,
    "image_relpath": DF["image_relpath"].values,
    "anomaly": False,
    "anomaly_score": 0.0,
})
df_txt2 = pd.DataFrame({
    "x": P_TXT[:, 0],
    "y": P_TXT[:, 1],
    "class_name": DF["class_name"].values,
    "class_id": DF["class_id"].values,
    "image_relpath": DF["image_relpath"].values,
    "anomaly": False,
    "anomaly_score": 0.0,
})

# -----------------------------
# Outlier detection orchestration
# -----------------------------

def run_outlier_detector(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Return labels(1=outlier), scores, and method name used."""
    m = out_method
    if m == "Isolation Forest":
        labels, scores = iforest_detect(X, contamination=float(contamination))
    elif m == "kNN Distance (Quantile)":
        labels, scores = _clip.knn_quantile_detect(X, k=int(knn_k), quantile=float(knn_q))
    elif m == "LOF (Local Outlier Factor)":
        labels, scores = _clip.lof_detect(X, n_neighbors=int(lof_k), contamination=float(contamination))
    elif m == "DBSCAN (noise)":
        labels, scores = _clip.dbscan_detect(X, eps=float(db_eps), min_samples=int(db_min))
    elif m == "Mahalanobis (robust)":
        labels, scores = _clip.mahalanobis_detect(X, contamination=float(contamination), robust=bool(robust))
    elif m == "PCA Reconstruction Error":
        try:
            val = float(pca_nc)
            nc = val if 0 < val < 1 else int(val)
        except Exception:
            nc = 0.9
        labels, scores = _clip.pca_recon_error_detect(X, n_components=nc, contamination=float(contamination))
    else:  # One-Class SVM
        labels, scores = _clip.ocsvm_detect(
            X, nu=float(contamination) if contamination > 0 else 0.01,
            kernel=str(ocsvm_kernel), gamma=str(ocsvm_gamma)
        )
    return labels.astype(np.int8, copy=False), scores.astype(np.float32, copy=False), m

# Choose feature space
if method_space.startswith("Raw"):
    X_det = J  # recommended
elif "Image" in method_space:
    X_det = P_IMG
else:
    X_det = P_TXT

OUT_LABELS = np.zeros(len(DF), dtype=np.int8)
OUT_SCORES = np.zeros(len(DF), dtype=np.float32)
used_method = None

if run_detection:
    OUT_LABELS, OUT_SCORES, used_method = run_outlier_detector(X_det)

# Attach labels/scores to dataframes
for d in (df_img2, df_txt2):
    d["anomaly"] = (OUT_LABELS == 1)
    d["anomaly_score"] = OUT_SCORES

# Clean mask & cleaned frames
if remove_outliers and run_detection:
    CLEAN_MASK = (OUT_LABELS == 0)
else:
    CLEAN_MASK = np.ones(len(DF), dtype=bool)

df_img2_clean = df_img2.loc[CLEAN_MASK].reset_index(drop=True)
df_txt2_clean = df_txt2.loc[CLEAN_MASK].reset_index(drop=True)

# KPIs + Download flags
leftKPI, rightKPI = st.columns(2)
with leftKPI:
    st.metric("Total samples", len(DF))
with rightKPI:
    st.metric("Outliers flagged", int((OUT_LABELS == 1).sum()) if run_detection else 0)

download_df = DF.copy()[["image_relpath", "class_name", "class_id"]]
download_df["is_outlier"] = (OUT_LABELS == 1)
download_df["outlier_score"] = OUT_SCORES
st.download_button(
    "Download outlier flags (CSV)",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="outlier_flags.csv",
    mime="text/csv",
)

# -----------------------------
# Plotting
# -----------------------------

def _scatter_with_outliers(df: pd.DataFrame, title: str):
    # Use only hover cols that actually exist
    hover_cols = [c for c in ["class_id", "image_relpath"] if c in df.columns]

    # Safe mask if 'anomaly' missing
    if "anomaly" in df.columns:
        mask = df["anomaly"].astype(bool)
    else:
        mask = pd.Series(False, index=df.index)

    base = px.scatter(
        df[~mask], x="x", y="y", color="class_name",
        hover_data=hover_cols,
        title=title,
        color_discrete_map=colors, opacity=0.85, render_mode="webgl"
    )
    base.update_traces(marker=dict(size=6, line=dict(width=0)))

    # Overlay outliers (if present)
    if mask.any():
        out = df[mask].copy()
        if "anomaly_score" not in out.columns:
            out["anomaly_score"] = 0.0
        base.add_trace(go.Scatter(
            x=out["x"], y=out["y"], mode="markers", name="Outliers",
            marker=dict(size=10, line=dict(width=2), symbol="x"),
            text=(out["class_name"] + " • score=" + out["anomaly_score"].round(3).astype(str)),
            hovertemplate="%{text}<extra>Outlier</extra>",
            showlegend=True,
        ))
    base.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return base

# Side-by-side plots (optionally clean-only)
left, right = st.columns(2, gap="large")
with left:
    st.subheader(f"Image embeddings • {method}" + (" • CLEAN" if show_clean_only else ""))
    to_plot = df_img2_clean if show_clean_only else df_img2
    st.plotly_chart(_scatter_with_outliers(to_plot, f"Image • {method}"), use_container_width=True, theme="streamlit")
with right:
    st.subheader(f"Text embeddings • {method}" + (" • CLEAN" if show_clean_only else ""))
    to_plot = df_txt2_clean if show_clean_only else df_txt2
    st.plotly_chart(_scatter_with_outliers(to_plot, f"Text • {method}"), use_container_width=True, theme="streamlit")

# Optional: single joint chart with connection lines (apply clean mask if asked)
st.subheader(f"Joint {method} (shared axes)" + (" • CLEAN" if show_clean_only else ""))
fig_joint = go.Figure()
idx_iter = np.where(CLEAN_MASK)[0] if show_clean_only else range(len(DF))
for i in idx_iter:
    fig_joint.add_trace(go.Scatter(
        x=[P_IMG[i,0], P_TXT[i,0]],
        y=[P_IMG[i,1], P_TXT[i,1]],
        mode="lines",
        line=dict(width=0.5),
        showlegend=False,
        hoverinfo="skip"
    ))
fig_joint.add_trace(go.Scatter(
    x=P_IMG[CLEAN_MASK,0] if show_clean_only else P_IMG[:,0],
    y=P_IMG[CLEAN_MASK,1] if show_clean_only else P_IMG[:,1],
    mode="markers", name="Image",
    marker=dict(size=6),
    text=DF.loc[CLEAN_MASK, "class_name"] if show_clean_only else DF["class_name"],
    hovertemplate="Image • %{text}<extra></extra>"
))
fig_joint.add_trace(go.Scatter(
    x=P_TXT[CLEAN_MASK,0] if show_clean_only else P_TXT[:,0],
    y=P_TXT[CLEAN_MASK,1] if show_clean_only else P_TXT[:,1],
    mode="markers", name="Text",
    marker=dict(size=6, symbol="diamond"),
    text=DF.loc[CLEAN_MASK, "class_name"] if show_clean_only else DF["class_name"],
    hovertemplate="Text • %{text}<extra></extra>"
))
fig_joint.update_layout(margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_joint, use_container_width=True, theme="streamlit")

# Optional: 2D projection of blended joint vectors (one point per item)
with st.expander("Single-view: Joint-blend projection (uses α above)"):
    PJ = project_2d(J, method, int(random_state), int(tsne_perplexity), int(umap_neighbors), float(umap_min_dist))
    df_joint = pd.DataFrame({
        "x": PJ[:,0], "y": PJ[:,1],
        "class_name": DF["class_name"].values,
        "class_id": DF["class_id"].values,            # include for hover
        "image_relpath": DF["image_relpath"].values,
        "anomaly": (OUT_LABELS == 1),
        "anomaly_score": OUT_SCORES,
    })
    to_plot = df_joint.loc[CLEAN_MASK].reset_index(drop=True) if show_clean_only else df_joint
    st.plotly_chart(_scatter_with_outliers(to_plot, f"Joint-blend • {method}"), use_container_width=True, theme="streamlit")

st.divider()
st.markdown("### 🔎 Optional: Cosine similarity sanity check (CLIP space)")
idx = st.slider("Pick an image index", 0, len(IMG)-1, 0)
if st.button("Top-5 closest texts for this image (cosine)"):
    sims_row = cosine_similarity(IMG[idx:idx+1], TXT)[0]
    topk = np.argsort(-sims_row)[:5]
    df_sim = pd.DataFrame({
        "rank": np.arange(1, len(topk)+1),
        "image_relpath": DF.iloc[topk]["image_relpath"].values,
        "class_name": DF.iloc[topk]["class_name"].values,
        "similarity": sims_row[topk],
    })
    st.dataframe(df_sim, use_container_width=True)

st.caption("Cosine similarities are computed in the original high-D CLIP space; 2-D maps are for visualization.")
# -----------------------------
