# app/main.py
# MMEE Embedding Viewer (uses utils/clip_utils)
# Streamlit app: projections (PCA / t-SNE / UMAP) for IMAGE and TEXT embeddings
# + Outlier detection (multiple methods), removal, and clean-view UI
#
# Usage:
#   streamlit run app/main.py

from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys, json, hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.linalg import orthogonal_procrustes
from importlib import import_module
from collections import defaultdict

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
    def tsne_project(X, n_components=2, perplexity=30, random_state=42):
        from sklearn.manifold import TSNE as _TSNE
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
# Dataset discovery & selection
# -----------------------------
st.set_page_config(page_title="MMEE • Embedding Viewer", layout="wide")
st.title("🕊️ MMEE • Embedding Viewer (data/* + utils/clip_utils)")

APP_DIR   = Path(__file__).resolve().parent
DATA_ROOT = PROJ_DIR / "data"

def discover_datasets(root: Path):
    out = {}
    if root.exists():
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            images = d / "images"
            caps   = d / "captions.json"
            labs   = d / "labels.json"
            if images.exists() and caps.exists() and labs.exists():
                out[d.name] = {"root": d, "images": images, "captions": caps, "labels": labs}
    return out

DATASETS = discover_datasets(DATA_ROOT)
if not DATASETS:
    st.error(f"No datasets found under {DATA_ROOT}. Expected subfolders with images/, captions.json, labels.json.")
    st.stop()

with st.sidebar:
    st.header("Dataset")
    dataset_name = st.selectbox("Choose dataset", list(DATASETS.keys()), index=0)

# Reset embeddings if dataset changed (prevents shape/key mismatches)
prev_ds = st.session_state.get("dataset_name")
if prev_ds != dataset_name:
    st.session_state["dataset_name"] = dataset_name
    for k in ("IMG_EMB", "TXT_EMB", "EMB_SIG", "EMB_PATHS"):
        st.session_state.pop(k, None)

IMAGES_DIR    = DATASETS[dataset_name]["images"]
CAPTIONS_JSON = DATASETS[dataset_name]["captions"]
LABELS_JSON   = DATASETS[dataset_name]["labels"]

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

def read_labels(labels_path: Path, image_paths: List[Path], images_root: Path) -> pd.DataFrame:
    """
    Read labels.json (dict/list) or infer from folder structure.
    - If dict value is a list (multi-label), we use the FIRST label as primary class for coloring.
    - If list is empty (e.g., COCO images without instances), we assign 'unlabeled'.
    """
    records: List[Dict[str, Any]] = []

    by_stem = {p.stem: p for p in image_paths}
    try:
        by_rel = {str(p.relative_to(images_root)): p for p in image_paths}
    except Exception:
        by_rel = {}

    used = set()
    if labels_path.exists():
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
                class_name = row.get("class_name") or _derive_class_from_folder(p, images_root)
                records.append({
                    "image_path": str(p),
                    "image_relpath": str(p.relative_to(images_root)) if p.is_relative_to(images_root) else p.name,
                    "class_name": class_name,
                    "class_id": int(class_id) if class_id is not None else None,
                })
                used.add(p)

        elif isinstance(data, dict):
            for key, meta in data.items():
                p = by_rel.get(key) or by_stem.get(Path(key).stem)
                if not p:
                    continue
                class_id = None
                if isinstance(meta, list):        # multi-label
                    classes = [str(x) for x in meta if str(x).strip()]
                    class_name = classes[0] if classes else "unlabeled"
                elif isinstance(meta, dict):     # explicit mapping
                    class_id = meta.get("class_id")
                    class_name = meta.get("class_name") or "unlabeled"
                else:                             # single string
                    class_name = str(meta) if meta else "unlabeled"

                records.append({
                    "image_path": str(p),
                    "image_relpath": str(p.relative_to(images_root)) if p.is_relative_to(images_root) else p.name,
                    "class_name": class_name,
                    "class_id": int(class_id) if class_id is not None else None,
                })
                used.add(p)

    # Fill any unlabeled with folder-derived class (for datasets like CUB)
    for p in image_paths:
        if p in used:
            continue
        class_name = _derive_class_from_folder(p, images_root)
        records.append({
            "image_path": str(p),
            "image_relpath": str(p.relative_to(images_root)) if p.is_relative_to(images_root) else p.name,
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

def captions_for_images(image_paths: List[Path], captions_map: Dict[str, List[str]], images_root: Path) -> List[List[str]]:
    out: List[List[str]] = []
    for p in image_paths:
        try:
            rel = str(p.relative_to(images_root))
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
# Helpers: signatures & projection
# -----------------------------
def signature_for(df: pd.DataFrame, dataset: str, model: str, weights: str, agg_tag: str) -> str:
    payload = {
        "dataset": dataset,
        "model": model,
        "weights": weights,
        "agg": agg_tag,
        "paths": df["image_relpath"].tolist() if "image_relpath" in df.columns else df["image_path"].tolist(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def _l2_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (X / n).astype(np.float32, copy=False)

def _rowwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity for pairs (Ai, Bi)."""
    A = _l2_normalize_np(A)
    B = _l2_normalize_np(B)
    return (A * B).sum(axis=1).astype(np.float32, copy=False)

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

def project_images_and_text(I: np.ndarray,
                            T: np.ndarray,
                            method: str,
                            random_state: int,
                            tsne_perplexity: int,
                            umap_neighbors: int,
                            umap_min_dist: float,
                            per_caption: bool,
                            cap_img_idx: list[int] | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    If not per_caption (and lengths match): Procrustes-align T to I (one caption per image).
    If per_caption: Procrustes-align each caption vector to its parent image vector
    using cap_img_idx, then project images + aligned captions on a shared 2D basis.
    """
    if not per_caption and I.shape[0] == T.shape[0]:
        I_n = _l2_normalize_np(I)
        T_n = _l2_normalize_np(T)
        Ic = I_n - I_n.mean(0, keepdims=True)
        Tc = T_n - T_n.mean(0, keepdims=True)
        R, s = orthogonal_procrustes(Tc, Ic)
        T_aligned = (Tc @ R) * s + I_n.mean(0, keepdims=True)
        X = np.vstack([I_n, T_aligned])
        Y = project_2d(X, method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist)
        n = I.shape[0]
        return Y[:n], Y[n:]

    if per_caption:
        assert cap_img_idx is not None and len(cap_img_idx) == T.shape[0], \
            "cap_img_idx must map each caption to its parent image index"
        I_n = _l2_normalize_np(I)
        T_n = _l2_normalize_np(T)
        I_rep = I_n[np.asarray(cap_img_idx, dtype=int), :]  # repeat parent image for each caption

        Ic = I_rep - I_rep.mean(0, keepdims=True)
        Tc = T_n - T_n.mean(0, keepdims=True)
        R, s = orthogonal_procrustes(Tc, Ic)
        T_aligned = (Tc @ R) * s + I_rep.mean(0, keepdims=True)

        X = np.vstack([I_n, T_aligned])
        Y = project_2d(X, method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist)
        n = I.shape[0]
        return Y[:n], Y[n:]

    # fallback (shouldn't hit)
    I_n = _l2_normalize_np(I)
    T_n = _l2_normalize_np(T)
    X = np.vstack([I_n, T_n])
    Y = project_2d(X, method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist)
    n = I.shape[0]
    return Y[:n], Y[n:]

# -----------------------------
# Sidebar controls (Model / Projection / Outliers)
# -----------------------------
with st.sidebar:
    st.header("Paths")
    st.code(str(IMAGES_DIR), language="bash")
    st.code(str(CAPTIONS_JSON), language="bash")
    st.code(str(LABELS_JSON), language="bash")

    st.header("Model")
    model_name = st.selectbox("CLIP model", ["ViT-B-32", "ViT-L-14", "ViT-L-14-336"], index=0)
    pretrained = st.selectbox("Weights", ["openai", "laion2b_s34b_b79k", "laion2b_s32b_b82k"], index=0)
    batch_img = st.slider("Image batch size", 8, 128, 64, step=8)

    st.header("Projection")
    method = st.selectbox("Method", ["PCA", "tSNE", "UMAP"], index=0)
    random_state = st.number_input("Random state", value=42, step=1)
    tsne_perplexity = st.slider("t-SNE perplexity", 5, 60, 30)
    umap_neighbors = st.slider("UMAP n_neighbors", 5, 100, 30)
    umap_min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)

    st.header("Captions")
    caption_mode = st.radio("Caption granularity", ["Aggregate per image", "Per-caption points"], index=0)
    per_caption = (caption_mode == "Per-caption points")
    if per_caption:
        caps_limit = st.slider("Captions per image to plot", 1, 10, 5, step=1)
        st.caption("We embed each caption separately and build a joint point for each (image, caption) pair.")
    else:
        text_agg = st.selectbox("Text aggregation", ["average", "first"], index=0)

    st.header("Joint & Outliers")
    alpha_joint = st.slider("Joint blend α (image ↔ text)", 0.0, 1.0, 0.5, step=0.05)
    method_space = st.selectbox(
        "Detection space",
        ["Raw joint (512D)", f"{method} 2D: Image", f"{method} 2D: Text"],
        index=0
    )

    st.header("Outlier method")
    out_method = st.selectbox(
        "Choose method",
        ["Isolation Forest","kNN Distance (Quantile)","LOF (Local Outlier Factor)",
         "DBSCAN (noise)","Mahalanobis (robust)","PCA Reconstruction Error","One-Class SVM"],
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
    outlier_size    = st.slider("Outlier marker size", 6, 20, 10, help="Size of the white X markers")
    outlier_width   = st.slider("Outlier stroke width", 1, 5, 2)

# -----------------------------
# Load dataset
# -----------------------------
with st.status("Loading dataset (images, labels, captions)…", expanded=False):
    IMG_PATHS = scan_images(IMAGES_DIR)
    if not IMG_PATHS:
        st.error("No images found under data/images.")
        st.stop()
    CAP_MAP = read_captions(CAPTIONS_JSON)
    DF = read_labels(LABELS_JSON, IMG_PATHS, IMAGES_DIR)

# -----------------------------
# Subset & sampling controls (dataset-aware)
# -----------------------------
# -----------------------------
# Subset & Sampling controls (dataset-aware)
# -----------------------------
all_classes = sorted(DF["class_name"].unique().tolist())

with st.sidebar:
    st.header("Subset & Sampling")
    n_classes = len(all_classes)

    if n_classes <= 1:
        # Edge case: only one class available
        st.info(f"Only one class detected for [{dataset_name}].")
        max_classes_to_show = 1
        default_classes = all_classes  # that's the only option
    else:
        max_classes_to_show = st.slider(
            "Max classes to include",
            min_value=1,
            max_value=n_classes,                       # strictly > 1 here
            value=min(30, n_classes),
            key=f"max_classes_{dataset_name}"
        )
        default_classes = all_classes[:max_classes_to_show]

    chosen = st.multiselect(
        "Choose classes (optional)",
        all_classes,
        default=default_classes,
        key=f"chosen_classes_{dataset_name}"
    )

    # keep the samples-per-class slider inside a valid range
    # compute an upper bound from the data to make the default meaningful
    max_per_class_in_data = int(DF["class_name"].value_counts().max()) if not DF.empty else 1
    upper_samples = max(1, min(500, max_per_class_in_data))
    default_samples = min(30, upper_samples)

    samples_per_class = st.slider(
        "Samples per class (0 = all)",
        min_value=0,
        max_value=upper_samples,
        value=default_samples,
        step=1,
        key=f"samples_per_class_{dataset_name}"
    )

# Apply class subset + per-class sampling
if chosen:
    DF = DF[DF["class_name"].isin(chosen)].reset_index(drop=True)
else:
    DF = DF[DF["class_name"].isin(default_classes)].reset_index(drop=True)

if samples_per_class > 0:
    DF = DF.groupby("class_name", group_keys=False).apply(
        lambda g: g.sample(min(len(g), samples_per_class), random_state=42)
    ).reset_index(drop=True)


st.success(f"[{dataset_name}] Using {len(DF)} images across {DF['class_name'].nunique()} classes.")

# Align captions per selected image order (per-image list)
ALL_CAPS = captions_for_images([Path(p) for p in DF["image_path"].tolist()], CAP_MAP, IMAGES_DIR)
caps_available = sum(1 for c in ALL_CAPS if c)
st.caption(f"Captions available for {caps_available}/{len(ALL_CAPS)} selected images.")

# Build per-caption payload if requested
cap_payload: List[List[str]]
cap_texts: List[str] = []
cap_img_idx: List[int] = []
if per_caption:
    cap_payload = []
    for i, caps in enumerate(ALL_CAPS):
        if not caps:
            continue
        take = caps[:caps_limit]
        for c in take:
            cap_payload.append([c])  # each item is a single-caption list
            cap_texts.append(c)
            cap_img_idx.append(i)
    n_txt_expected = len(cap_payload)
    agg_tag = f"percap_{caps_limit}"
else:
    cap_payload = ALL_CAPS
    n_txt_expected = len(ALL_CAPS)
    agg_tag = f"agg_{'avg' if 'text_agg' in locals() and text_agg=='average' else 'first'}"

# -------- Invalidate embeddings if subset or mode changed --------
cur_paths = tuple(DF["image_path"].tolist())
prev_paths = st.session_state.get("EMB_PATHS")
if prev_paths != cur_paths:
    for k in ("IMG_EMB", "TXT_EMB", "EMB_SIG"):
        st.session_state.pop(k, None)
    st.session_state["EMB_PATHS"] = cur_paths

CUR_SIG = signature_for(DF, dataset_name, model_name, pretrained, agg_tag)

# ⚡️ Cached wrappers around utils.clip_utils (dataset-aware cache key)
@st.cache_data(show_spinner=False)
def _img_emb_cached(paths: List[str], model: str, weights: str, bs: int, key: str) -> np.ndarray:
    return _compute_image_embeddings(paths, model_name=model, pretrained=weights, batch_size=bs)

@st.cache_data(show_spinner=False)
def _txt_emb_cached(caps: List[List[str]], model: str, weights: str, agg: str, key: str) -> np.ndarray:
    # agg is "average"/"first" in aggregate mode; ignored for per-caption (each list has one string)
    return _compute_text_embeddings(caps, model_name=model, pretrained=weights, aggregate=agg)

# -----------------------------
# Buttons to compute embeddings
# -----------------------------
colA, colB = st.columns(2)
with colA:
    if st.button("🖼️ Compute IMAGE embeddings", use_container_width=True):
        with st.spinner("Encoding images with CLIP…"):
            IMG = _img_emb_cached(DF["image_path"].tolist(), model_name, pretrained, batch_img, CUR_SIG)
            st.session_state.IMG_EMB = IMG
            st.session_state.EMB_SIG = CUR_SIG
            st.success(f"Image embeddings: {IMG.shape}")
with colB:
    if st.button("📝 Compute TEXT embeddings", use_container_width=True):
        with st.spinner("Encoding captions with CLIP…"):
            agg = text_agg if not per_caption else "first"
            TXT = _txt_emb_cached(cap_payload, model_name, pretrained, agg, CUR_SIG)
            st.session_state.TXT_EMB = TXT
            st.session_state.EMB_SIG = CUR_SIG
            st.success(f"Text embeddings: {TXT.shape}")

# Require fresh, matching embeddings
if "IMG_EMB" not in st.session_state or "TXT_EMB" not in st.session_state:
    st.warning("Compute both IMAGE and TEXT embeddings to proceed.")
    st.stop()
if st.session_state.get("EMB_SIG") != CUR_SIG:
    for k in ("IMG_EMB", "TXT_EMB", "EMB_SIG"):
        st.session_state.pop(k, None)
    st.warning("Settings changed. Please recompute IMAGE and TEXT embeddings.")
    st.stop()

IMG = st.session_state.IMG_EMB                   # shape = n_images x d
TXT = st.session_state.TXT_EMB                   # shape = n_images (agg) or n_caps (per-caption)
n_images = len(DF)
if IMG.shape[0] != n_images or TXT.shape[0] != n_txt_expected:
    for k in ("IMG_EMB", "TXT_EMB", "EMB_SIG"):
        st.session_state.pop(k, None)
    st.warning("Subset/caption mode changed. Please recompute IMAGE and TEXT embeddings.")
    st.stop()

# -----------------------------
# Build projections + pairwise cosine similarities
# -----------------------------
if per_caption:
    # project images (unique) and text (per-caption) together, with alignment
    P_IMG, P_TXT = project_images_and_text(
        IMG, TXT, method, int(random_state), int(tsne_perplexity),
        int(umap_neighbors), float(umap_min_dist),
        per_caption=True, cap_img_idx=cap_img_idx
    )
    # joint embedding per caption: repeat each image embedding for each caption and blend
    IMG_rep = IMG[np.asarray(cap_img_idx, dtype=int), :] if len(cap_img_idx) > 0 \
              else np.zeros((0, IMG.shape[1]), dtype=np.float32)
    J = joint_weighted_embeddings(IMG_rep, TXT, alpha=alpha_joint)

    # cosine similarity for (image, caption) pairs (rowwise)
    cos_pairs_caps = _rowwise_cosine(IMG_rep, TXT) if len(IMG_rep) else np.zeros((0,), dtype=np.float32)
    # average per-image cosine for hover on image points
    sim_by_img = defaultdict(list)
    for j, i in enumerate(cap_img_idx):
        sim_by_img[i].append(float(cos_pairs_caps[j]))
    cos_per_image = np.full(n_images, np.nan, dtype=np.float32)
    for i, vals in sim_by_img.items():
        cos_per_image[i] = float(np.mean(vals))
else:
    # regular equal-length joint projection (aggregate)
    P_IMG, P_TXT = project_images_and_text(
        IMG, TXT, method, int(random_state), int(tsne_perplexity),
        int(umap_neighbors), float(umap_min_dist),
        per_caption=False, cap_img_idx=None
    )
    J = joint_weighted_embeddings(IMG, TXT, alpha=alpha_joint)
    cos_pairs_caps = _rowwise_cosine(IMG, TXT)             # one per image
    cos_per_image = cos_pairs_caps.copy()

# -----------------------------
# DataFrames for plotting (add caption text + cosine_sim)
# -----------------------------
class_names_sorted = sorted(DF["class_name"].unique())
colors = {
    name: f"hsl({int(360*i/max(1,len(class_names_sorted)))},70%,50%)"
    for i, name in enumerate(class_names_sorted)
}

# Image points (always one per image)
df_img2 = pd.DataFrame({
    "x": P_IMG[:, 0],
    "y": P_IMG[:, 1],
    "class_name": DF["class_name"].values,
    "class_id": DF["class_id"].values,
    "image_relpath": DF["image_relpath"].values,
    "caption_short": [""] * n_images,
    "cosine_sim": cos_per_image,        # may be NaN in per-caption mode for images without captions
    "anomaly": False,
    "anomaly_score": 0.0,
})

# Text points (per-image OR per-caption)
if per_caption:
    parent_class   = DF["class_name"].values
    parent_id      = DF["class_id"].values
    parent_relpath = DF["image_relpath"].values
    cap_short = [ (c[:120] + "…") if len(c) > 120 else c for c in cap_texts ]
    df_txt2 = pd.DataFrame({
        "x": P_TXT[:, 0],
        "y": P_TXT[:, 1],
        "class_name": [parent_class[i] for i in cap_img_idx],
        "class_id":   [int(parent_id[i]) for i in cap_img_idx],
        "image_relpath": [parent_relpath[i] for i in cap_img_idx],
        "caption_short": cap_short,
        "cosine_sim": cos_pairs_caps,    # one per caption
        "anomaly": False,
        "anomaly_score": 0.0,
    })
else:
    first_caps = [ (caps[0] if caps else "") for caps in cap_payload ]
    cap_short = [ (c[:120] + "…") if len(c) > 120 else c for c in first_caps ]
    df_txt2 = pd.DataFrame({
        "x": P_TXT[:, 0],
        "y": P_TXT[:, 1],
        "class_name": DF["class_name"].values,
        "class_id": DF["class_id"].values,
        "image_relpath": DF["image_relpath"].values,
        "caption_short": cap_short,
        "cosine_sim": cos_pairs_caps,    # one per image
        "anomaly": False,
        "anomaly_score": 0.0,
    })

# -----------------------------
# Outlier detection orchestration
# -----------------------------
def run_outlier_detector(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
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

# Choose feature space + attach anomalies to the correct frame
OUT_LABELS = None
OUT_SCORES = None
target_df = None
if run_detection:
    if method_space.startswith("Raw"):
        # Raw joint space: per-caption → J per caption; aggregate → J per image
        X_det = J
        target_df = df_txt2 if per_caption or (TXT.shape[0] == df_txt2.shape[0]) else df_img2
    elif "Image" in method_space:
        X_det = P_IMG
        target_df = df_img2
    else:  # "Text"
        X_det = P_TXT
        target_df = df_txt2

    OUT_LABELS, OUT_SCORES, used_method = run_outlier_detector(X_det)
    target_df["anomaly"] = (OUT_LABELS == 1)
    target_df["anomaly_score"] = OUT_SCORES
else:
    used_method = None

# Clean mask & cleaned frames (respect which df had anomalies)
def get_clean(df: pd.DataFrame) -> pd.DataFrame:
    if remove_outliers and run_detection:
        return df.loc[~df["anomaly"].astype(bool)].reset_index(drop=True)
    return df

df_img2_clean = get_clean(df_img2)
df_txt2_clean = get_clean(df_txt2)

# KPIs + Download flags (from the target df if detection ran)
leftKPI, rightKPI = st.columns(2)
with leftKPI:
    st.metric("Total image samples", len(df_img2))
with rightKPI:
    flagged = int(target_df["anomaly"].sum()) if (run_detection and target_df is not None) else 0
    st.metric("Outliers flagged", flagged)

download_df = (target_df if (run_detection and target_df is not None) else df_txt2).copy()
download_df = download_df[["image_relpath", "class_name", "class_id", "caption_short", "cosine_sim", "anomaly", "anomaly_score"]]
st.download_button(
    "Download outlier flags (CSV)",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{dataset_name}_outlier_flags.csv",
    mime="text/csv",
)

# -----------------------------
# Plotting helpers
# -----------------------------
def _outlier_trace_from_df(df: pd.DataFrame, name="Outliers"):
    """Create a white X outlier trace with rich hover content from df rows where anomaly==True."""
    out = df[df["anomaly"].astype(bool)].copy()
    if out.empty:
        return None
    # build rich hover text
    txt = (
        "🖼 %{customdata[0]}<br>"  # image_relpath
        "📝 %{customdata[1]}<br>"
        "score=%{customdata[3]:.3f}"
    )
    return go.Scatter(
        x=out["x"], y=out["y"], mode="markers", name=name,
        marker=dict(symbol="x", size=outlier_size, color="white",
                    line=dict(width=outlier_width, color="white")),
        customdata=np.stack([
            out["image_relpath"].astype(str).values,
            out["caption_short"].fillna("").astype(str).values,
            out["cosine_sim"].astype(float).values,
            out["anomaly_score"].astype(float).values
        ], axis=1),
        hovertemplate=txt + "<extra>Outlier</extra>",
        showlegend=True,
    )

def _scatter_with_outliers(df: pd.DataFrame, title: str):
    # include cosine_sim in hover
    hover_cols = [c for c in ["class_id", "image_relpath", "caption_short", "cosine_sim"] if c in df.columns]
    mask = df["anomaly"].astype(bool) if "anomaly" in df.columns else pd.Series(False, index=df.index)

    base = px.scatter(
        df[~mask], x="x", y="y", color="class_name",
        hover_data=hover_cols,
        title=title,
        color_discrete_map=colors, opacity=0.85, render_mode="webgl"
    )
    base.update_traces(marker=dict(size=6, line=dict(width=0)))

    out_trace = _outlier_trace_from_df(df)
    if out_trace is not None:
        base.add_trace(out_trace)

    base.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return base

# -----------------------------
# Side-by-side plots
# -----------------------------
left, right = st.columns(2, gap="large")
with left:
    st.subheader(f"[{dataset_name}] Image embeddings • {method}" + (" • CLEAN" if show_clean_only else ""))
    st.plotly_chart(_scatter_with_outliers(df_img2_clean if show_clean_only else df_img2, f"Image • {method}"),
                    use_container_width=True, theme="streamlit")
with right:
    title_txt = "Text (per-caption)" if per_caption else "Text (per-image)"
    st.subheader(f"[{dataset_name}] {title_txt} embeddings • {method}" + (" • CLEAN" if show_clean_only else ""))
    st.plotly_chart(_scatter_with_outliers(df_txt2_clean if show_clean_only else df_txt2, f"Text • {method}"),
                    use_container_width=True, theme="streamlit")

# -----------------------------
# Joint chart with connection lines (image ↔ caption)
# -----------------------------
st.subheader(f"[{dataset_name}] Joint {method} (shared axes)" + (" • CLEAN" if show_clean_only else ""))
fig_joint = go.Figure()

# Draw connection lines
if per_caption:
    rel_to_index = {r: i for i, r in enumerate(df_img2["image_relpath"])}
    src_df = df_txt2_clean if show_clean_only else df_txt2
    for j in range(len(src_df)):
        rel = src_df.iloc[j]["image_relpath"]
        i = rel_to_index.get(rel, None)
        if i is None:
            continue
        fig_joint.add_trace(go.Scatter(
            x=[df_img2.iloc[i]["x"], src_df.iloc[j]["x"]],
            y=[df_img2.iloc[i]["y"], src_df.iloc[j]["y"]],
            mode="lines",
            line=dict(width=0.5),
            showlegend=False,
            hoverinfo="skip"
        ))
else:
    idx_iter = np.where((df_txt2_clean["x"].notna()))[0] if show_clean_only else range(len(DF))
    for i in idx_iter:
        fig_joint.add_trace(go.Scatter(
            x=[df_img2.iloc[i]["x"], df_txt2.iloc[i]["x"]],
            y=[df_img2.iloc[i]["y"], df_txt2.iloc[i]["y"]],
            mode="lines",
            line=dict(width=0.5),
            showlegend=False,
            hoverinfo="skip"
        ))

# Base markers
def _base_marker(df, name):
    return go.Scatter(
        x=df["x"], y=df["y"], mode="markers", name=name,
        marker=dict(size=6),
        customdata=np.stack([
            df["class_name"].astype(str).values,
            df["image_relpath"].astype(str).values,
            df["caption_short"].fillna("").astype(str).values,
            df["cosine_sim"].astype(float).values
        ], axis=1),
        hovertemplate=(
            "class_name=%{customdata[0]}<br>"
            "x=%{x:.6f}<br>y=%{y:.6f}<br>"
            "image_relpath=%{customdata[1]}<br>"
            "caption_short=%{customdata[2]}<br>"
            "cosine_sim=%{customdata[3]:.3f}<extra></extra>"
        )
    )

fig_joint.add_trace(_base_marker(df_img2_clean if show_clean_only else df_img2, "Image"))
fig_joint.add_trace(_base_marker(df_txt2_clean if show_clean_only else df_txt2,
                                 "Text (per-caption)" if per_caption else "Text"))

# Outlier overlays on joint plot (for whichever side has anomalies)
ot_img = _outlier_trace_from_df(df_img2_clean if show_clean_only else df_img2, "Outliers")
ot_txt = _outlier_trace_from_df(df_txt2_clean if show_clean_only else df_txt2, "Outliers")
if ot_img is not None: fig_joint.add_trace(ot_img)
if ot_txt is not None: fig_joint.add_trace(ot_txt)

fig_joint.update_layout(margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_joint, use_container_width=True, theme="streamlit")

# -----------------------------
# Optional: 2D projection of blended joint vectors (one point per item)
# -----------------------------
with st.expander("Single-view: Joint-blend projection (uses α above)"):
    PJ = project_2d(J, method, int(random_state), int(tsne_perplexity), int(umap_neighbors), float(umap_min_dist))

    if per_caption:
        anomaly_src = (df_txt2 if not show_clean_only else df_txt2_clean)
        anom_vals = anomaly_src["anomaly"].astype(bool).values if run_detection else np.zeros(len(PJ), bool)
        score_vals = anomaly_src["anomaly_score"].astype(float).values if run_detection else np.zeros(len(PJ), float)
        cap_short_j = [ (c[:120] + "…") if len(c) > 120 else c for c in cap_texts ]
        cos_sim_j = cos_pairs_caps
        classes_j = [DF.iloc[i]["class_name"] for i in cap_img_idx]
        class_id_j = [int(DF.iloc[i]["class_id"]) for i in cap_img_idx]
        rels_j = [DF.iloc[i]["image_relpath"] for i in cap_img_idx]
    else:
        # choose anomaly source depending on detection space
        source_df = df_txt2 if (method_space.startswith("Raw") or "Text" in method_space) else df_img2
        source_df = (source_df if not show_clean_only else get_clean(source_df))
        anom_vals = source_df["anomaly"].astype(bool).values if run_detection else np.zeros(len(PJ), bool)
        score_vals = source_df["anomaly_score"].astype(float).values if run_detection else np.zeros(len(PJ), float)
        first_caps = [ (caps[0] if caps else "") for caps in cap_payload ]
        cap_short_j = [ (c[:120] + "…") if len(c) > 120 else c for c in first_caps ]
        cos_sim_j = cos_pairs_caps
        classes_j = DF["class_name"].values
        class_id_j = DF["class_id"].values
        rels_j = DF["image_relpath"].values

    df_joint = pd.DataFrame({
        "x": PJ[:,0], "y": PJ[:,1],
        "class_name": classes_j,
        "class_id":   class_id_j,
        "image_relpath": rels_j,
        "caption_short": cap_short_j,
        "cosine_sim": cos_sim_j,
        "anomaly": anom_vals,
        "anomaly_score": score_vals,
    })

    def _scatter(df: pd.DataFrame, title: str):
        hover_cols = [c for c in ["class_id","image_relpath","caption_short","cosine_sim"] if c in df.columns]
        mask = df["anomaly"].astype(bool)
        base = px.scatter(df[~mask], x="x", y="y", color="class_name",
                          hover_data=hover_cols, title=title,
                          opacity=0.85, render_mode="webgl")
        base.update_traces(marker=dict(size=6, line=dict(width=0)))

        out_trace = _outlier_trace_from_df(df)
        if out_trace is not None:
            base.add_trace(out_trace)

        base.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        return base

    st.plotly_chart(_scatter(df_joint, f"Joint-blend • {method}"), use_container_width=True, theme="streamlit")

# -----------------------------
# Cosine similarity sanity check (CLIP space)
# -----------------------------
st.divider()
st.markdown("### 🔎 Optional: Cosine similarity sanity check (CLIP space)")
idx = st.slider("Pick an image index", 0, len(IMG)-1, 0)
if st.button("Top-5 closest texts for this image (cosine)"):
    sims_row = cosine_similarity(IMG[idx:idx+1], TXT)[0]
    topk = np.argsort(-sims_row)[:5]
    if per_caption:
        rels = [DF.iloc[cap_img_idx[i]]["image_relpath"] for i in topk]
        caps_show = [cap_texts[i] for i in topk]
        classes = [DF.iloc[cap_img_idx[i]]["class_name"] for i in topk]
    else:
        rels = DF["image_relpath"].values[topk]
        caps_show = [ (cap_payload[i][0] if cap_payload[i] else "") for i in topk ]
        classes = DF["class_name"].values[topk]
    df_sim = pd.DataFrame({
        "rank": np.arange(1, len(topk)+1),
        "image_relpath": rels,
        "class_name": classes,
        "caption": caps_show,
        "cosine_sim": sims_row[topk],
    })
    st.dataframe(df_sim, use_container_width=True)

st.caption("Per-caption mode shows one point per caption for Text and Joint; hover on outliers shows the actual caption, image name, cosine similarity, and the detector's outlier score.")
# -----------------------------
