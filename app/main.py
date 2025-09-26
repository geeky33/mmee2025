# app/main.py
# MMEE Embedding Viewer (uses utils/clip_utils)
# Streamlit app: projections (PCA / t-SNE / UMAP) for IMAGE and TEXT embeddings
# + Outlier detection (multiple methods), removal, and clean-view UI
# + Authenticity evaluation vs injected bad captions (Precision/Recall/F1 + Confusion Matrix + Examples)
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

# Silence optional QuickGELU warning if you see it
import warnings
warnings.filterwarnings("ignore", message="QuickGELU mismatch.*")

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

# ---- Captions source toggle + GT load ----
CAPTIONS_NOISE_JSON = CAPTIONS_JSON.with_name("captions_with_noise.json")
BAD_GT_JSON = CAPTIONS_JSON.with_name("bad_caption_gt.json")

with st.sidebar:
    st.header("Captions source")
    options = ["Original"] + (["With noise"] if CAPTIONS_NOISE_JSON.exists() else [])
    captions_choice = st.radio(
        "Select captions file",
        options,
        index=0,
        help="Use tools/add_bad_captions.py to generate captions_with_noise.json + bad_caption_gt.json"
    )

ACTIVE_CAPTIONS_JSON = CAPTIONS_NOISE_JSON if (captions_choice == "With noise" and CAPTIONS_NOISE_JSON.exists()) else CAPTIONS_JSON

BAD_GT: Dict[str, List[int]] = {}
if captions_choice == "With noise" and BAD_GT_JSON.exists():
    try:
        with open(BAD_GT_JSON, "r", encoding="utf-8") as f:
            BAD_GT = json.load(f)  # dict: image_relpath/name -> [0/1,...]
        st.caption("Ground-truth for injected bad captions: found ✅")
    except Exception as e:
        st.warning(f"Failed to load bad_caption_gt.json: {e}")
else:
    st.caption("Ground-truth for bad captions: not active (Original captions).")

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
                if isinstance(meta, list):
                    classes = [str(x) for x in meta if str(x).strip()]
                    class_name = classes[0] if classes else "unlabeled"
                elif isinstance(meta, dict):
                    class_id = meta.get("class_id")
                    class_name = meta.get("class_name") or "unlabeled"
                else:
                    class_name = str(meta) if meta else "unlabeled"

                records.append({
                    "image_path": str(p),
                    "image_relpath": str(p.relative_to(images_root)) if p.is_relative_to(images_root) else p.name,
                    "class_name": class_name,
                    "class_id": int(class_id) if class_id is not None else None,
                })
                used.add(p)

    # Fill any unlabeled with folder-derived class
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
    A = _l2_normalize_np(A); B = _l2_normalize_np(B)
    return (A * B).sum(axis=1).astype(np.float32, copy=False)

@st.cache_data(show_spinner=False)
def project_2d(X: np.ndarray, method: str, random_state: int,
               tsne_perplexity: int, umap_neighbors: int, umap_min_dist: float) -> np.ndarray:
    m = method.lower()
    if m == "pca":
        return pca_project(X, n_components=2)
    if m == "tsne":
        return tsne_project(X, n_components=2, perplexity=int(tsne_perplexity), random_state=int(random_state))
    if m == "umap":
        coords_2d, _ = umap_project(
            X, n_neighbors=int(umap_neighbors), min_dist=float(umap_min_dist), random_state=int(random_state)
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
    if not per_caption and I.shape[0] == T.shape[0]:
        I_n = _l2_normalize_np(I); T_n = _l2_normalize_np(T)
        Ic = I_n - I_n.mean(0, keepdims=True); Tc = T_n - T_n.mean(0, keepdims=True)
        R, s = orthogonal_procrustes(Tc, Ic)
        T_aligned = (Tc @ R) * s + I_n.mean(0, keepdims=True)
        X = np.vstack([I_n, T_aligned])
        Y = project_2d(X, method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist)
        n = I.shape[0]
        return Y[:n], Y[n:]

    if per_caption:
        assert cap_img_idx is not None and len(cap_img_idx) == T.shape[0], \
            "cap_img_idx must map each caption to its parent image index"
        I_n = _l2_normalize_np(I); T_n = _l2_normalize_np(T)
        I_rep = I_n[np.asarray(cap_img_idx, dtype=int), :]  # repeat parent image for each caption
        Ic = I_rep - I_rep.mean(0, keepdims=True); Tc = T_n - T_n.mean(0, keepdims=True)
        R, s = orthogonal_procrustes(Tc, Ic)
        T_aligned = (Tc @ R) * s + I_rep.mean(0, keepdims=True)
        X = np.vstack([I_n, T_aligned])
        Y = project_2d(X, method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist)
        n = I.shape[0]
        return Y[:n], Y[n:]

    # include umap_neighbors (bugfix: avoid passing min_dist twice)
    I_n = _l2_normalize_np(I); T_n = _l2_normalize_np(T)
    X = np.vstack([I_n, T_n])
    Y = project_2d(X, method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist)
    n = I.shape[0]
    return Y[:n], Y[n:]

# -----------------------------
# Sidebar controls (Model / Projection / Outliers / Evaluation)
# -----------------------------
with st.sidebar:
    st.header("Paths")
    st.code(str(IMAGES_DIR), language="bash")
    st.code(str(ACTIVE_CAPTIONS_JSON), language="bash")
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
        caps_limit = st.number_input(
            "Max captions/image (0 = all)",
            min_value=0, value=0, step=1,
            help="We embed the first K captions per image. Set 0 to use all captions."
        )
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
    outlier_size    = st.slider("Outlier marker size", 6, 20, 10, help="Size of the X markers")
    outlier_width   = st.slider("Outlier stroke width", 1, 5, 2)

    st.header("Evaluation (vs GT)")
    show_eval = st.checkbox("Show authenticity metrics & examples", value=True)
    overlay_auth = st.checkbox("Overlay authenticity on plots", value=True)
    n_examples = st.slider("Examples per bucket (TP/FP/FN)", 1, 10, 3)

# -----------------------------
# Load dataset
# -----------------------------
with st.status("Loading dataset (images, labels, captions)…", expanded=False):
    IMG_PATHS = scan_images(IMAGES_DIR)
    if not IMG_PATHS:
        st.error("No images found under data/images.")
        st.stop()
    CAP_MAP = read_captions(ACTIVE_CAPTIONS_JSON)
    DF = read_labels(LABELS_JSON, IMG_PATHS, IMAGES_DIR)

# -----------------------------
# Subset & Sampling controls (dataset-aware)
# -----------------------------
all_classes = sorted(DF["class_name"].unique().tolist())

with st.sidebar:
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
# ---- Tell the user whether we’re using all captions or only first K ----
if per_caption:
    sel_note = "all captions per image" if caps_limit == 0 else f"first {caps_limit} captions/image"
    st.info(f"GT positives computed over current subset and {sel_note}.")

# Build per-caption payload if requested
cap_payload: List[List[str]]
cap_texts: List[str] = []
cap_img_idx: List[int] = []
if per_caption:
    cap_payload = []
    for i, caps in enumerate(ALL_CAPS):
        if not caps:
            continue
        take = caps if caps_limit == 0 else caps[:caps_limit]
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

IMG = st.session_state.IMG_EMB
TXT = st.session_state.TXT_EMB
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
    P_IMG, P_TXT = project_images_and_text(
        IMG, TXT, method, int(random_state), int(tsne_perplexity),
        int(umap_neighbors), float(umap_min_dist),
        per_caption=True, cap_img_idx=cap_img_idx
    )
    IMG_rep = IMG[np.asarray(cap_img_idx, dtype=int), :] if len(cap_img_idx) > 0 \
              else np.zeros((0, IMG.shape[1]), dtype=np.float32)
    J = joint_weighted_embeddings(IMG_rep, TXT, alpha=alpha_joint)
    cos_pairs_caps = _rowwise_cosine(IMG_rep, TXT) if len(IMG_rep) else np.zeros((0,), dtype=np.float32)

    sim_by_img = defaultdict(list)
    for j, i in enumerate(cap_img_idx):
        sim_by_img[i].append(float(cos_pairs_caps[j]))
    cos_per_image = np.full(n_images, np.nan, dtype=np.float32)
    for i, vals in sim_by_img.items():
        cos_per_image[i] = float(np.mean(vals))
else:
    P_IMG, P_TXT = project_images_and_text(
        IMG, TXT, method, int(random_state), int(tsne_perplexity),
        int(umap_neighbors), float(umap_min_dist),
        per_caption=False, cap_img_idx=None
    )
    J = joint_weighted_embeddings(IMG, TXT, alpha=alpha_joint)
    cos_pairs_caps = _rowwise_cosine(IMG, TXT)
    cos_per_image = cos_pairs_caps.copy()

# -----------------------------
# DataFrames for plotting (add caption text + cosine_sim)
# -----------------------------
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
    "caption_short": [""] * n_images,
    "cosine_sim": cos_per_image,
    "anomaly": False,
    "anomaly_score": 0.0,
})

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
        "cosine_sim": cos_pairs_caps,
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
        "cosine_sim": cos_pairs_caps,
        "anomaly": False,
        "anomaly_score": 0.0,
    })

# ---- attach ground-truth 'is_bad' in per-caption mode ----
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

# ---- KPI counters for GT positives (evaluated-set vs all captions for selected images) ----
if per_caption and BAD_GT:
    gt_pos_eval = int(df_txt2["is_bad"].sum())   # positives among the captions we actually evaluate
    selected_rels = set(df_txt2["image_relpath"])
    gt_pos_all = 0
    for rel in selected_rels:
        seq = BAD_GT.get(rel) or BAD_GT.get(Path(rel).name)
        if seq:
            gt_pos_all += int(sum(seq))

    k1, k2 = st.columns(2)

# -----------------------------
# Outlier detection
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

OUT_LABELS = None
OUT_SCORES = None
target_df = None
if run_detection:
    if method_space.startswith("Raw"):
        X_det = J
        target_df = df_txt2 if per_caption or (TXT.shape[0] == df_txt2.shape[0]) else df_img2
    elif "Image" in method_space:
        X_det = P_IMG; target_df = df_img2
    else:
        X_det = P_TXT; target_df = df_txt2

    OUT_LABELS, OUT_SCORES, used_method = run_outlier_detector(X_det)
    target_df["anomaly"] = (OUT_LABELS == 1)
    target_df["anomaly_score"] = OUT_SCORES
else:
    used_method = None

# -----------------------------
# KPIs
# -----------------------------
# number of GT positives visible in the *current* selection (after class/sample filters and caps_limit)
pos_in_sel = int(df_txt2["is_bad"].sum()) if per_caption and ("is_bad" in df_txt2.columns) else None

leftKPI, midKPI, rightKPI = st.columns(3)
with leftKPI:
    st.metric("Total image samples", len(df_img2))
with midKPI:
    st.metric("GT positives in selection", pos_in_sel if pos_in_sel is not None else "—")
with rightKPI:
    flagged = int(target_df["anomaly"].sum()) if (run_detection and target_df is not None) else 0
    st.metric("Outliers flagged", flagged)

if per_caption and ("is_bad" in df_txt2.columns):
    st.caption(f"GT positives computed over current subset and {'all captions' if caps_limit==0 else f'first {caps_limit} captions'}/image.")

# -----------------------------
# Authenticity metrics + Confusion Matrix + Examples + CSVs
# -----------------------------
def show_examples(title: str, df_src: pd.DataFrame, sample_idx: np.ndarray, max_n: int):
    st.markdown(f"**{title}**")
    if len(sample_idx) == 0:
        st.caption("None.")
        return
    picks = sample_idx[:max_n]
    cols = st.columns(min(3, len(picks)))
    for i, idx in enumerate(picks):
        row = df_src.iloc[idx]
        img_path = IMAGES_DIR / row["image_relpath"]
        with cols[i % len(cols)]:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            st.caption(f"**{row['image_relpath']}**")
            st.write(row["caption_short"] or "(no caption)")
            st.code(f"score={row['anomaly_score']:.3f} • cosine={row['cosine_sim']:.3f}", language="text")

metrics_text = None
tp_idx = fp_idx = fn_idx = tn_idx = np.array([], dtype=int)

if show_eval and run_detection and per_caption and ("is_bad" in df_txt2.columns):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    y_true = df_txt2["is_bad"].astype(int).values
    y_pred = df_txt2["anomaly"].astype(int).values

    P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    metrics_text = f"Precision={P:.3f} • Recall={R:.3f} • F1={F1:.3f} • TP={tp} FP={fp} FN={fn} TN={tn}"

    st.info(f"Positives in current subset: **{int(y_true.sum())}** "
            f"(after class/sample filters and {'all' if caps_limit==0 else f'first {caps_limit}'} captions/image)")
    st.success(f"Detection vs GT (per-caption): {metrics_text}")

    # Confusion matrix heatmap
    cm = np.array([[tn, fp],[fn, tp]], dtype=int)
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred 0 (clean)", "Pred 1 (outlier)"],
            y=["True 0 (clean)", "True 1 (bad cap)"],
            text=cm.astype(str),
            texttemplate="%{text}",
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Count: %{z}<extra></extra>"
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix (Per-caption detection vs Ground Truth)",
        xaxis_title="Prediction", yaxis_title="Ground Truth",
        margin=dict(l=10, r=10, t=40, b=10), height=360
    )
    st.plotly_chart(fig_cm, use_container_width=True, theme="streamlit")

    # Bucket indices
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]

    # Qualitative examples
    with st.expander("Qualitative examples (images + captions)"):
        show_examples("✅ True Positives (authentic outliers)", df_txt2, tp_idx, n_examples)
        show_examples("❌ False Positives (flagged but clean)", df_txt2, fp_idx, n_examples)
        show_examples("⚠️ False Negatives (missed bad captions)", df_txt2, fn_idx, n_examples)

    # ----- CSV exports for all 4 buckets -----
    def _csv_for(df_src: pd.DataFrame, idxs: np.ndarray, bucket: str) -> pd.DataFrame:
        if len(idxs) == 0:
            return pd.DataFrame(columns=[
                "bucket","class_name","class_id","image_relpath","caption_short",
                "cosine_sim","anomaly_score","x","y","class_color"
            ])
        sub = df_src.iloc[idxs].copy()
        sub["bucket"] = bucket
        sub["class_color"] = sub["class_name"].map(colors).fillna("")
        return sub[[
            "bucket","class_name","class_id","image_relpath","caption_short",
            "cosine_sim","anomaly_score","x","y","class_color"
        ]]

    df_tp_csv = _csv_for(df_txt2, tp_idx, "TP")
    df_fp_csv = _csv_for(df_txt2, fp_idx, "FP")
    df_fn_csv = _csv_for(df_txt2, fn_idx, "FN")
    df_tn_csv = _csv_for(df_txt2, tn_idx, "TN")
    df_all_csv = pd.concat([df_tp_csv, df_fp_csv, df_fn_csv, df_tn_csv], ignore_index=True)

    with st.expander("📥 Download detections (CSV)"):
        st.download_button(
            "Download ALL buckets (TP/FP/FN/TN)",
            data=df_all_csv.to_csv(index=False),
            file_name="detections_all_buckets.csv",
            mime="text/csv",
            use_container_width=True
        )
        c1,c2,c3,c4 = st.columns(4)
        c1.download_button("TP.csv", df_tp_csv.to_csv(index=False), "tp.csv", "text/csv")
        c2.download_button("FP.csv", df_fp_csv.to_csv(index=False), "fp.csv", "text/csv")
        c3.download_button("FN.csv", df_fn_csv.to_csv(index=False), "fn.csv", "text/csv")
        c4.download_button("TN.csv", df_tn_csv.to_csv(index=False), "tn.csv", "text/csv")

# -----------------------------
# Plotting helpers
# -----------------------------
def _auth_traces_for_text(df: pd.DataFrame):
    """
    Return Plotly traces for text points authenticity when available.
    - True outliers (is_bad==1 & anomaly==1): red X
    - False positives (is_bad==0 & anomaly==1): white X
    - Missed bad (is_bad==1 & anomaly==0): orange diamond-open
    """
    traces = []
    if "is_bad" not in df.columns or "anomaly" not in df.columns:
        return traces

    # True outliers
    sel = (df["is_bad"].astype(int) == 1) & (df["anomaly"].astype(bool))
    true_out = df[sel]
    if not true_out.empty:
        txt = ("🖼 %{customdata[0]}<br>📝 %{customdata[1]}<br>"
               "score=%{customdata[3]:.3f}")
        traces.append(go.Scatter(
            x=true_out["x"], y=true_out["y"], mode="markers", name="True outliers",
            marker=dict(symbol="x", size=10, color="red", line=dict(width=2, color="red")),
            customdata=np.stack([
                true_out["image_relpath"].astype(str).values,
                true_out["caption_short"].fillna("").astype(str).values,
                true_out["cosine_sim"].astype(float).values,
                true_out["anomaly_score"].astype(float).values
            ], axis=1),
            hovertemplate=txt + "<extra>True outlier</extra>",
            showlegend=True,
        ))

    # False positives
    sel = (df["is_bad"].astype(int) == 0) & (df["anomaly"].astype(bool))
    fp = df[sel]
    if not fp.empty:
        txt = ("🖼 %{customdata[0]}<br>📝 %{customdata[1]}<br>"
               "score=%{customdata[3]:.3f}")
        traces.append(go.Scatter(
            x=fp["x"], y=fp["y"], mode="markers", name="False positives",
            marker=dict(symbol="x", size=10, color="white", line=dict(width=2, color="white")),
            customdata=np.stack([
                fp["image_relpath"].astype(str).values,
                fp["caption_short"].fillna("").astype(str).values,
                fp["cosine_sim"].astype(float).values,
                fp["anomaly_score"].astype(float).values
            ], axis=1),
            hovertemplate=txt + "<extra>False positive</extra>",
            showlegend=True,
        ))

    # Missed bad captions (FN)
    sel = (df["is_bad"].astype(int) == 1) & (~df["anomaly"].astype(bool))
    fn = df[sel]
    if not fn.empty:
        txt = ("🖼 %{customdata[0]}<br>📝 %{customdata[1]}<br>"
               "cosine=%{customdata[2]:.3f}")
        traces.append(go.Scatter(
            x=fn["x"], y=fn["y"], mode="markers", name="Missed bad captions",
            marker=dict(symbol="diamond-open", size=9, line=dict(width=2, color="orange")),
            customdata=np.stack([
                fn["image_relpath"].astype(str).values,
                fn["caption_short"].fillna("").astype(str).values,
                fn["cosine_sim"].astype(float).values
            ], axis=1),
            hovertemplate=txt + "<extra>False negative</extra>",
            showlegend=True,
        ))
    return traces

def _outlier_traces_by_class(df: pd.DataFrame, name_prefix="Outlier"):
    """
    Build per-class outlier 'X' traces so that outliers keep the same color as their class.
    Used when authenticity overlays are OFF.
    """
    out = df[df["anomaly"].astype(bool)].copy()
    if out.empty:
        return []

    traces = []
    for cls, g in out.groupby("class_name", sort=False):
        cls_color = colors.get(cls, None)

        custom = np.stack([
            g["class_name"].astype(str).values,
            g["image_relpath"].astype(str).values,
            g["caption_short"].fillna("").astype(str).values,
            g["cosine_sim"].astype(float).values,
            g["anomaly_score"].astype(float).values
        ], axis=1)

        hover = (
            "class=%{customdata[0]}<br>"
            "🖼 %{customdata[1]}<br>"
            "📝 %{customdata[2]}<br>"
            "cosine=%{customdata[3]:.3f}<br>"
            "score=%{customdata[4]:.3f}<extra></extra>"
        )

        traces.append(go.Scatter(
            x=g["x"], y=g["y"], mode="markers", name=f"{name_prefix} · {cls}",
            marker=dict(
                symbol="x", size=outlier_size,
                color=cls_color,                      # class color
                line=dict(width=outlier_width, color=cls_color)
            ),
            customdata=custom,
            hovertemplate=hover,
            showlegend=True,
        ))
    return traces

def _scatter_with_outliers(df: pd.DataFrame, title: str, authenticity_for_text: bool = False):
    hover_cols = [c for c in ["class_name", "class_id", "image_relpath", "caption_short", "cosine_sim"] if c in df.columns]
    mask = df["anomaly"].astype(bool) if "anomaly" in df.columns else pd.Series(False, index=df.index)

    base = px.scatter(
        df[~mask], x="x", y="y", color="class_name",
        hover_data=hover_cols,
        title=title,
        color_discrete_map=colors, opacity=0.85, render_mode="webgl"
    )
    base.update_traces(marker=dict(size=6, line=dict(width=0)))

    if not authenticity_for_text:
        for tr in _outlier_traces_by_class(df, name_prefix="Outlier"):
            base.add_trace(tr)
    else:
        for tr in _auth_traces_for_text(df):
            base.add_trace(tr)

    base.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return base

# -----------------------------
# Side-by-side plots
# -----------------------------
left, right = st.columns(2, gap="large")
with left:
    st.subheader(f"[{dataset_name}] Image embeddings • {method}" + (" • CLEAN" if show_clean_only else ""))
    st.plotly_chart(_scatter_with_outliers(df_img2 if not show_clean_only else df_img2[df_img2["anomaly"]==False],
                                           f"Image • {method}", authenticity_for_text=False),
                    use_container_width=True, theme="streamlit")
with right:
    title_txt = "Text (per-caption)" if per_caption else "Text (per-image)"
    st.subheader(f"[{dataset_name}] {title_txt} embeddings • {method}" + (" • CLEAN" if show_clean_only else ""))
    authenticity = overlay_auth and per_caption and ("is_bad" in df_txt2.columns)
    st.plotly_chart(_scatter_with_outliers(df_txt2 if not show_clean_only else df_txt2[df_txt2["anomaly"]==False],
                                           f"Text • {method}", authenticity_for_text=authenticity),
                    use_container_width=True, theme="streamlit")

# -----------------------------
# Joint chart with connection lines (image ↔ caption)
# -----------------------------
st.subheader(f"[{dataset_name}] Joint {method} (shared axes)" + (" • CLEAN" if show_clean_only else ""))
fig_joint = go.Figure()

# Draw connection lines
if per_caption:
    rel_to_index = {r: i for i, r in enumerate(df_img2["image_relpath"])}
    src_df = df_txt2 if not show_clean_only else df_txt2[df_txt2["anomaly"]==False]
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
    idx_iter = np.where((df_txt2["x"].notna()))[0] if not show_clean_only else np.where((df_txt2["anomaly"]==False).values)[0]
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

fig_joint.add_trace(_base_marker(df_img2, "Image"))
fig_joint.add_trace(_base_marker(df_txt2, "Text (per-caption)" if per_caption else "Text"))

# Overlays on joint plot
if overlay_auth and per_caption and ("is_bad" in df_txt2.columns):
    for tr in _auth_traces_for_text(df_txt2):
        fig_joint.add_trace(tr)
else:
    for tr in _outlier_traces_by_class(df_img2, "Outlier"):
        fig_joint.add_trace(tr)
    for tr in _outlier_traces_by_class(df_txt2, "Outlier"):
        fig_joint.add_trace(tr)

fig_joint.update_layout(margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_joint, use_container_width=True, theme="streamlit")

# -----------------------------
# Single-view: Joint-blend projection (uses α above)
# -----------------------------
with st.expander("Single-view: Joint-blend projection (uses α above)"):
    PJ = project_2d(J, method, int(random_state), int(tsne_perplexity), int(umap_neighbors), float(umap_min_dist))

    if per_caption:
        source_df = df_txt2
        anom_vals = source_df["anomaly"].astype(bool).values if run_detection else np.zeros(len(PJ), bool)
        score_vals = source_df["anomaly_score"].astype(float).values if run_detection else np.zeros(len(PJ), float)
        cap_short_j = [ (c[:120] + "…") if len(c) > 120 else c for c in cap_texts ]
        cos_sim_j = cos_pairs_caps
        classes_j = [DF.iloc[i]["class_name"] for i in cap_img_idx]
        class_id_j = [int(DF.iloc[i]["class_id"]) for i in cap_img_idx]
        rels_j = [DF.iloc[i]["image_relpath"] for i in cap_img_idx]
    else:
        source_df = df_txt2 if (method_space.startswith("Raw") or "Text" in method_space) else df_img2
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
        hover_cols = [c for c in ["class_name","class_id","image_relpath","caption_short","cosine_sim"] if c in df.columns]
        mask = df["anomaly"].astype(bool)
        base = px.scatter(df[~mask], x="x", y="y", color="class_name",
                          hover_data=hover_cols, title=title,
                          color_discrete_map=colors,
                          opacity=0.85, render_mode="webgl")
        base.update_traces(marker=dict(size=6, line=dict(width=0)))

        for tr in _outlier_traces_by_class(df):
            base.add_trace(tr)

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

st.caption("Per-caption mode shows one point per caption for Text and Joint; evaluation compares detector outliers vs bad_caption_gt.json (authenticity).")
