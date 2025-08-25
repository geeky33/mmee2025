# MMEE CUB-200 Embedding Viewer (uses utils/clip_utils)
# Streamlit app: side-by-side 2D projections (PCA / t-SNE / UMAP) for
# IMAGE and TEXT embeddings computed with CLIP (OpenCLIP by default)
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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap  # umap-learn
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# -- ensure project root (parent of app/) is on sys.path
PROJ_DIR = Path(__file__).resolve().parents[1]
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

# 🔗 Import reusable CLIP helpers (OpenCLIP)
from utils.clip_utils import (
    compute_image_embeddings as _compute_image_embeddings,
    compute_text_embeddings as _compute_text_embeddings,
    cosine_similarity,
)

# -----------------------------
# Paths / Project layout
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJ_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
CAPTIONS_JSON = DATA_DIR / "captions.json"
LABELS_JSON = DATA_DIR / "labels.json"

# Optional on-disk caches dir (you can wire this later)
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
                    # allow plain string label, e.g., "Cedar Waxwing"
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
# Projection helpers
# -----------------------------

@st.cache_data(show_spinner=False)
def project_2d(X: np.ndarray, method: str, random_state: int,
               tsne_perplexity: int, umap_neighbors: int, umap_min_dist: float) -> np.ndarray:
    m = method.lower()

    if m == "pca":
        return PCA(n_components=2).fit_transform(X)

    if m == "tsne":
        # make perplexity valid for small N; TSNE requires perplexity < n_samples
        max_perp = max(5, min(tsne_perplexity, max(5, X.shape[0] - 1)))
        from sklearn.manifold import TSNE as _TSNE
        base = dict(n_components=2, perplexity=max_perp, init="pca", random_state=random_state)

        # Try a few kwargs combos to support different sklearn versions
        for extra in (
            dict(learning_rate="auto", n_iter=1000),    # older API
            dict(learning_rate=200.0,   n_iter=1000),   # older API w/ float lr
            dict(learning_rate="auto",  max_iter=1000), # newer API
            dict(learning_rate=200.0,   max_iter=1000),
            dict(learning_rate=200.0),                  # last resort: rely on defaults
            {}
        ):
            try:
                return _TSNE(**base, **extra).fit_transform(X)
            except TypeError:
                continue
        raise RuntimeError("t-SNE init failed: unsupported sklearn TSNE signature on this install.")

    if m == "umap":
        if not HAS_UMAP:
            raise RuntimeError("umap-learn not installed. Install it with:\n  pip install umap-learn")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
            metric="cosine",           # IMPORTANT for L2-normalized CLIP vectors
            random_state=int(random_state),
        )
        return reducer.fit_transform(X)

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
    T_aligned = T
    X = np.vstack([IMG, T_aligned])
    Y = project_2d(X, method, random_state, tsne_perplexity, umap_neighbors, umap_min_dist)
    n = IMG.shape[0]
    return Y[:n], Y[n:]

# -----------------------------
# Streamlit UI
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

    st.header("Filter / Sampling")
    max_classes_to_show = st.slider("Max classes", 5, 200, 30)
    samples_per_class = st.slider("Samples per class", 5, 100, 30)

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
    chosen = st.multiselect("Choose classes (optional)", all_classes, default=all_classes[:max_classes_to_show])

if chosen:
    DF = DF[DF["class_name"].isin(chosen)].reset_index(drop=True)
else:
    DF = DF[DF["class_name"].isin(all_classes[:max_classes_to_show])].reset_index(drop=True)

if samples_per_class > 0:
    DF = DF.groupby("class_name", group_keys=False).apply(
        lambda g: g.sample(min(len(g), samples_per_class), random_state=42)
    ).reset_index(drop=True)

st.write(f"Using **{len(DF)}** images across **{DF['class_name'].nunique()}** classes.")

# Align captions per selected image order
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

# Joint projection (shared basis)
with st.spinner(f"Projecting jointly with {method}…"):
    P_IMG, P_TXT = joint_project_2d(
        IMG, TXT, method, int(random_state), int(tsne_perplexity),
        int(umap_neighbors), float(umap_min_dist)
    )

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
})

df_txt2 = pd.DataFrame({
    "x": P_TXT[:, 0],
    "y": P_TXT[:, 1],
    "class_name": DF["class_name"].values,
    "class_id": DF["class_id"].values,
    "image_relpath": DF["image_relpath"].values,
})

# Side-by-side plots (same axes now)
left, right = st.columns(2, gap="large")
with left:
    st.subheader(f"Image embeddings • {method}")
    fig_img = px.scatter(
        df_img2, x="x", y="y", color="class_name",
        hover_data=["class_id", "image_relpath"],
        title=f"Image • {method}",
        color_discrete_map=colors, opacity=0.85, render_mode="webgl"
    )
    fig_img.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig_img.update_traces(marker=dict(size=6, line=dict(width=0)))
    st.plotly_chart(fig_img, use_container_width=True, theme="streamlit")

with right:
    st.subheader(f"Text embeddings • {method}")
    fig_txt = px.scatter(
        df_txt2, x="x", y="y", color="class_name",
        hover_data=["class_id", "image_relpath"],
        title=f"Text • {method}",
        color_discrete_map=colors, opacity=0.85, render_mode="webgl"
    )
    fig_txt.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig_txt.update_traces(marker=dict(size=6, line=dict(width=0)))
    st.plotly_chart(fig_txt, use_container_width=True, theme="streamlit")

# Optional: single joint chart with connection lines
st.subheader(f"Joint {method} (shared axes)")
fig_joint = go.Figure()
# connection lines
for i in range(len(DF)):
    fig_joint.add_trace(go.Scatter(
        x=[P_IMG[i,0], P_TXT[i,0]],
        y=[P_IMG[i,1], P_TXT[i,1]],
        mode="lines",
        line=dict(width=0.5),
        showlegend=False,
        hoverinfo="skip"
    ))
# points
fig_joint.add_trace(go.Scatter(
    x=P_IMG[:,0], y=P_IMG[:,1],
    mode="markers", name="Image",
    marker=dict(size=6),
    text=DF["class_name"],
    hovertemplate="Image • %{text}<extra></extra>"
))
fig_joint.add_trace(go.Scatter(
    x=P_TXT[:,0], y=P_TXT[:,1],
    mode="markers", name="Text",
    marker=dict(size=6, symbol="diamond"),
    text=DF["class_name"],
    hovertemplate="Text • %{text}<extra></extra>"
))
fig_joint.update_layout(margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_joint, use_container_width=True, theme="streamlit")

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