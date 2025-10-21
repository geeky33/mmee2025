"""
CLIP utilities for joint image/text embeddings (OpenCLIP backend).

Public API (cleaned to mentor request):
- detect_device() -> torch.device
- load_open_clip(...) -> (model, preprocess, tokenizer, device)
- compute_image_embeddings(...) -> np.ndarray [N, D]
- compute_text_embeddings(...) -> np.ndarray [N, D]
- encode_texts(...) -> np.ndarray [M, D]
- l2_normalize(x), is_l2_normalized(x), validate_joint_inputs(img, txt)
- cosine_similarity(A, B)
- topk_text_for_image(image_vecs, text_vecs, k=5)

# Joint embedding
- joint_weighted_embeddings(img, txt, alpha=0.5) -> np.ndarray [N, D]

# Dimensionality reduction
- pca_project, tsne_project, umap_project

# Outlier detection
# - *_detect : runs on the entire dataset (global view, all classes together)
# - *_by_class : runs separately per class (local view), normalizes scores, then merges
#   → This is the recommended mode for reducing missed bad captions.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

# -----------------------------
# Optional deps
# -----------------------------
try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None

try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None

try:
    import umap
except Exception:
    umap = None

try:
    from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
except Exception:
    NearestNeighbors = None
    LocalOutlierFactor = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

try:
    import open_clip
except Exception as e:
    open_clip = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

# -----------------------------
# Core helpers
# -----------------------------
def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _assert_open_clip():
    if open_clip is None:
        raise RuntimeError(
            "open_clip_torch not installed. Install it with:\n"
            "  pip install open_clip_torch"
        ) from _IMPORT_ERR

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (x / n).astype(np.float32, copy=False)

def is_l2_normalized(x: np.ndarray, atol: float = 1e-3) -> bool:
    if x.size == 0:
        return True
    norms = np.linalg.norm(x, axis=1)
    return bool(np.all(np.isfinite(norms)) and np.all(np.abs(norms - 1.0) <= atol))

def validate_joint_inputs(img: np.ndarray, txt: np.ndarray, strict: bool = False) -> None:
    if img.shape != txt.shape:
        raise ValueError(f"IMG and TXT must have same shape, got {img.shape} vs {txt.shape}")
    if strict and (not is_l2_normalized(img) or not is_l2_normalized(txt)):
        raise ValueError("Strict validation failed: inputs are not L2-normalized rows.")

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return (A @ B.T).astype(np.float32, copy=False)

# -----------------------------
# Model loading
# -----------------------------
def load_open_clip(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[torch.device] = None,
):
    _assert_open_clip()
    if device is None:
        device = detect_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer, device

# -----------------------------
# Batch helper
# -----------------------------
def _batch(it: Iterable, size: int) -> Iterator[list]:
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

# -----------------------------
# Encoding (images / texts)
# -----------------------------
def compute_image_embeddings(
    image_paths: Sequence[str | Path],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    model, preprocess, _, dev = load_open_clip(model_name, pretrained, device)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for chunk in _batch(map(str, image_paths), batch_size):
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in chunk]
            imgs_t = torch.stack(imgs, dim=0).to(dev, non_blocking=True)
            f = model.encode_image(imgs_t)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    if feats:
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    return np.zeros((0, 512), np.float32)

def encode_texts(
    texts: Sequence[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 256,
    prompt_template: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    model, _, tokenizer, dev = load_open_clip(model_name, pretrained, device)
    if prompt_template:
        texts = [prompt_template.format(t) if t else prompt_template.format("") for t in texts]
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for chunk in _batch(texts, batch_size):
            toks = tokenizer(list(chunk)).to(dev, non_blocking=True)
            f = model.encode_text(toks)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    if feats:
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    return np.zeros((0, 512), np.float32)

def compute_text_embeddings(
    all_captions: Sequence[Sequence[str]],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 256,
    aggregate: str = "average",
    prompt_template: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    flat: List[str] = []
    owners: List[int] = []
    for i, caps in enumerate(all_captions):
        caps = [c for c in caps if c]
        if not caps:
            flat.append("")
            owners.append(i)
            continue
        if aggregate == "first":
            flat.append(caps[0]); owners.append(i)
        else:
            for c in caps: flat.append(c); owners.append(i)
    Z = encode_texts(flat, model_name, pretrained, batch_size, prompt_template, device)
    N, D = len(all_captions), Z.shape[1] if Z.shape[0] else 512
    out = np.zeros((N, D), np.float32); cnt = np.zeros(N, np.int32)
    for vec, owner in zip(Z, owners): out[owner] += vec; cnt[owner] += 1
    cnt[cnt == 0] = 1; out /= cnt[:, None]
    return l2_normalize(out)

# -----------------------------
# Joint / projection helpers
# -----------------------------
def joint_weighted_embeddings(img: np.ndarray, txt: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    validate_joint_inputs(img, txt, strict=False)
    I = l2_normalize(img); T = l2_normalize(txt)
    J = alpha * I + (1.0 - alpha) * T
    return l2_normalize(J)

def pca_project(J: np.ndarray, n_components: int = 2) -> np.ndarray:
    if PCA is None: raise RuntimeError("scikit-learn PCA not installed")
    n_components = min(n_components, J.shape[1])
    return PCA(n_components=n_components).fit_transform(J).astype(np.float32)

def tsne_project(J: np.ndarray, n_components: int = 2, perplexity: float = 30.0,
                 learning_rate: str | float = "auto", random_state: int = 42, init: str = "pca") -> np.ndarray:
    if TSNE is None: raise RuntimeError("scikit-learn TSNE not installed")
    coords = TSNE(n_components=n_components, perplexity=perplexity,
                  learning_rate=learning_rate, random_state=random_state,
                  init=init, metric="cosine").fit_transform(J)
    return coords.astype(np.float32)

def umap_project(J: np.ndarray, n_neighbors: int = 60, min_dist: float = 0.15, random_state: int = 42):
    if umap is None: raise RuntimeError("umap-learn not installed")
    J50 = PCA(n_components=min(50, J.shape[1])).fit_transform(J) if PCA else J
    reducer2 = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric="cosine", random_state=random_state)
    reducer3 = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric="cosine", random_state=random_state)
    return reducer2.fit_transform(J50), reducer3.fit_transform(J50)

# -----------------------------
# Convenience
# -----------------------------
def topk_text_for_image(image_vecs: np.ndarray, text_vecs: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    sims = cosine_similarity(image_vecs, text_vecs)
    if sims.size == 0: return np.zeros((0, k), np.int64), np.zeros((0, k), np.float32)
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-s, axis=1)
    return (np.take_along_axis(idx, order, axis=1).astype(np.int64),
            np.take_along_axis(s, order, axis=1).astype(np.float32))

# -----------------------------
# Outlier detection (4 methods)
# -----------------------------
def iforest_detect(X: np.ndarray, contamination: float = 0.05, n_estimators: int = 300,
                   max_samples: str | int = "auto", random_state: int = 42):
    if IsolationForest is None: raise RuntimeError("IsolationForest not available")
    X = np.asarray(X, np.float32)
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                            max_samples=max_samples, random_state=random_state, n_jobs=-1)
    model.fit(X)
    scores = -model.score_samples(X).astype(np.float32)
    labels = (model.predict(X) == -1).astype(np.int8)
    return labels, scores

def lof_detect(X: np.ndarray, n_neighbors: int = 20, contamination: float = 0.05, scale: bool = True):
    if LocalOutlierFactor is None: raise RuntimeError("LOF not available")
    X = np.asarray(X, np.float32)
    if scale and StandardScaler is not None: X = StandardScaler().fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, max(2, len(X)-1)),
                             contamination=contamination, novelty=False, n_jobs=-1)
    labels_raw = lof.fit_predict(X)
    labels = (labels_raw == -1).astype(np.int8)
    scores = -lof.negative_outlier_factor_.astype(np.float32)
    return labels, scores

def knn_quantile_detect(X: np.ndarray, k: int = 10, quantile: float = 0.98, scale: bool = True):
    if NearestNeighbors is None: raise RuntimeError("NearestNeighbors not available")
    X = np.asarray(X, np.float32)
    if scale and StandardScaler is not None: X = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(k, max(2, len(X)-1))).fit(X)
    dists, _ = nn.kneighbors(X); kth = dists[:, -1].astype(np.float32)
    thr = float(np.quantile(kth, quantile)) if len(kth) else np.inf
    labels = (kth >= thr).astype(np.int8); scores = kth
    return labels, scores
def dbscan_detect(X, eps: float = 0.8, min_samples: int = 10, auto_eps: bool = False, k: int = 10, q: float = 0.95, scale: bool = True):
    if auto_eps and NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=min(k, max(2, len(X)-1))).fit(X)
        dists, _ = nn.kneighbors(X)
        eps = float(np.quantile(dists[:, -1], q))
    from sklearn.cluster import DBSCAN
    X = np.asarray(X, np.float32)
    if scale and StandardScaler is not None: X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels_raw = db.labels_; labels = (labels_raw == -1).astype(np.int8)
    if NearestNeighbors is None:
        scores = labels.astype(np.float32)
    else:
        k = max(2, min_samples); nn = NearestNeighbors(n_neighbors=min(k, max(2, len(X)-1))).fit(X)
        d, _ = nn.kneighbors(X); scores = d[:, -1].astype(np.float32)
    return labels, scores

# -----------------------------
# Per-class wrappers
# -----------------------------
def _detect_by_class(X, classes, fn, normalize_scores=True, **kwargs):
    X = np.asarray(X, np.float32); classes = np.asarray(classes)
    n = len(X); labels = np.zeros(n, np.int8); scores = np.zeros(n, np.float32)
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]; Xi = X[idx]
        if len(idx) < 5: continue
        li, si = fn(Xi, **kwargs)
        if normalize_scores and len(si) > 1:
            m, s = float(np.mean(si)), float(np.std(si) or 1.0); si = (si - m) / s
        labels[idx], scores[idx] = li, si
    return labels, scores

def iforest_by_class(X, classes, **kwargs): return _detect_by_class(X, classes, iforest_detect, **kwargs)
def lof_by_class(X, classes, **kwargs): return _detect_by_class(X, classes, lof_detect, **kwargs)
def knn_quantile_by_class(X, classes, **kwargs): return _detect_by_class(X, classes, knn_quantile_detect, **kwargs)
def dbscan_by_class(X, classes, **kwargs): return _detect_by_class(X, classes, dbscan_detect, **kwargs)

# -----------------------------
# Public API
# -----------------------------
__all__ = [
    "detect_device", "load_open_clip",
    "compute_image_embeddings", "encode_texts", "compute_text_embeddings",
    "cosine_similarity", "topk_text_for_image",
    "l2_normalize", "is_l2_normalized", "validate_joint_inputs",
    "joint_weighted_embeddings", "pca_project", "tsne_project", "umap_project",
    "iforest_detect", "lof_detect", "knn_quantile_detect", "dbscan_detect",
    "iforest_by_class", "lof_by_class", "knn_quantile_by_class", "dbscan_by_class",
]
