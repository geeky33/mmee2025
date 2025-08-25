# utils/clip_utils.py
"""
CLIP utilities for joint image/text embeddings (OpenCLIP backend).

Public API:
- detect_device() -> torch.device
- load_open_clip(model_name="ViT-B-32", pretrained="openai") -> (model, preprocess, tokenizer, device)
- compute_image_embeddings(image_paths, ...) -> np.ndarray [N, D]
- compute_text_embeddings(all_captions, ...) -> np.ndarray [N, D]
- encode_texts(texts, ...) -> np.ndarray [M, D]
- l2_normalize(x) -> np.ndarray
- is_l2_normalized(x, atol=1e-3) -> bool
- validate_joint_inputs(img, txt, strict=False) -> None
- cosine_similarity(A, B) -> np.ndarray [A.shape[0], B.shape[0]]
- topk_text_for_image(image_vecs, text_vecs, k=5) -> (idx, scores)

# Joint embedding
- joint_weighted_embeddings(img, txt, alpha=0.5) -> np.ndarray [N, D]

# Dimensionality reduction
- pca_project(J, n_components=2) -> np.ndarray [N, 2]
- tsne_project(J, n_components=2, perplexity=30.0, learning_rate='auto', random_state=42, init='pca') -> np.ndarray [N, 2]
- umap_project(J, n_neighbors=60, min_dist=0.15, random_state=42) -> (coords_2d [N,2], coords_3d [N,3])

# Outlier detection (Isolation Forest + others)
- iforest_detect(X, contamination=0.05, ...) -> (labels [N], scores [N]) or (..., model)
- iforest_on_raw(E, **kwargs)
- iforest_on_pca(E, n_components=2, **kwargs) -> (coords, labels, scores[, model])
- iforest_on_tsne(E, n_components=2, **kwargs) -> (coords, labels, scores[, model])
- iforest_on_umap2d(E, **kwargs) -> (coords_2d, labels, scores[, model])
- iforest_on_umap3d(E, **kwargs) -> (coords_3d, labels, scores[, model])

- knn_quantile_detect(X, k=10, quantile=0.98, scale=True)
- lof_detect(X, n_neighbors=20, contamination=0.05, scale=True)
- dbscan_detect(X, eps=0.8, min_samples=10, scale=True)
- mahalanobis_detect(X, contamination=0.05, robust=True)
- pca_recon_error_detect(X, n_components=0.9, contamination=0.05, scale=True)
- ocsvm_detect(X, nu=0.05, kernel='rbf', gamma='scale', scale=True)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

# Optional deps for DR / anomaly detection
try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None  # type: ignore

try:
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover
    TSNE = None  # type: ignore

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover
    IsolationForest = None  # type: ignore

try:
    import umap  # umap-learn
except Exception:  # pragma: no cover
    umap = None  # type: ignore

try:
    from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
except Exception:  # pragma: no cover
    NearestNeighbors = None  # type: ignore
    LocalOutlierFactor = None  # type: ignore

try:
    from sklearn.covariance import MinCovDet, EmpiricalCovariance
except Exception:  # pragma: no cover
    MinCovDet = None  # type: ignore
    EmpiricalCovariance = None  # type: ignore

try:
    from sklearn.svm import OneClassSVM
except Exception:  # pragma: no cover
    OneClassSVM = None  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    StandardScaler = None  # type: ignore

# Lazy import for OpenCLIP with helpful error
try:
    import open_clip  # type: ignore
except Exception as e:  # pragma: no cover
    open_clip = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


# -----------------------------
# Core helpers
# -----------------------------

def detect_device() -> torch.device:
    """Pick best available device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _assert_open_clip():
    if open_clip is None:  # pragma: no cover
        raise RuntimeError(
            "open_clip_torch not installed. Install it with:\n"
            "  pip install open_clip_torch"
        ) from _IMPORT_ERR


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (x / n).astype(np.float32, copy=False)


def is_l2_normalized(x: np.ndarray, atol: float = 1e-3) -> bool:
    """Return True if all row norms are ~1 within tolerance."""
    if x.size == 0:
        return True
    norms = np.linalg.norm(x, axis=1)
    return bool(np.all(np.isfinite(norms)) and np.all(np.abs(norms - 1.0) <= atol))


def validate_joint_inputs(img: np.ndarray, txt: np.ndarray, strict: bool = False) -> None:
    """Validate shape compatibility; optionally require inputs to be L2-normalized."""
    if img.shape != txt.shape:
        raise ValueError(f"IMG and TXT must have same shape, got {img.shape} vs {txt.shape}")
    if strict and (not is_l2_normalized(img) or not is_l2_normalized(txt)):
        raise ValueError("Strict validation failed: inputs are not L2-normalized rows.")


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalized embeddings."""
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
    """Load an OpenCLIP model + preprocess + tokenizer."""
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
    """Encode images into CLIP space (L2-normalized)."""
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
        X = np.concatenate(feats, axis=0)
        return X.astype(np.float32, copy=False)
    return np.zeros((0, 512), np.float32)  # safe default for ViT-B/32


def encode_texts(
    texts: Sequence[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 256,
    prompt_template: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Encode a list of raw texts into CLIP space (L2-normalized)."""
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
        X = np.concatenate(feats, axis=0)
        return X.astype(np.float32, copy=False)
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
    """Encode per-image captions and aggregate to one text vector per image."""
    # Flatten
    flat: List[str] = []
    owners: List[int] = []
    for i, caps in enumerate(all_captions):
        caps = [c for c in caps if c]
        if not caps:
            flat.append("")
            owners.append(i)
            continue
        if aggregate == "first":
            flat.append(caps[0])
            owners.append(i)
        else:  # "average"
            for c in caps:
                flat.append(c)
                owners.append(i)

    Z = encode_texts(
        flat,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        prompt_template=prompt_template,
        device=device,
    )

    # Aggregate back to N (mean then re-normalize)
    N = len(all_captions)
    if Z.shape[0] == 0:
        return np.zeros((N, 512), dtype=np.float32)

    D = Z.shape[1]
    out = np.zeros((N, D), dtype=np.float32)
    cnt = np.zeros((N,), dtype=np.int32)
    for vec, owner in zip(Z, owners):
        out[owner] += vec
        cnt[owner] += 1
    cnt[cnt == 0] = 1
    out /= cnt[:, None]
    return l2_normalize(out)


# -----------------------------
# Joint / projection helpers
# -----------------------------

def joint_weighted_embeddings(img: np.ndarray, txt: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    joint = normalize( alpha * normalize(img) + (1 - alpha) * normalize(txt) )
    alpha in [0,1]: 1 → pure image; 0 → pure text.
    """
    validate_joint_inputs(img, txt, strict=False)
    I = l2_normalize(img)
    T = l2_normalize(txt)
    J = alpha * I + (1.0 - alpha) * T
    return l2_normalize(J)


def pca_project(J: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA projection to n_components. Requires scikit-learn."""
    if PCA is None:  # pragma: no cover
        raise RuntimeError("scikit-learn not installed. Install it with:\n  pip install scikit-learn")
    n_components = min(n_components, J.shape[1])
    coords = PCA(n_components=n_components).fit_transform(J.astype(np.float32, copy=False))
    return coords.astype(np.float32, copy=False)


def tsne_project(
    J: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    random_state: int = 42,
    init: str = "pca",
) -> np.ndarray:
    """t-SNE projection to n_components using cosine metric."""
    if TSNE is None:  # pragma: no cover
        raise RuntimeError("scikit-learn not installed. Install it with:\n  pip install scikit-learn")
    coords = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
        init=init,
        metric="cosine",
    ).fit_transform(J.astype(np.float32, copy=False))
    return coords.astype(np.float32, copy=False)


def umap_project(
    J: np.ndarray,
    n_neighbors: int = 60,
    min_dist: float = 0.15,
    random_state: int = 42,
):
    """
    UMAP → 2D and 3D (cosine metric; optional PCA-50 pre-step for stability on larger sets).
    Returns: (coords_2d [N,2], coords_3d [N,3])
    """
    if umap is None:  # pragma: no cover
        raise RuntimeError("umap-learn not installed. Install it with:\n  pip install umap-learn")

    # Optional PCA→UMAP stabilizer
    if PCA is None:  # pragma: no cover
        J50 = J
    else:
        k = min(50, J.shape[1]) if J.shape[1] > 2 else J.shape[1]
        J50 = PCA(n_components=k).fit_transform(J)

    reducer2 = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=random_state
    )
    coords_2d = reducer2.fit_transform(J50)

    reducer3 = umap.UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=random_state
    )
    coords_3d = reducer3.fit_transform(J50)

    return coords_2d, coords_3d


# -----------------------------
# Convenience: top-k retrieval
# -----------------------------

def topk_text_for_image(
    image_vecs: np.ndarray,  # [Ni, D], L2-normalized
    text_vecs: np.ndarray,   # [Nt, D], L2-normalized
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k texts per image by cosine similarity."""
    sims = cosine_similarity(image_vecs, text_vecs)  # [Ni, Nt]
    if sims.size == 0:
        return np.zeros((0, k), dtype=np.int64), np.zeros((0, k), dtype=np.float32)

    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-s, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    s_sorted = np.take_along_axis(s, order, axis=1)
    return idx_sorted.astype(np.int64, copy=False), s_sorted.astype(np.float32, copy=False)


# -----------------------------
# Outlier detection: Isolation Forest
# -----------------------------

def iforest_detect(
    X: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 300,
    max_samples: str | int = "auto",
    random_state: int = 42,
    return_model: bool = False,
):
    """
    Isolation Forest anomaly detection on feature space X.

    Returns
    -------
    (labels, scores) or (labels, scores, model)
      labels: int8 array [N], 1 = outlier, 0 = inlier
      scores: float32 array [N], higher = more anomalous
    """
    if IsolationForest is None:  # pragma: no cover
        raise RuntimeError("scikit-learn not installed. Install it with:\n  pip install scikit-learn")

    X = np.asarray(X, dtype=np.float32, order="C")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    # sklearn's score_samples: higher => more normal. Flip so higher => more anomalous.
    scores = (-model.score_samples(X)).astype(np.float32, copy=False)
    labels = (model.predict(X) == -1).astype(np.int8, copy=False)  # -1 outlier, 1 inlier
    if return_model:
        return labels, scores, model
    return labels, scores


def iforest_on_raw(E: np.ndarray, **kwargs):
    """Isolation Forest directly on the (high-D) embedding space E."""
    return iforest_detect(E, **kwargs)


def iforest_on_pca(
    E: np.ndarray,
    n_components: int = 2,
    return_model: bool = False,
    **kwargs,
):
    """PCA -> Isolation Forest. Returns (coords, labels, scores[, model])"""
    coords = pca_project(E, n_components=n_components)
    out = iforest_detect(coords, return_model=return_model, **kwargs)
    return (coords, *out)


def iforest_on_tsne(
    E: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    random_state: int = 42,
    init: str = "pca",
    return_model: bool = False,
    **kwargs,
):
    """t-SNE -> Isolation Forest. Returns (coords, labels, scores[, model])"""
    coords = tsne_project(
        E,
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
        init=init,
    )
    out = iforest_detect(coords, return_model=return_model, **kwargs)
    return (coords, *out)


def iforest_on_umap2d(
    E: np.ndarray,
    n_neighbors: int = 60,
    min_dist: float = 0.15,
    random_state: int = 42,
    return_model: bool = False,
    **kwargs,
):
    """UMAP(2D) -> Isolation Forest. Returns (coords_2d, labels, scores[, model])"""
    coords_2d, _ = umap_project(E, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    out = iforest_detect(coords_2d, return_model=return_model, **kwargs)
    return (coords_2d, *out)


def iforest_on_umap3d(
    E: np.ndarray,
    n_neighbors: int = 60,
    min_dist: float = 0.15,
    random_state: int = 42,
    return_model: bool = False,
    **kwargs,
):
    """UMAP(3D) -> Isolation Forest. Returns (coords_3d, labels, scores[, model])"""
    _, coords_3d = umap_project(E, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    out = iforest_detect(coords_3d, return_model=return_model, **kwargs)
    return (coords_3d, *out)


# -----------------------------
# Additional detectors
# -----------------------------

def knn_quantile_detect(
    X: np.ndarray,
    k: int = 10,
    quantile: float = 0.98,
    scale: bool = True,
):
    """k-NN distance thresholding (top quantile = outliers)."""
    if NearestNeighbors is None:
        raise RuntimeError("scikit-learn not installed (NearestNeighbors needed).")
    X = np.asarray(X, dtype=np.float32, order="C")
    if scale and StandardScaler is not None:
        X = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(k, max(2, len(X)-1)), metric="minkowski")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)  # distances to neighbors
    kth = dists[:, -1].astype(np.float32, copy=False)
    thr = float(np.quantile(kth, quantile))
    labels = (kth >= thr).astype(np.int8, copy=False)
    scores = kth  # larger = more anomalous
    return labels, scores


def lof_detect(
    X: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.05,
    scale: bool = True,
):
    """Local Outlier Factor: 1=outlier labels, positive scores (higher=more outlying)."""
    if LocalOutlierFactor is None:
        raise RuntimeError("scikit-learn not installed (LocalOutlierFactor needed).")
    X = np.asarray(X, dtype=np.float32, order="C")
    if scale and StandardScaler is not None:
        X = StandardScaler().fit_transform(X)
    lof = LocalOutlierFactor(
        n_neighbors=min(n_neighbors, max(2, len(X)-1)),
        contamination=contamination,
        novelty=False,
        metric="minkowski",
    )
    labels_raw = lof.fit_predict(X)  # -1 outlier, 1 inlier
    labels = (labels_raw == -1).astype(np.int8, copy=False)
    scores = (-lof.negative_outlier_factor_).astype(np.float32, copy=False)  # flip sign
    return labels, scores


def dbscan_detect(
    X: np.ndarray,
    eps: float = 0.8,
    min_samples: int = 10,
    scale: bool = True,
):
    """DBSCAN: label noise (-1) as outliers. Score = kNN distance as proxy."""
    try:
        from sklearn.cluster import DBSCAN
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn not installed (DBSCAN needed).") from e

    X = np.asarray(X, dtype=np.float32, order="C")
    if scale and StandardScaler is not None:
        X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="euclidean").fit(X)
    labels_raw = db.labels_
    labels = (labels_raw == -1).astype(np.int8, copy=False)

    if NearestNeighbors is None:
        scores = labels.astype(np.float32, copy=False)
    else:
        k = max(2, int(min_samples))
        nn = NearestNeighbors(n_neighbors=min(k, max(2, len(X)-1)))
        nn.fit(X)
        d, _ = nn.kneighbors(X)
        scores = d[:, -1].astype(np.float32, copy=False)
    return labels, scores


def mahalanobis_detect(
    X: np.ndarray,
    contamination: float = 0.05,
    robust: bool = True,
):
    """Mahalanobis distance with robust MinCovDet (fallback to Empirical)."""
    if MinCovDet is None or EmpiricalCovariance is None:
        raise RuntimeError("scikit-learn not installed (covariance needed).")
    X = np.asarray(X, dtype=np.float32, order="C")
    if StandardScaler is not None:
        X = StandardScaler().fit_transform(X)
    cov = MinCovDet().fit(X) if robust else EmpiricalCovariance().fit(X)
    d2 = cov.mahalanobis(X).astype(np.float32, copy=False)  # squared distance
    thr = float(np.quantile(d2, 1.0 - contamination))
    labels = (d2 >= thr).astype(np.int8, copy=False)
    scores = d2
    return labels, scores


def pca_recon_error_detect(
    X: np.ndarray,
    n_components: int | float = 0.9,
    contamination: float = 0.05,
    scale: bool = True,
):
    """Reconstruction error from PCA as outlier score."""
    if PCA is None:
        raise RuntimeError("scikit-learn not installed (PCA needed).")
    X = np.asarray(X, dtype=np.float32, order="C")
    scaler = None
    if scale and StandardScaler is not None:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
    else:
        Xs = X
    p = PCA(n_components=n_components).fit(Xs)
    Xh = p.inverse_transform(p.transform(Xs))
    err = np.mean((Xs - Xh) ** 2, axis=1).astype(np.float32, copy=False)
    thr = float(np.quantile(err, 1.0 - contamination))
    labels = (err >= thr).astype(np.int8, copy=False)
    scores = err
    return labels, scores


def ocsvm_detect(
    X: np.ndarray,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    scale: bool = True,
):
    """One-Class SVM: -1 outlier → label 1; decision_function negated as score."""
    if OneClassSVM is None:
        raise RuntimeError("scikit-learn not installed (OneClassSVM needed).")
    X = np.asarray(X, dtype=np.float32, order="C")
    if scale and StandardScaler is not None:
        X = StandardScaler().fit_transform(X)
    clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    y = clf.fit_predict(X)  # -1 outlier, 1 inlier
    labels = (y == -1).astype(np.int8, copy=False)
    scores = (-clf.decision_function(X)).astype(np.float32, copy=False)  # higher = more anomalous
    return labels, scores


__all__ = [
    "detect_device",
    "load_open_clip",
    "compute_image_embeddings",
    "encode_texts",
    "compute_text_embeddings",
    "cosine_similarity",
    "topk_text_for_image",
    "l2_normalize",
    "is_l2_normalized",
    "validate_joint_inputs",
    "joint_weighted_embeddings",
    "pca_project",
    "tsne_project",
    "umap_project",
    "iforest_detect",
    "iforest_on_raw",
    "iforest_on_pca",
    "iforest_on_tsne",
    "iforest_on_umap2d",
    "iforest_on_umap3d",
    "knn_quantile_detect",
    "lof_detect",
    "dbscan_detect",
    "mahalanobis_detect",
    "pca_recon_error_detect",
    "ocsvm_detect",
]
