# utils/clip_utils.py
"""
CLIP utilities for joint image/text embeddings (OpenCLIP backend).

Public API:
- detect_device() -> torch.device
- load_open_clip(model_name="ViT-B-32", pretrained="openai") -> (model, preprocess, tokenizer, device)
- compute_image_embeddings(image_paths, model_name, pretrained, batch_size=64) -> np.ndarray [N, D]
- compute_text_embeddings(all_captions, model_name, pretrained, batch_size=256, aggregate="average", prompt_template=None) -> np.ndarray [N, D]
- encode_texts(texts, model_name, pretrained, batch_size=256, prompt_template=None) -> np.ndarray [M, D]
- joint_weighted_embeddings(img, txt, alpha=0.5) -> np.ndarray [N, D]
- umap_project(J, n_neighbors=60, min_dist=0.15, random_state=42) -> (coords_2d, coords_3d)
- cosine_similarity(A, B) -> np.ndarray [A.shape[0], B.shape[0]]
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

# Optional deps for projection helpers
try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None  # type: ignore

try:
    import umap  # umap-learn
except Exception:  # pragma: no cover
    umap = None  # type: ignore

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
    """Load an OpenCLIP model + preprocess + tokenizer.

    Returns
    -------
    (model, preprocess, tokenizer, device)
    """
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
            f = model.encode_image(imgs_t)              # projected CLIP features
            f = f / f.norm(dim=-1, keepdim=True)        # L2 in torch
            feats.append(f.cpu().numpy())
    if feats:
        X = np.concatenate(feats, axis=0)
        return X.astype(np.float32, copy=False)
    # Fallback shape when no images; 512 is common for ViT-B/32. Safe default.
    return np.zeros((0, 512), np.float32)


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
            f = model.encode_text(toks)                 # projected CLIP features
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
    """Encode per-image captions and aggregate to one text vector per image.

    Parameters
    ----------
    all_captions : list[list[str]] where inner list are captions for image i
    aggregate   : "average" (mean across captions, then L2-normalize) or
                  "first" (only the first available caption)
    prompt_template : optional string like "A photo of {}."
    """
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
    Create joint embeddings by a convex combination of L2-normalized image/text vectors.
    joint = normalize( alpha * img + (1 - alpha) * txt )

    alpha in [0,1]:
      1.0 -> pure image; 0.0 -> pure text; 0.5 -> equal blend.
    """
    assert img.shape == txt.shape, "IMG and TXT must have same shape"
    I = l2_normalize(img)
    T = l2_normalize(txt)
    J = alpha * I + (1.0 - alpha) * T
    return l2_normalize(J)


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
        J50 = J  # proceed without PCA if sklearn not present
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
    row = np.arange(sims.shape[0])[:, None]
    s = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-s, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    s_sorted = np.take_along_axis(s, order, axis=1)
    return idx_sorted.astype(np.int64, copy=False), s_sorted.astype(np.float32, copy=False)


__all__ = [
    "detect_device",
    "load_open_clip",
    "compute_image_embeddings",
    "encode_texts",
    "compute_text_embeddings",
    "cosine_similarity",
    "topk_text_for_image",
    "l2_normalize",
    "joint_weighted_embeddings",
    "umap_project",
]
