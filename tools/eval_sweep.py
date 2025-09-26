#!/usr/bin/env python3
import argparse, json, itertools, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

# project import path
import sys
PROJ_DIR = Path(__file__).resolve().parents[1]
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from utils.clip_utils import (
    compute_image_embeddings, compute_text_embeddings, cosine_similarity,
    # optional helpers if present
    knn_quantile_detect, lof_detect, dbscan_detect, mahalanobis_detect,
    pca_recon_error_detect, ocsvm_detect
)

# simple local fallbacks for IF if needed
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def l2n(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (X / n).astype(np.float32, copy=False)

def joint_weighted(img, txt, alpha=0.5):
    I = l2n(img); T = l2n(txt)
    J = alpha * I + (1.0 - alpha) * T
    J /= np.maximum(np.linalg.norm(J, axis=1, keepdims=True), 1e-12)
    return J.astype(np.float32, copy=False)

def read_captions(path: Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v if str(x).strip()]
        elif v is None:
            out[str(k)] = []
        else:
            out[str(k)] = [str(v)]
    return out

def scan_images(images_dir: Path):
    exts = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])

def eval_one(y_true, y_pred):
    P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return dict(precision=P, recall=R, f1=F1, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def detect(method, X, params):
    m = method
    if m == "Isolation Forest":
        if IsolationForest is None:
            raise RuntimeError("sklearn IsolationForest not available")
        model = IsolationForest(
            n_estimators=int(params.get("n_estimators", 300)),
            contamination=float(params.get("contamination", 0.03)),
            max_samples=params.get("max_samples", "auto"),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )
        model.fit(X)
        labels = (model.predict(X) == -1).astype(np.int8)
        scores = (-model.score_samples(X)).astype(np.float32)
        return labels, scores
    if m == "kNN Distance (Quantile)":
        return knn_quantile_detect(X, k=int(params["k"]), quantile=float(params["quantile"]))
    if m == "LOF":
        return lof_detect(X, n_neighbors=int(params["n_neighbors"]), contamination=float(params["contamination"]))
    if m == "DBSCAN":
        return dbscan_detect(X, eps=float(params["eps"]), min_samples=int(params["min_samples"]))
    if m == "Mahalanobis":
        return mahalanobis_detect(X, contamination=float(params["contamination"]), robust=bool(params["robust"]))
    if m == "PCA Recon":
        return pca_recon_error_detect(X, n_components=params["n_components"], contamination=float(params["contamination"]))
    if m == "OCSVM":
        return ocsvm_detect(X, nu=float(params["nu"]), kernel=str(params["kernel"]), gamma=str(params["gamma"]))
    raise ValueError(f"Unknown method {method}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset folder (e.g., data/cub-200)")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--weights", default="openai")
    ap.add_argument("--alpha", type=float, default=0.5, help="Joint blend alpha (image↔text)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--space", choices=["raw_joint", "text_only", "image_only"], default="raw_joint")
    ap.add_argument("--out", default="sweep_results.csv")
    args = ap.parse_args()

    np.random.seed(args.seed)

    droot = Path(args.dataset)
    images_dir = droot / "images"
    captions_noise = droot / "captions_with_noise.json"
    gt_path = droot / "bad_caption_gt.json"

    if not captions_noise.exists() or not gt_path.exists():
        raise SystemExit("Need captions_with_noise.json and bad_caption_gt.json in the dataset folder.")

    # Load data
    paths = [str(p) for p in scan_images(images_dir)]
    caps_map = read_captions(captions_noise)
    # Build per-caption list aligned with image order
    img_rel = []
    cap_texts = []
    cap_img_idx = []
    for i, p in enumerate(paths):
        key_opts = [str(Path(p).relative_to(images_dir)), Path(p).name]
        clist = None
        for k in key_opts:
            if k in caps_map:
                clist = caps_map[k]
                break
        if clist is None:
            continue
        for c in clist:
            img_rel.append(key_opts[-1])
            cap_texts.append(c)
            cap_img_idx.append(i)

    if len(cap_texts) == 0:
        raise SystemExit("No captions matched images.")

    # Ground truth flags aligned by running index per image
    with open(gt_path, "r", encoding="utf-8") as f:
        GT = json.load(f)
    counter = defaultdict(int)
    y_true = []
    for rel in img_rel:
        k = counter[rel]
        seq = GT.get(rel) or GT.get(Path(rel).name)
        flag = int(seq[k]) if (seq and k < len(seq)) else 0
        y_true.append(flag)
        counter[rel] += 1
    y_true = np.asarray(y_true, dtype=np.int32)

    # Embeddings
    IMG = compute_image_embeddings(paths, model_name=args.model, pretrained=args.weights, batch_size=args.batch)
    TXT = compute_text_embeddings([[t] for t in cap_texts], model_name=args.model, pretrained=args.weights, aggregate="first")
    IMG_rep = IMG[np.asarray(cap_img_idx, dtype=int), :]
    if args.space == "raw_joint":
        X = joint_weighted(IMG_rep, TXT, alpha=args.alpha)
    elif args.space == "text_only":
        X = TXT
    else:
        X = IMG_rep

    # Search space
    grids = {
        "Isolation Forest": dict(
            contamination=[0.01, 0.02, 0.03, 0.05, 0.1],
            n_estimators=[200, 300],
        ),
        "kNN Distance (Quantile)": dict(
            k=[5, 10, 20, 30],
            quantile=[0.95, 0.975, 0.99, 0.995]
        ),
        "LOF": dict(
            n_neighbors=[10, 20, 40, 60],
            contamination=[0.02, 0.03, 0.05]
        ),
        "DBSCAN": dict(
            eps=[0.5, 0.8, 1.0, 1.5],
            min_samples=[5, 10, 20]
        ),
        "Mahalanobis": dict(
            contamination=[0.02, 0.03, 0.05],
            robust=[True, False]
        ),
        "PCA Recon": dict(
            n_components=[0.85, 0.9, 0.95],
            contamination=[0.02, 0.03, 0.05]
        ),
        "OCSVM": dict(
            nu=[0.01, 0.02, 0.05],
            kernel=["rbf"],
            gamma=["scale", "auto"]
        ),
    }

    rows = []
    for method, grid in grids.items():
        keys = list(grid.keys())
        for values in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, values))
            try:
                y_pred, _ = detect(method, X, params)
            except Exception as e:
                rows.append(dict(method=method, params=params, error=str(e), precision=0, recall=0, f1=0))
                continue
            y_pred = np.asarray(y_pred, dtype=np.int32)
            m = eval_one(y_true, y_pred)
            rows.append(dict(method=method, params=params, **m))

    df = pd.DataFrame(rows)
    # Best per method
    best = (df.sort_values(["method", "f1"], ascending=[True, False])
              .groupby("method", as_index=False).first())
    df.to_csv(args.out, index=False)
    best.to_csv(Path(args.out).with_name("sweep_leaderboard.csv"), index=False)

    # Pretty print
    print("\n=== Leaderboard (best F1 per method) ===")
    print(best[["method", "precision", "recall", "f1", "params"]].to_string(index=False))
    print(f"\nWrote:\n - {args.out}\n - {Path(args.out).with_name('sweep_leaderboard.csv')}\n")

if __name__ == "__main__":
    main()
