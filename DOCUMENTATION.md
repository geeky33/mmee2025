# MMEE Embedding Viewer — Technical Documentation

> **Product**: Multimodal Embedding Explorer (MMEE) — *Embedding Viewer*
>
> **Audience**: ML engineers, data scientists, research students (intermediate Python/ML; basic Streamlit).
>
> **Purpose & Scope**: This document onboards users to the Streamlit-based Embedding Viewer, explains data requirements, core workflows (projection, alignment, outlier analysis, authenticity evaluation), and provides troubleshooting and best practices.

---

## 1) High‑Level Overview

The **MMEE Embedding Viewer** is a Streamlit application for exploring **image–text embeddings** (e.g., CLIP) and running **class‑wise outlier detection** with optional **per‑caption authenticity evaluation**.

**Key capabilities:**

* Load **dataset folders** under `data/` containing `images/`, `captions.json`, and `labels.json`.
* Compute **image and text embeddings** via `utils/clip_utils` (or built‑in fallbacks).
* Visualize in 2D via **PCA**, **t‑SNE**, or **UMAP** (with Procrustes alignment of text→image for joint plots).
* Explore **per‑image** vs **per‑caption** text points; render **joint plots** with linking lines.
* Run **classwise outlier detectors** (Isolation Forest, LOF, kNN‑quantile, DBSCAN), with **union**/**intersection** combiner.
* Optional **fusion scorer** (Logistic Regression with CV) on handcrafted features to improve detection.
* If `captions_with_noise.json` and `bad_caption_gt.json` exist, compute **authenticity metrics** with confusion matrix, PR/ROC, and ranked sanity tables.
* Deterministic **validation split** for thresholding and per‑class PR/ROC.
* CSV **exports** for TP/FP/FN/TN; interactive **Plotly** charts.

---

## 2) Prerequisites

* **Python** ≥ 3.9
* Suggested: **virtualenv/conda**
* **Streamlit** + common ML packages (see `requirements.txt`)
* Optional: **GPU** for faster embedding inference

**Data layout (per dataset):**

```
data/
  <dataset_name>/
    images/                  # image files in subfolders or flat
    captions.json            # mapping: relpath/name/stem -> list of captions
    labels.json              # mapping/list → class name/id per image
    captions_with_noise.json # optional (same schema as captions.json)
    bad_caption_gt.json      # optional (per-caption ground truth 0/1)
```

> `labels.json` accepted forms: list of objects or dict keyed by relative path/name/stem. Class can be embedded as `class_name` / `class_id`, or inferred from folder.

---

## 3) Install & Run

```bash
# clone and enter
git clone <your-repo-url>
cd <repo>

# create env
python -m venv .venv
source .venv/bin/activate  # fish: source .venv/bin/activate.fish

# install deps
pip install -r requirements.txt

# launch app
streamlit run app/main.py
```

> Ensure `data/` contains at least one dataset with the expected files; otherwise the app stops with a guided error.

---

## 4) Architecture & Modules

### 4.1 Repository Layout

```
├── .devcontainer/
│   └── devcontainer.json
├── .gitignore
├── DOCUMENTATION.md
├── app/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data_loaders/
│   │   ├── __init__.py
│   │   ├── caption_loader.py
│   │   ├── dataset_loader.py
│   │   └── label_loader.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── evaluation.py
│   │   ├── fusion_scorer.py
│   │   └── outlier_detectors.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── compute.py
│   │   └── projection.py
│   ├── main.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── display.py
│   │   └── sidebar.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── clip_utils.py
│   │   ├── helpers.py
│   │   └── validation.py
│   └── visualization/
│       ├── __init__.py
│       ├── plotly_helpers.py
│       └── trace_builders.py
├── backend/
│   ├── graph.py
│   └── main.py
├── cache/
│   ├── embeddings/
│   ├── projections/
│   └── thumbs/
├── requirements.txt
├── tools/
│   ├── add_bad_captions.py
│   ├── check_gt.py
│   └── eval_sweep.py
└── utils/
    ├── cache_utils.py
    ├── clip_utils.py
    ├── io_utils.py
    ├── prepare_coco.py
    ├── prepare_cub.py
    ├── prepare_groundcap.py
    ├── proj_utils.py
    └── siglip_utils.py
```

**Entry point**: `app/main.py` (Streamlit). Launch with:

```bash
streamlit run app/main.py
```

### 4.2 What Each Folder/File Does

#### Top level

* **.devcontainer/devcontainer.json** — VS Code Dev Containers config for reproducible setup (Python version, extensions, ports).
* **.gitignore** — Git exclusions (e.g., `cache/`, virtual envs, artifacts).
* **DOCUMENTATION.md** — This documentation; can be published or moved to `/docs`.
* **requirements.txt** — Python dependencies (Streamlit, scikit‑learn, plotly, umap‑learn, etc.).

#### `app/` — Streamlit front‑end and orchestration

* **app/main.py** — **Primary entry point/UI**. Coordinates dataset discovery, embedding computation, projections, Procrustes alignment, outlier detection, fusion scorer, evaluation, Plotly charts, CSV exports.
* **app/config/**

  * `settings.py` — Centralized UI defaults, constants (e.g., default perplexity, neighbors), paths, and feature toggles.
* **app/data_loaders/**

  * `dataset_loader.py` — Scans `data/` to list datasets, resolves root paths.
  * `caption_loader.py` — Loads `captions.json` or `captions_with_noise.json`, normalizes key formats (relpath/name/stem→list[str]).
  * `label_loader.py` — Loads `labels.json`, supports list/dict schemas, fills missing `class_id`, infers from folder when absent.
* **app/detection/**

  * `outlier_detectors.py` — Per‑class detectors (IsolationForest, LOF, kNN‑quantile, DBSCAN) with score normalization.
  * `fusion_scorer.py` — Feature engineering (cosine, residual, oddness, caption length, detector score) + LogisticRegression CV thresholding.
  * `evaluation.py` — Metrics (Precision/Recall/F1), confusion matrix, per‑class PR/ROC on frozen validation split, CSV bucket writers.
* **app/embedding/**

  * `compute.py` — Bridges to `utils/clip_utils`/`utils/siglip_utils`; computes image/text embeddings with batching and caching.
  * `projection.py` — PCA/t‑SNE/UMAP wrappers, 2D/3D handling, caching, plus **orthogonal Procrustes** alignment helpers.
* **app/ui/**

  * `sidebar.py` — All sidebar controls (dataset, model/weights, projection params, caption mode, detectors, fusion, evaluation toggles).
  * `display.py` — KPIs, tables, download buttons, info/warning/status blocks.
* **app/utils/**

  * `clip_utils.py` — App‑scoped helpers (thin wrappers) to call global `utils/clip_utils.py` and provide **safe fallbacks** if missing.
  * `helpers.py` — Hash/signature utilities, normalization, cosine helpers, deterministic split mask.
  * `validation.py` — Schema checks, defensive guards, user‑facing error messages.
* **app/visualization/**

  * `plotly_helpers.py` — Common plotting styles, color maps, axes formatting.
  * `trace_builders.py` — Builders for image/text scatters, joint link lines, outlier/PR/ROC traces.

#### `backend/` — Optional service layer (non‑UI)

* **backend/main.py** — API/bootstrap for running embedding / projection / detection as services (e.g., FastAPI). Useful for headless jobs or remote execution.
* **backend/graph.py** — Graph utilities (e.g., building joint graphs, neighbor indices, or serving graph data to the UI).

> The Streamlit app does **not** require the backend to run locally, but these modules support service/automation use cases.

#### `cache/`

* **embeddings/** — Cached `.npy` or parquet embeddings keyed by signature.
* **projections/** — Cached 2D/3D projection arrays.
* **thumbs/** — (Optional) generated thumbnails for faster UI previews.

#### `tools/` — Utilities & scripts

* **add_bad_captions.py** — Injects noise into captions to create `captions_with_noise.json` and paired `bad_caption_gt.json`.
* **check_gt.py** — Validates ground‑truth alignment, counts positives, spot‑checks schema.
* **eval_sweep.py** — Batch runs across detectors/params to collect metrics; useful for ablations.

#### `utils/` — Global library (model/data utilities)

* **cache_utils.py** — Signature hashing, path‑safe cache keys, read/write helpers.
* **clip_utils.py** — Core CLIP image/text embedding functions (called by `app/embedding/compute.py`), optional projection wrappers.
* **io_utils.py** — I/O helpers: safe JSON/CSV read/write, recursive file discovery, image filters.
* **prepare_coco.py / prepare_cub.py / prepare_groundcap.py** — Dataset converters to the expected `images/`, `captions.json`, `labels.json` format.
* **proj_utils.py** — Projection utilities (PCA/TSNE/UMAP adapters), conditionally used where app‑local fallbacks aren’t desired.
* **siglip_utils.py** — Alternative encoder support (e.g., SigLIP) for experimentation.

---

## 5) Features — Detailed

### 5.1 Dataset Selection & Subsetting

* Select dataset from sidebar (`data/<dataset>` auto‑discovered).
* **Class filters**: choose classes; set **samples per class** (0 = all).
* Captions source: **Original** or **With noise** (if `captions_with_noise.json` exists). If noise GT present, authenticity paths are enabled.

### 5.2 Embedding Computation

* Buttons: **Compute IMAGE embeddings** / **Compute TEXT embeddings**
* Uses `utils/clip_utils.compute_image_embeddings()` and `.compute_text_embeddings()` with model/weights selection:

  * Models: `ViT-B-32`, `ViT-L-14`, `ViT-L-14-336`
  * Weights: `openai`, `laion2b_s34b_b79k`, `laion2b_s32b_b82k`
* **Batch size** configurable for images.
* **Caption granularity**:

  * *Aggregate per image*: text aggregation `average` / `first`
  * *Per‑caption points*: each caption becomes a text sample, with `caps_limit` per image

### 5.3 Projection & Joint Alignment

* Methods: **PCA**, **t‑SNE** (perplexity), **UMAP** (n_neighbors, min_dist)
* **project_images_and_text(...)** does:

  * L2 normalize image/text
  * Center & align text to image embedding space via **orthogonal Procrustes**
  * Stack and project to 2D with chosen method
* **Joint blend (α)**: create a single 512‑D joint vector per entity as `α·image + (1−α)·text`; separately project this for a unified view.

### 5.4 Cosine Similarity & Sanity Checks

* Compute **row‑wise cosine** between paired image/text embeddings
* Quick UI: “Top‑5 closest texts for this image (cosine)” to inspect retrieval sanity

### 5.5 Outlier Detection (Per‑Class)

* Choose **detection space**:

  * Raw joint (512‑D joint blend)
  * 2D Image or 2D Text projection space
* Methods (applied **per class**):

  * **Isolation Forest**
  * **kNN Distance (Quantile)** — outlier if k‑th neighbor dist ≥ quantile
  * **LOF** (Local Outlier Factor)
  * **DBSCAN** — noise points as outliers; optional auto‑eps via k‑NN
* **Combiner** across classes:

  * **Union** (recall‑oriented): flagged if any class model flags it
  * **Intersection** (precision‑oriented): flagged only if all class models flag
* **Tunable parameters**: contamination, k, quantile, n_neighbors, eps/min_samples

### 5.6 Fusion Scorer (Optional)

* Enable **feature‑fusion scorer**: Logistic Regression on handcrafted features
* **Cross‑validation** (StratifiedKFold) on **validation split** to pick threshold at target recall
* Outputs **fused score** and **fused prediction** per caption (if per‑caption mode)

### 5.7 Authenticity Evaluation (Per‑Caption mode)

* Requires `bad_caption_gt.json` alongside `captions_with_noise.json`
* Metrics: **Precision, Recall, F1**, **Confusion Matrix**, **PR/ROC per class (validation)**
* **Ranked tables** for TP/FP/FN (top‑N by score); **CSV exports** for all buckets
* KPIs shown at top: total samples, GT positives, outliers flagged

### 5.8 Visualization

* **Side‑by‑side** scatter plots: Image vs Text embeddings, with class color and optional outlier overlays
* **Joint plot** with **connection lines** (image↔caption) and overlays for authenticity buckets or outliers
* **Single‑view Joint‑blend** (projected from α‑blend 512‑D)
* All charts are **interactive Plotly** (pan/zoom/hover)

---

## 6) Typical Workflow (Step‑by‑Step)

1. **Choose dataset** in the sidebar; verify paths display under *Paths*.
2. **Select captions source**: *Original* or *With noise* (if available).
3. **Set model & weights**, **batch size** for images.
4. **Pick projection method** (PCA/t‑SNE/UMAP) and tune its hyper‑parameters.
5. Choose **Caption granularity**:

   * *Aggregate per image*: pick `average` or `first`
   * *Per‑caption*: optionally cap captions per image
6. **Compute embeddings** (IMAGE then TEXT). The app enforces recompute on setting changes.
7. Optionally **subset classes** and **sample per class** to focus exploration.
8. Configure **Joint & Outliers**:

   * Blend `α`, detection space, and outlier method + params
   * Choose combiner (Union/Intersection)
9. **Enable Fusion scorer** (optional) and set CV folds, target recall, validation fraction.
10. **Run detection**, view KPIs, plots, **Evaluation** and **Sanity tables**; export CSVs.
11. Use **Cosine sanity** tool to manually inspect nearest captions for an image.

---

## 7) Caching, Signatures & Invalidation

* A **signature** is computed from dataset name, model, weights, aggregation tag, and image paths.
* Embeddings cached by signature via `@st.cache_data`. When dataset/filters change, cache is invalidated.
* Projection results are also cached by method + hyper‑parameters → fast iteration.

---

## 8) Best Practices & Tips

* **Normalize** embeddings before any projection (already enforced).
* Prefer **PCA→UMAP** for speed + stability; lower UMAP `min_dist` to separate tight clusters.
* For large datasets, use **class sampling** to keep plots responsive.
* Start outlier detection in **Raw joint (512‑D)** for better signal; then visualize flagged points in 2D.
* When using **t‑SNE**, tune **perplexity** (commonly 20–40) and fix `random_state` for reproducibility.
* For **DBSCAN auto‑eps**, ensure each class has ≥ ~20 samples for a sensible k‑NN estimate.

---

## 9) Troubleshooting / FAQ

**The app says no datasets found.**
Ensure a path like `data/<dataset>/images/` with `captions.json` and `labels.json` exists.

**Embeddings shape mismatch / asks to recompute.**
You changed dataset, filters, or caption granularity. Recompute IMAGE and TEXT.

**UMAP not installed.**
Install `umap-learn` or rely on PCA/t‑SNE fallbacks.

**Plots are slow / browser lags.**
Reduce samples per class, or switch to PCA. Use *Show only clean* to hide outliers while panning.

**No PR/ROC or metrics shown.**
Per‑caption authenticity requires `bad_caption_gt.json` and *Captions source = With noise*.

**DBSCAN marks everything as noise.**
Decrease `eps` or increase `min_samples`; ensure class has enough points.

**Fusion scorer unstable.**
Increase validation fraction, or reduce feature set by disabling fusion.

---

## 10) Limitations & Future Work

* **High‑D to 2D loss**: projections are approximate; always corroborate with cosine and metrics.
* **Classwise detectors** assume sufficient per‑class samples for stable thresholds.
* **Fusion features** are handcrafted; consider **learned encoders** or **calibrated scoring**.
* Extend to **TriMap/PaCMAP/PHATE** for alternative projections.
* Add **3D WebGL** scatter and **image thumbnails** on hover.
* Support **batch export** of flagged samples and **active‑learning loops**.

---

## 11) API Surfaces (from `clip_utils.py` expected)

* `compute_image_embeddings(paths: List[str], model_name: str, pretrained: str, batch_size: int) -> np.ndarray`
* `compute_text_embeddings(captions: List[List[str]], model_name: str, pretrained: str, aggregate: str) -> np.ndarray`
* Optional:

  * `pca_project(X, n_components=2) -> np.ndarray`
  * `tsne_project(X, n_components=2, perplexity: int, random_state: int) -> np.ndarray`
  * `umap_project(X, n_neighbors: int, min_dist: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]`
  * `joint_weighted_embeddings(img, txt, alpha) -> np.ndarray`

> If any are missing, the app provides **robust fallbacks**.

---

## 12) Security & Privacy Notes

* No network calls are made by the app itself; data stays local.
* Ensure your datasets do not contain sensitive information or PII.

---

## 13) Glossary

* **Procrustes Alignment**: Finds an orthogonal transform that best aligns two point sets (text→image space here).
* **Union vs Intersection**: Combining per‑class outlier flags to bias toward recall (union) or precision (intersection).
* **Authenticity**: Here, detecting **bad/perturbed captions** vs ground truth in `bad_caption_gt.json`.

---

## 14) Changelog (for this Viewer)

* Added **per‑caption mode** with GT alignment and **frozen validation split**.
* Introduced **fusion scorer** (LR + CV) with target recall calibration.
* Unified **CSV exports** for TP/FP/FN/TN and **ranked sanity tables**.
* Added robust **fallbacks** for projection and joint blending.

---

## 15) License & Attribution

* Cite CLIP/OpenCLIP when applicable.
* See repository `LICENSE` for terms.

---

### Quick Start (TL;DR)

1. Put your dataset under `data/<name>/` with `images/`, `captions.json`, `labels.json`.
2. `pip install -r requirements.txt`; run `streamlit run app/main.py`.
3. Choose dataset, compute **IMAGE** and **TEXT** embeddings.
4. Pick projection; enable **Run detection**; (optional) enable **Fusion**.
5. Inspect plots, metrics, and export CSVs.
