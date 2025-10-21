#  MMEE — Multimodal Embedding Explorer

The **Multimodal Embedding Explorer (MMEE)** is an interactive visualization and analysis tool built with **Streamlit** and **FastAPI** for exploring **joint image-text embeddings** derived from CLIP and similar models. It helps researchers, data scientists, and developers analyze alignment, detect outliers, and evaluate dataset authenticity in multimodal datasets.

---

##  Key Features

* **Image & Text Embedding Projection:**
  Visualize embeddings using **PCA**, **t-SNE**, and **UMAP** for both image and text data.

* **Joint Alignment & Similarity:**
  Align embeddings via **Orthogonal Procrustes** and analyze cosine similarity between image-text pairs.

* **Outlier Detection (Per-Class):**
  Detect anomalies in embeddings using methods like **Isolation Forest**, **LOF**, **kNN Quantile**, and **DBSCAN**.

* **Fusion Scorer:**
  Combine multiple features (cosine similarity, residuals, caption length, etc.) into a single prediction using **Logistic Regression with CV**.

* **Evaluation Dashboard:**
  Interactive confusion matrices, ROC/PR curves, and ranked tables comparing predictions vs ground truth.

* **Joint 2D Visualization:**
  Integrated image-text plots with connection lines showing embedding alignment.

---

## Project Overview

* **Goal:** Provide a fast, GPU-optimized, and intuitive interface for visualizing and diagnosing multimodal embeddings.
* **Use Cases:**

  * Inspect CLIP embedding quality
  * Detect mislabeled or noisy captions
  * Explore cross-modal alignment (image ↔ text)
  * Compare outlier detection strategies

---

## 🗂️ Repository Structure

```
├── .devcontainer/                # VSCode remote dev setup
│   └── devcontainer.json
├── .gitignore                    # Standard Git ignore file
├── DOCUMENTATION.md              # Full technical documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── app/                          # Streamlit app source
│   ├── main.py                   # Entry point for the Streamlit UI
│   ├── config/                   # Configuration management
│   │   ├── settings.py           # Environment and path configs
│   ├── data_loaders/             # Input data parsing
│   │   ├── caption_loader.py     # Handles caption reading
│   │   ├── dataset_loader.py     # Discovers datasets under /data
│   │   └── label_loader.py       # Reads and processes label JSONs
│   ├── detection/                # Outlier detection logic
│   │   ├── evaluation.py         # Precision/recall, confusion metrics
│   │   ├── fusion_scorer.py      # Logistic regression fusion model
│   │   └── outlier_detectors.py  # IF, LOF, kNN, DBSCAN implementations
│   ├── embedding/                # Embedding computation & projections
│   │   ├── compute.py            # Image/Text embedding functions
│   │   └── projection.py         # PCA, t-SNE, UMAP utilities
│   ├── ui/                       # UI rendering components
│   │   ├── display.py            # Visualization layout
│   │   └── sidebar.py            # Sidebar parameter controls
│   ├── utils/                    # Shared utilities
│   │   ├── clip_utils.py         # Core CLIP embedding utilities
│   │   ├── helpers.py            # Misc. helper functions
│   │   └── validation.py         # Input validation & sanity checks
│   └── visualization/            # Plotly-based charts
│       ├── plotly_helpers.py     # Plotly configs & styling
│       └── trace_builders.py     # Custom traces (authenticity/outliers)
|
├── cache/                        # Cached embeddings & projections
│   ├── embeddings/
│   ├── projections/
│   └── thumbs/
├── requirements.txt              # Python dependencies
├── tools/                        # Utility scripts
│   ├── add_bad_captions.py       # Injects noisy captions for testing
│   ├── check_gt.py               # Ground-truth checker
│   └── eval_sweep.py             # Evaluation sweeps over models
└── utils/                        # Shared cross-module utilities
    ├── prepare_coco.py           # Prepare COCO dataset
    ├── prepare_cub.py            # Prepare CUB dataset
    ├── prepare_groundcap.py      # Prepare GroundCap dataset
```

### ▶️ Entry Point

The application starts from:

```
streamlit run app/main.py
```

This launches the full interactive Streamlit UI.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/mmee.git
   cd mmee
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**

   ```bash
   streamlit run app/main.py
   ```

5. **Access the app:**
   Open the displayed local URL (typically `http://localhost:8501`).

---

##  Usage Overview

* **Dataset Selection:** Choose any dataset from the sidebar (expects `images/`, `captions.json`, and `labels.json`).
* **Embedding Computation:** Click buttons to compute image/text embeddings.
* **Projection Settings:** Pick PCA, t-SNE, or UMAP; tune parameters.
* **Outlier Detection:** Run Isolation Forest, LOF, or kNN to find anomalies.
* **Evaluation:** View confusion matrix, precision-recall stats, and sanity tables.
* **Visualization:** Inspect alignment in Image/Text/Joint embedding plots.

---

##  Technologies Used

* **Frontend:** Streamlit + Plotly
* **ML Libraries:** scikit-learn, UMAP-learn, NumPy, Pandas
* **Model:** CLIP (OpenAI / LAION variants)

---

## 📁 Data Format

Each dataset folder under `data/` should contain:

```
images/               # All image files
captions.json         # {"img_1.jpg": ["caption1", "caption2", ...], ...}
labels.json           # {"img_1.jpg": {"class_name": "cat", "class_id": 1}, ...}
```

---

## 💡 Tips

* Use **UMAP** for preserving local structure and neighborhood relations.
* Use **Isolation Forest** for general-purpose outlier detection.
* For noisy datasets, generate test files via `tools/add_bad_captions.py`.


---

## 🧑‍💻 Contributors

* **Aarya Pandey** — Project Developer
* **Mentors:** Laurens Hogeweg, Rajesh Gangireddy & Samet Akcay — Intel OpenVINO Toolkit (GSoC 2025)

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

**MMEE — Visualize, Compare, and Understand Multimodal Embeddings.**
