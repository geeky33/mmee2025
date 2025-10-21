# Contributing to MMEE — Embedding Viewer

Thanks for your interest in contributing! This guide explains how to set up your environment, propose changes, and submit pull requests with confidence.

> **Entry point:** The Streamlit app starts at `app/main.py`.

---

## Table of Contents

* [Code of Conduct](#code-of-conduct)
* [Project Setup](#project-setup)
* [Development Workflow](#development-workflow)
* [Branching & Commits](#branching--commits)
* [Style, Linting & Types](#style-linting--types)
* [Testing & QA](#testing--qa)
* [Running the App](#running-the-app)
* [Adding New Features](#adding-new-features)

  * [New Outlier Detector](#new-outlier-detector)
  * [New Projection Method](#new-projection-method)
  * [New Data Loader](#new-data-loader)
* [Docs & Screenshots](#docs--screenshots)
* [Datasets, Privacy & Large Files](#datasets-privacy--large-files)
* [Issue Labels](#issue-labels)
* [PR Checklist](#pr-checklist)

---

## Code of Conduct

This project follows a standard, respectful Code of Conduct. Be kind, assume positive intent, and keep reviews constructive. Harassment or discriminatory behavior will not be tolerated.

---

## Project Setup

### 1) Clone & Environment

```bash
git clone <your-repo-url>
cd <repo>

python -m venv .venv
# bash/zsh
source .venv/bin/activate
# fish
# source .venv/bin/activate.fish

pip install -r requirements.txt
```

### 2) Optional: Dev Container

If you use VS Code, open the repo in a **Dev Container** (`.devcontainer/devcontainer.json`) for a reproducible environment.

### 3) Data Layout

Place datasets in `data/<dataset>/`:

```
images/                  # required
captions.json            # required
labels.json              # required
captions_with_noise.json # optional
bad_caption_gt.json      # optional
```

---

## Development Workflow

1. **Open an issue** (bug/feature) and discuss scope.
2. **Fork** the repo (or create a feature branch in the main repo if you have access).
3. **Create a branch**: `feat/<short-name>` or `fix/<short-name>`.
4. **Develop** with small, focused commits.
5. **Run linters/tests** locally.
6. **Open a PR** with a clear description, screenshots/GIFs for UI changes, and “How I tested this.”
7. Address review comments and keep a friendly, factual tone.

---

## Branching & Commits

* **Default branch**: `main`
* **Feature branches**: `feat/<topic>`; **bugfix**: `fix/<topic>`; **docs**: `docs/<topic>`
* **Commit style** (conventional):

  * `feat: add per-caption fusion scorer`
  * `fix: correct Procrustes alignment when per_caption=True`
  * `docs: expand troubleshooting for DBSCAN`
  * `refactor: split projection helpers`

Keep commits atomic; prefer present tense and clear intent.

---

## Style, Linting & Types

* **Python version**: 3.9+
* **Formatting**: `black`
* **Imports**: `isort`
* **Linting**: `ruff`
* **Typing**: add type hints where practical; run `mypy` if configured.
* **Docstrings**: Google or NumPy style (be consistent within a module).

> If you’re not sure, run a quick formatter pass before committing.

Example pre-commit setup (optional):

```bash
pip install pre-commit
pre-commit install
```

---

## Testing & QA

* Unit tests live under `tests/` (create if missing). Prefer pytest.
* For UI logic, factor non-UI code into `app/utils/`, `app/embedding/`, etc., and test those units.
* Add **sample datasets** (tiny) under `data/fixtures/` for deterministic tests.
* Manual QA steps should be listed in your PR.

---

## Running the App

```bash
streamlit run app/main.py
```

Key places during dev:

* `app/ui/sidebar.py` — tweak controls and defaults.
* `app/embedding/compute.py` — add encoders, caching, batching.
* `app/embedding/projection.py` — add/reuse projection helpers.
* `app/detection/` — detectors, fusion scorer, evaluation.

---

## Adding New Features

### New Outlier Detector

1. Implement the detector in `app/detection/outlier_detectors.py` with a function signature similar to existing ones:

   ```py
   def mydetector_single_class(X: np.ndarray, **params) -> tuple[np.ndarray, np.ndarray]:
       """Return (labels, scores) for a single class subset.
       labels ∈ {0,1}, scores: float (higher = more anomalous)."""
   ```
2. Register it in the sidebar options (`app/ui/sidebar.py`).
3. Integrate per-class loop via `run_classwise_detector(...)` or add a new branch mirroring existing methods.
4. Update docs in `DOCUMENTATION.md` if the method has unique parameters.

### New Projection Method

1. Add a wrapper in `app/embedding/projection.py` or `utils/proj_utils.py`.
2. Accept method hyper‑parameters in the sidebar, then call via `project_2d(...)`.
3. Ensure 2D ndarray output (N×2). For optional 3D views, keep (N×3) behind a flag.

### New Data Loader

1. Implement in `app/data_loaders/` (e.g., `foobar_loader.py`).
2. Normalize to the canonical in‑memory schema expected by `app/main.py`:

   * images list, captions map, labels DataFrame with `class_name`, `class_id`, `image_relpath`.
3. Hook into discovery (`dataset_loader.py`) or call explicitly.

---

## Docs & Screenshots

* Keep **`DOCUMENTATION.md`** up‑to‑date when adding user‑visible features.
* Store demo screenshots/GIFs under `docs/assets/` (git‑friendly sizes); reference them relatively in docs.

---

## Datasets, Privacy & Large Files

* Do **not** commit private datasets or PII. Prefer synthetic or public samples.
* Large artifacts should use Git LFS or be ignored via `.gitignore` (e.g., `cache/`, `*.npy`).
* Respect dataset licenses when sharing.

---

## Issue Labels

* `bug`, `feature`, `documentation`, `good first issue`, `help wanted`, `performance`, `infra`, `ui/ux`.

Use `good first issue` only for self‑contained, well‑scoped tasks with guidance.

---

## PR Checklist

* [ ] Linked issue and scope explained
* [ ] Small, focused commits; conventional messages
* [ ] Linted & formatted (black/isort/ruff)
* [ ] Types/docstrings added where useful
* [ ] Unit tests updated/added (if applicable)
* [ ] Manual QA steps described; screenshots/GIFs for UI
* [ ] No secrets/PII; large files excluded

Welcome aboard — and thanks for helping improve MMEE! 🙌
