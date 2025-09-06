#!/usr/bin/env python3
import argparse, json, os, sys, hashlib, io
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image

DATASET_ID = "daniel3303/GroundCap"  # public HF dataset


# ----------------------------
# Optional deps (import lazily)
# ----------------------------
def _try_import_datasets():
    try:
        from datasets import load_dataset
        return load_dataset
    except Exception:
        return None

def _try_import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except Exception:
        return None

def _have_spacy():
    try:
        import spacy  # noqa
        return True
    except Exception:
        return False


# ----------------------------
# Image helpers
# ----------------------------
def save_image(pil_img: Image.Image, out_path: Path, jpeg_quality=92):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path, format="JPEG", quality=jpeg_quality)

def pil_from_any(x) -> Image.Image:
    """
    Convert a datasets.Image feature item (PIL/ndarray/bytes/path/dict) to PIL.Image.
    """
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict):
        if "bytes" in x and x["bytes"] is not None:
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
        if "path" in x and x["path"]:
            return Image.open(x["path"]).convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    try:
        import numpy as np  # noqa
        if isinstance(x, np.ndarray):  # type: ignore[name-defined]
            return Image.fromarray(x)
    except Exception:
        pass
    # final fallback: treat as path-like
    return Image.open(str(x)).convert("RGB")


# ----------------------------
# Caption & label helpers
# ----------------------------
def normalize_caption(c: Any) -> Optional[str]:
    if c is None:
        return None
    s = str(c).strip()
    return s if s else None

def labels_from_fields(row: Dict[str, Any], candidate_fields: List[str]) -> List[str]:
    merged: List[str] = []
    for f in candidate_fields:
        if f not in row:
            continue
        val = row[f]
        if val is None:
            continue
        if isinstance(val, list):
            merged.extend([str(x).strip() for x in val if str(x).strip()])
        elif isinstance(val, dict):
            # Flatten keys/values that look like class strings
            for k, v in val.items():
                if isinstance(v, (str, int, float)) and str(v).strip():
                    merged.append(str(v).strip())
                if isinstance(k, (str, int)) and str(k).strip():
                    if isinstance(k, str):
                        merged.append(k.strip())
        else:
            s = str(val).strip()
            if s:
                merged.append(s)
    # de-dup while keeping order
    seen, uniq = set(), []
    for x in merged:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def labels_from_caption_regex(text: str, max_k: int = 2) -> List[str]:
    import re
    STOP = set("""
        a an the and or but if then while of to for on in at from by with without over under between across
        this that these those is are was were be been being it its they them he she you i we us our your
    """.split())
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-']+", text)]
    words = [w for w in words if w not in STOP and len(w) > 2]
    return words[:max_k]

def labels_from_caption_spacy(text: str, max_k: int = 2) -> List[str]:
    import spacy
    # small model is enough; we only need noun chunks
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
        if "parser" not in nlp.pipe_names:
            nlp.enable_pipe("parser")
    except Exception:
        # fallback to default english if model isn't downloaded
        try:
            nlp = spacy.blank("en")
            nlp.add_pipe("parser")
        except Exception:
            # ultimate fallback to regex
            return labels_from_caption_regex(text, max_k=max_k)

    doc = nlp(text)
    chunks = [nc.text.lower().strip() for nc in getattr(doc, "noun_chunks", [])]
    if not chunks:
        # fallback to token lemmas if parser produced no chunks
        toks = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
        chunks = toks
    out, seen = [], set()
    for s in chunks:
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= max_k:
            break
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output dir (e.g., data/groundcap)")
    ap.add_argument("--limit", type=int, default=10000, help="Max number of items to export")
    ap.add_argument("--split", type=str, default="train", help="Dataset split (train/validation/test if available)")
    ap.add_argument("--start", type=int, default=0, help="Skip first N items (paging)")
    ap.add_argument("--hf_token", type=str, default=None, help="HF token (or login via huggingface-cli)")
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode (doesn't download full shards)")
    ap.add_argument("--max-captions", type=int, default=10, help="Keep up to N captions per image (to limit file size)")
    ap.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality for exported images")

    ap.add_argument(
        "--label-source",
        choices=["auto", "fields", "caption", "caption_spacy", "none"],
        default="auto",
        help=(
            "How to produce labels for coloring/filtering:\n"
            " - fields: use dataset fields like objects/categories if present\n"
            " - caption: keywords from first caption (regex)\n"
            " - caption_spacy: noun phrases from first caption (requires spaCy)\n"
            " - none: leave []\n"
            " - auto (default): fields if present, else caption_spacy if spaCy available, else caption"
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    caps_out = out_dir / "captions.json"
    labs_out = out_dir / "labels.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset (prefer streaming if requested)
    ds = None
    load_dataset = _try_import_datasets()
    if load_dataset is not None:
        try:
            if args.streaming:
                ds = load_dataset(DATASET_ID, split=args.split, use_auth_token=args.hf_token, streaming=True)
            else:
                ds = load_dataset(DATASET_ID, split=args.split, use_auth_token=args.hf_token)
        except Exception as e:
            print(f"[warn] load_dataset failed: {e}", file=sys.stderr)

    # 2) Fallback: snapshot → local load (no streaming)
    if ds is None:
        snapshot_download = _try_import_snapshot_download()
        if snapshot_download is None:
            print("[error] huggingface_hub not available and datasets.load_dataset failed; cannot proceed.", file=sys.stderr)
            sys.exit(1)
        try:
            local_repo = snapshot_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                use_auth_token=args.hf_token,
                local_dir=out_dir / "_hf_snapshot",
                local_dir_use_symlinks=False,
            )
            if load_dataset is None:
                print("[error] datasets lib not available to read the snapshot. Install `pip install datasets`.", file=sys.stderr)
                sys.exit(1)
            ds = load_dataset(str(local_repo), split=args.split)
        except Exception as e:
            print(f"[error] snapshot_download failed: {e}", file=sys.stderr)
            print("Tips:\n"
                  " - Ensure internet access or set HF mirror: export HF_ENDPOINT=https://hf-mirror.com\n"
                  " - Login: `huggingface-cli login`\n"
                  " - Or download on a machine with internet and copy the snapshot folder.", file=sys.stderr)
            sys.exit(1)

    # Figure out fields
    try:
        columns = set(ds.column_names)  # standard dataset
    except Exception:
        # streaming iterable dataset
        try:
            first = next(iter(ds))
        except StopIteration:
            print("[error] dataset is empty.", file=sys.stderr)
            sys.exit(1)
        columns = set(first.keys())

    # Heuristic fields
    possible_caption_fields = [c for c in ["caption", "text", "instruction", "output"] if c in columns]
    caption_field = possible_caption_fields[0] if possible_caption_fields else None
    image_field = "image" if "image" in columns else None
    label_candidates = [c for c in ["objects", "object_names", "labels", "categories", "actions"] if c in columns]

    if image_field is None or caption_field is None:
        print(f"[error] Expected fields not found. Columns present: {sorted(columns)}", file=sys.stderr)
        sys.exit(1)

    # Decide label mode
    label_mode = args.label_source
    if label_mode == "auto":
        if label_candidates:
            label_mode = "fields"
        else:
            label_mode = "caption_spacy" if _have_spacy() else "caption"

    if label_mode == "caption_spacy" and not _have_spacy():
        print("[warn] spaCy not installed; falling back to 'caption' mode.", file=sys.stderr)
        label_mode = "caption"

    captions_map: Dict[str, List[str]] = {}
    labels_map: Dict[str, List[str]] = {}

    # Iteration helpers for streaming vs non-streaming
    start = max(0, int(args.start))
    limit = int(args.limit)

    def _row_iter():
        if args.streaming:
            # IterableDataset: we need to skip & stop manually
            it = iter(ds)
            skipped = 0
            taken = 0
            for row in it:
                if skipped < start:
                    skipped += 1
                    continue
                yield row
                taken += 1
                if taken >= limit:
                    break
        else:
            end = min(len(ds), start + limit)
            for i in range(start, end):
                yield ds[i]

    exported = 0
    for row in _row_iter():
        # image
        try:
            pil = pil_from_any(row[image_field])
        except Exception as e:
            print(f"[warn] skipping due to bad image: {e}", file=sys.stderr)
            continue

        # filename: stable by hashing raw bytes (deterministic)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        digest = hashlib.sha1(buf.getvalue()).hexdigest()[:16]
        rel = f"{digest}.jpg"
        out_path = img_dir / rel
        save_image(pil.convert("RGB"), out_path, jpeg_quality=args.jpeg_quality)

        # captions list
        cap_val = row.get(caption_field, None)
        caps: List[str] = []
        if isinstance(cap_val, list):
            for c in cap_val:
                c2 = normalize_caption(c)
                if c2:
                    caps.append(c2)
        else:
            c2 = normalize_caption(cap_val)
            if c2:
                caps.append(c2)
        if len(caps) > args.max_captions:
            caps = caps[:args.max_captions]
        captions_map[f"images/{rel}"] = caps

        # labels
        labels: List[str] = []
        if label_mode == "fields":
            labels = labels_from_fields(row, label_candidates)
        elif label_mode == "caption_spacy":
            if caps:
                labels = labels_from_caption_spacy(caps[0], max_k=2)
        elif label_mode == "caption":
            if caps:
                labels = labels_from_caption_regex(caps[0], max_k=2)
        elif label_mode == "none":
            labels = []
        else:
            labels = []  # safety

        labels_map[f"images/{rel}"] = labels

        exported += 1
        if exported % 500 == 0:
            print(f"[info] exported {exported}")

        # extra guard for non-streaming: stop if hit limit early
        if not args.streaming and exported >= limit:
            break

    with open(caps_out, "w", encoding="utf-8") as f:
        json.dump(captions_map, f, ensure_ascii=False, indent=2)
    with open(labs_out, "w", encoding="utf-8") as f:
        json.dump(labels_map, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote {exported} items to: {out_dir}")
    print(f" - images/: {len(list(img_dir.glob('*.jpg')))}")
    print(f" - captions.json keys: {len(captions_map)}")
    print(f" - labels.json keys: {len(labels_map)}")
    if label_mode != "fields":
        nonempty = sum(1 for v in labels_map.values() if v)
        print(f" - non-empty labels: {nonempty} ({nonempty/max(1,exported):.1%}) via '{label_mode}' mode")

if __name__ == "__main__":
    import io  # used in pil_from_any
    main()
