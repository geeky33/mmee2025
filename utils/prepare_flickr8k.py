#!/usr/bin/env python3
"""
Prepare Flickr8k for MMEE:
- Parse captions.txt -> captions.json (per-image list of 5 captions)
- Derive simple class labels from captions -> labels.json (string class_name)
- (Optional) Inject noisy captions -> captions_with_noise.json + bad_caption_gt.json
- Copy or link Images/ -> data/<dataset>/images/

Usage:
  python utils/prepare_flickr8k.py \
    --flickr8k-root /path/to/Flickr8k \
    --out data/flickr8k \
    [--link-images] [--noise-rate 0.15]
"""

import argparse, json, os, re, shutil, sys, random
from collections import defaultdict, Counter
from pathlib import Path

def try_import_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None
        return nlp
    except Exception:
        return None

# Very small noun-ish keyword list as fallback if spaCy not available.
FALLBACK_KEYWORDS = [
    "person","man","woman","boy","girl","child","kid","baby","people",
    "dog","cat","bird","horse","cow","sheep","elephant","bear","zebra","giraffe",
    "car","truck","bus","bicycle","motorcycle","boat","train","airplane",
    "ball","kite","frisbee","skateboard","snowboard","surfboard",
    "tree","flower","grass","road","street","building","house","bridge","bench",
    "beach","sea","river","lake","mountain","hill",
]

def parse_captions_txt(captions_txt: Path) -> dict[str, list[str]]:
    """
    Robust parser for multiple Flickr8k variants:
    - 'img.jpg#0<TAB>Caption...'
    - 'img.jpg,Caption...' (CSV)
    - 'img.jpg#0 Caption...' (space after hash)
    - 'img.jpg Caption...'  (space separated)
    Returns: { "img.jpg": [cap1, cap2, ...] }
    """
    per_image: dict[str, list[str]] = defaultdict(list)
    with captions_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            img = None
            cap = None

            # 1) Try TAB
            if "\t" in line:
                left, cap = line.split("\t", 1)
                img = left.split("#")[0]

            # 2) Try CSV (first comma splits filename and caption)
            elif "," in line and line.lower().endswith((".jpg", ".jpeg", ".png")) is False:
                # be careful: captions may also contain commas; only split the first
                parts = line.split(",", 1)
                if len(parts) == 2 and parts[0].lower().endswith((".jpg", ".jpeg", ".png")):
                    img, cap = parts[0], parts[1].strip()

            # 3) Try space after '#idx'  -> 'img.jpg#0 Caption...'
            if img is None or cap is None:
                m = re.match(r"^([^\s#]+?\.(?:jpg|jpeg|png))(?:#\d+)?\s+(.*)$", line, flags=re.IGNORECASE)
                if m:
                    img, cap = m.group(1), m.group(2).strip()

            # 4) Fallback: 'img.jpg#k<SEP>cap' where SEP could be ':' or '|'
            if img is None or cap is None:
                m = re.match(r"^([^\s#]+?\.(?:jpg|jpeg|png))(?:#\d+)?[:|]\s*(.*)$", line, flags=re.IGNORECASE)
                if m:
                    img, cap = m.group(1), m.group(2).strip()

            if img and cap:
                per_image[img].append(cap)

    return per_image


def derive_labels(per_image_caps: dict[str, list[str]], nlp=None) -> dict[str, str]:
    """
    Returns mapping: image_name -> class_name (string).
    Strategy:
      - If spaCy model is present: pick most frequent NOUN lemma across captions
      - Else fallback to keyword scan; pick most frequent matching keyword
      - If nothing found, label 'unlabeled'
    """
    labels: dict[str, str] = {}
    for img, caps in per_image_caps.items():
        votes = []
        if nlp is not None:
            for cap in caps:
                doc = nlp(cap)
                nouns = [t.lemma_.lower() for t in doc if t.pos_ == "NOUN"]
                if nouns:
                    votes.append(nouns[0])
        else:
            # fallback: keyword scan by word boundary
            for cap in caps:
                words = re.findall(r"[a-zA-Z]+", cap.lower())
                for w in words:
                    if w in FALLBACK_KEYWORDS:
                        votes.append(w)
                        break
        if votes:
            labels[img] = Counter(votes).most_common(1)[0][0]
        else:
            labels[img] = "unlabeled"
    return labels

def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def copy_or_link_images(src_images: Path, dst_images: Path, link: bool):
    dst_images.mkdir(parents=True, exist_ok=True)
    if link:
        # Try to hardlink/symlink; fallback to copy for unsupported OS
        for p in sorted(src_images.glob("*.jpg")):
            dst = dst_images / p.name
            if dst.exists(): continue
            try:
                # Try hardlink (works on most OS without admin)
                os.link(p, dst)
            except Exception:
                try:
                    # Try symlink
                    os.symlink(p, dst)
                except Exception:
                    shutil.copy2(p, dst)
    else:
        # Copy all images
        for p in sorted(src_images.glob("*.jpg")):
            shutil.copy2(p, dst_images / p.name)

def make_noise_files(per_image_caps: dict[str, list[str]], noise_rate: float,
                     out_caps_with_noise: Path, out_bad_gt: Path):
    """
    Noise model:
      - For each image, with probability noise_rate, replace ONE of its captions with a random caption from pool
      - bad_caption_gt.json format: image_name -> [0/1,...] flags per caption index
    """
    rng = random.Random(42)

    # Flatten pool of (img, caption)
    all_caps = [(img, cap) for img, caps in per_image_caps.items() for cap in caps]
    all_text_pool = [cap for _, cap in all_caps]

    caps_noisy = {}
    bad_gt = {}

    for img, caps in per_image_caps.items():
        flags = [0] * len(caps)
        new_caps = caps[:]
        if rng.random() < noise_rate and len(all_text_pool) > 0:
            # pick a random position to corrupt
            j = rng.randrange(len(caps))
            # ensure we don't pick from same image caption to make it truly mismatched more often
            corrupt = rng.choice(all_text_pool)
            # if by chance same text, pick again a couple times
            tries = 0
            while corrupt == caps[j] and tries < 5:
                corrupt = rng.choice(all_text_pool); tries += 1
            new_caps[j] = corrupt
            flags[j] = 1
        caps_noisy[img] = new_caps
        bad_gt[img] = flags

    write_json(caps_noisy, out_caps_with_noise)
    write_json(bad_gt, out_bad_gt)

def main():
    ap = argparse.ArgumentParser(description="Prepare Flickr8k for MMEE.")
    ap.add_argument("--flickr8k-root", required=True,
                    help="Path containing Images/ and captions.txt")
    ap.add_argument("--out", required=True,
                    help="Output dataset dir (e.g., data/flickr8k)")
    ap.add_argument("--dataset-name", default=None,
                    help="Optional: Name printed in logs only")
    ap.add_argument("--link-images", action="store_true",
                    help="Hardlink/symlink images instead of copying")
    ap.add_argument("--noise-rate", type=float, default=0.0,
                    help="Fraction (0..1) of images to corrupt one caption; 0 to skip")
    args = ap.parse_args()

    root = Path(args.flickr8k_root).resolve()
    out_root = Path(args.out).resolve()
    ds_name = args.dataset_name or out_root.name

    images_src = root / "Images"
    captions_txt = root / "captions.txt"

    if not images_src.exists():
        sys.exit(f"ERROR: {images_src} not found")
    if not captions_txt.exists():
        sys.exit(f"ERROR: {captions_txt} not found")

    print(f"==> Preparing Flickr8k -> {out_root} (dataset={ds_name})")
    out_images = out_root / "images"
    out_caps = out_root / "captions.json"
    out_labels = out_root / "labels.json"
    out_caps_noise = out_root / "captions_with_noise.json"
    out_bad_gt = out_root / "bad_caption_gt.json"

    # 1) Parse captions.txt
    per_image_caps = parse_captions_txt(captions_txt)
    n_imgs = len(per_image_caps)
    n_caps = sum(len(v) for v in per_image_caps.values())
    print(f"   - Found {n_imgs} images with {n_caps} total captions")

    # 2) Derive labels (spaCy if available; else fallback)
    nlp = try_import_spacy()
    labels_map = derive_labels(per_image_caps, nlp=nlp)
    n_unlabeled = sum(1 for v in labels_map.values() if v == "unlabeled")
    print(f"   - Derived labels for {n_imgs - n_unlabeled} images "
          f"({n_unlabeled} unlabeled); unique classes = {len(set(labels_map.values()))}")

    # 3) Copy or link images
    print(f"   - {'Linking' if args.link_images else 'Copying'} images -> {out_images}")
    copy_or_link_images(images_src, out_images, link=args.link_images)

    # 4) Write captions.json + labels.json
    print(f"   - Writing {out_caps}")
    write_json(per_image_caps, out_caps)

    # labels.json shape expected by your app can simply be: {img: 'class_name'}
    print(f"   - Writing {out_labels}")
    write_json(labels_map, out_labels)

    # 5) Optional: add noise files
    if args.noise_rate and args.noise_rate > 0.0:
        print(f"   - Injecting noisy captions at rate={args.noise_rate:.2f}")
        make_noise_files(per_image_caps, args.noise_rate, out_caps_noise, out_bad_gt)
        print(f"   - Wrote {out_caps_noise} and {out_bad_gt}")

    print("✅ Done.")
    print(f"Now point the app to dataset: {out_root}")
    print("In Streamlit, select this dataset and compute embeddings for images & text.")

if __name__ == "__main__":
    main()
