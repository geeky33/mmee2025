#!/usr/bin/env python3
import json, argparse, random
from pathlib import Path
from collections import defaultdict

def load_labels(labels_path: Path):
    if not labels_path.exists():
        return {}
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize to: image_relpath -> class_name
    if isinstance(data, list):
        out = {}
        for row in data:
            key = row.get("image_relpath") or row.get("image_path") or row.get("path") or row.get("file")
            cls = row.get("class_name")
            if key and cls:
                out[str(key)] = str(cls)
        return out
    elif isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[str(k)] = str(v.get("class_name", "unlabeled"))
            elif isinstance(v, list):
                out[str(k)] = str(v[0]) if v else "unlabeled"
            else:
                out[str(k)] = str(v) if v else "unlabeled"
        return out
    return {}

def main():
    ap = argparse.ArgumentParser(description="Inject random (bad) captions per image and produce ground-truth flags.")
    ap.add_argument("--captions", required=True, type=Path, help="Path to captions.json")
    ap.add_argument("--labels", type=Path, default=None, help="Path to labels.json (optional, for cross-class sampling)")
    ap.add_argument("--out-captions", type=Path, default=None, help="Output captions_with_noise.json")
    ap.add_argument("--out-gt", type=Path, default=None, help="Output bad_caption_gt.json")
    ap.add_argument("--k", type=int, default=1, help="Number of bad captions to inject per image")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--cross_class_only", action="store_true",
                    help="Inject captions only from *different* classes (requires labels.json)")
    ap.add_argument("--dedup", action="store_true",
                    help="Avoid injecting duplicates of existing captions for an image")
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.captions, "r", encoding="utf-8") as f:
        caps = json.load(f)  # {image_key: [caption, ...]}

    labels = load_labels(args.labels) if args.labels else {}
    # Build class->images map
    class_of = defaultdict(lambda: "unlabeled")
    for k in caps.keys():
        class_of[k] = labels.get(k, "unlabeled")

    # Build pools per class (to support cross_class_only)
    class_to_captions = defaultdict(list)
    global_pool = []
    for img, clist in caps.items():
        for c in clist:
            txt = str(c).strip()
            if not txt:
                continue
            global_pool.append((img, txt))
            class_to_captions[class_of[img]].append((img, txt))

    if not global_pool:
        raise RuntimeError("No captions found to build a pool from.")

    out_caps = {}
    out_gt = {}

    for img, clist in caps.items():
        existing = [str(c).strip() for c in clist if str(c).strip()]
        bad_cands = []

        # choose sampling pool
        if args.cross_class_only and args.labels:
            my_cls = class_of[img]
            pool = []
            for cls, items in class_to_captions.items():
                if cls != my_cls:
                    pool.extend(items)
        else:
            pool = global_pool

        # ensure we don't sample from own image captions if we want strictly mismatched
        pool_filtered = [(im, t) for (im, t) in pool if im != img]

        # optionally avoid duplicates
        if args.dedup:
            existing_set = set(existing)
            pool_filtered = [(im, t) for (im, t) in pool_filtered if t not in existing_set]

        if len(pool_filtered) < args.k:
            # fall back to full pool if too small after filtering
            pool_filtered = [(im, t) for (im, t) in pool if im != img]

        # sample K bad captions
        bad_texts = [t for (_, t) in random.sample(pool_filtered, k=min(args.k, len(pool_filtered)))]
        # Construct outputs
        new_list = existing + bad_texts
        gt = [0]*len(existing) + [1]*len(bad_texts)

        out_caps[img] = new_list
        out_gt[img] = gt

    out_caps_path = args.out_captions or args.captions.with_name("captions_with_noise.json")
    out_gt_path = args.out_gt or args.captions.with_name("bad_caption_gt.json")

    with open(out_caps_path, "w", encoding="utf-8") as f:
        json.dump(out_caps, f, ensure_ascii=False, indent=2)
    with open(out_gt_path, "w", encoding="utf-8") as f:
        json.dump(out_gt, f, ensure_ascii=False, indent=2)

    print(f"Wrote:\n  {out_caps_path}\n  {out_gt_path}")

if __name__ == "__main__":
    main()
