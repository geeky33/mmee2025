import argparse, json, os, shutil, sys
from collections import defaultdict
from pathlib import Path

def try_symlink(src: Path, dst: Path) -> bool:
    try:
        if dst.exists() or dst.is_symlink():
            return True
        dst.symlink_to(src, target_is_directory=True)
        return True
    except Exception:
        return False

def safe_copy_subset(src_dir: Path, dst_dir: Path, filenames):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, fn in enumerate(sorted(filenames)):
        s = src_dir / fn
        d = dst_dir / fn
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.exists():
            shutil.copy2(s, d)

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_id_to_file(*image_lists):
    id2file = {}
    for imgs in image_lists:
        if not imgs: 
            continue
        for im in imgs:
            id2file[im["id"]] = im["file_name"]
    return id2file

def main():
    ap = argparse.ArgumentParser(description="Convert COCO val2017 to images/ + captions.json + labels.json")
    ap.add_argument("--coco-dir", type=Path, required=True, help="Path to data/ms-coco")
    ap.add_argument("--split", default="val2017", help="Split name (default: val2017)")
    ap.add_argument("--max-caps-per-image", type=int, default=5)
    ap.add_argument("--copy-instead-of-symlink", action="store_true", help="Force copy images instead of symlink")
    args = ap.parse_args()

    coco_dir = args.coco_dir
    split = args.split
    images_src = coco_dir / split
    ann_dir = coco_dir / "annotations"
    cap_json = ann_dir / f"captions_{split}.json"
    inst_json = ann_dir / f"instances_{split}.json"
    out_dir = coco_dir

    if not images_src.exists():
        sys.exit(f"[!] Missing images folder: {images_src}")
    if not cap_json.exists() or not inst_json.exists():
        sys.exit(f"[!] Missing COCO annotations in {ann_dir}")

    cap = load_json(cap_json)
    inst = load_json(inst_json)

    # Build mappings
    # Use union of images listed across both files
    id2file = build_id_to_file(cap.get("images", []), inst.get("images", []))

    # captions: filename -> list[str]
    caps = defaultdict(list)
    for a in cap.get("annotations", []):
        img_id = a["image_id"]
        fn = id2file.get(img_id)
        if fn:
            if len(caps[fn]) < args.max_caps_per_image:
                txt = (a.get("caption") or "").strip()
                if txt:
                    caps[fn].append(txt)

    # labels: filename -> list[str] (multi-label)
    cat_id2name = {c["id"]: c["name"] for c in inst.get("categories", [])}
    labels_set = defaultdict(set)
    for a in inst.get("annotations", []):
        img_id = a["image_id"]
        fn = id2file.get(img_id)
        if fn:
            cname = cat_id2name.get(a["category_id"])
            if cname:
                labels_set[fn].add(cname)
    labels = {k: sorted(v) for k, v in labels_set.items()}

    # Ensure filenames present in at least one mapping are kept, and fill missing with []
    all_fns = set(id2file.values())
    keep_fns = set(caps.keys()) | set(labels.keys())
    if not keep_fns:
        sys.exit("[!] No images found after parsing annotations.")
    # Optionally restrict to those that have captions (common for captioning)
    keep_fns = set(caps.keys())  # comment this line if you want the union

    # Trim mappings to the kept set & fill empties
    caps = {k: v for k, v in caps.items() if k in keep_fns}
    labels = {k: labels.get(k, []) for k in keep_fns}

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "captions.json").open("w", encoding="utf-8") as f:
        json.dump(caps, f, ensure_ascii=False)
    with (out_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)

    # Make images/ folder inside ms-coco (symlink to val2017 or copy subset)
    images_dst = out_dir / "images"
    if images_dst.exists() and not images_dst.is_symlink():
        print(f"[i] {images_dst} already exists (not a symlink). Leaving as-is.")
    else:
        ok = False
        if not args.copy_instead_of_symlink:
            ok = try_symlink(images_src.resolve(), images_dst)
        if not ok:
            print("[i] Symlink unavailable; copying only used files to images/ ...")
            if images_dst.exists() or images_dst.is_symlink():
                images_dst.unlink(missing_ok=True)
            safe_copy_subset(images_src, images_dst, keep_fns)

    print(f"[✓] Wrote {out_dir/'captions.json'} and {out_dir/'labels.json'}")
    print(f"[✓] Images folder ready at {images_dst} (symlink or subset copy)")
    print(f"[i] Images with captions kept: {len(caps)}")

if __name__ == "__main__":
    main()
