#!/usr/bin/env python3
import argparse, json, shutil, re
from pathlib import Path
from tqdm import tqdm

def read_lines(p: Path):
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]

def build_cub_indexes(cub_root: Path):
    # images.txt: "img_id relative/path.jpg"
    id2rel = {}
    for line in read_lines(cub_root / "images.txt"):
        i, rel = line.split(" ", 1)
        id2rel[int(i)] = rel

    id2cls = {}
    for line in read_lines(cub_root / "image_class_labels.txt"):
        i, cid = line.split()
        id2cls[int(i)] = int(cid)

    clsid2name = {}
    for line in read_lines(cub_root / "classes.txt"):
        cid, cname = line.split(" ", 1)
        species = cname.split(".", 1)[-1].replace("_", " ")
        clsid2name[int(cid)] = species

    return id2rel, id2cls, clsid2name

def index_captions(caps_root: Path):
    by_rel, by_id = {}, {}

    def store_rel(cls_name, stem, texts):
        if texts: by_rel[(cls_name.lower(), stem.lower())] = texts
    def store_id(img_id, texts):
        if texts: by_id[int(img_id)] = texts

    # allow passing .../text_c10 or .../text
    text_c10_roots = []
    text_roots = []
    if (caps_root / "text_c10").is_dir(): text_c10_roots.append(caps_root / "text_c10")
    if caps_root.name == "text_c10":       text_c10_roots.append(caps_root)
    if (caps_root / "text").is_dir():      text_roots.append(caps_root / "text")
    if caps_root.name == "text":           text_roots.append(caps_root)

    # ---- text_c10: supports BOTH <class>/<stem>/*.txt AND <class>/<stem>.txt, plus id variants
    for d in text_c10_roots:
        for cls_dir in d.iterdir():
            if not cls_dir.is_dir(): continue
            for item in cls_dir.iterdir():
                if item.is_dir():
                    texts = [p.read_text().strip() for p in sorted(item.glob("*.txt")) if p.read_text().strip()]
                    store_rel(cls_dir.name, item.name, texts)
                elif item.is_file() and item.suffix == ".txt":
                    texts = [ln.strip() for ln in item.read_text().splitlines() if ln.strip()]
                    store_rel(cls_dir.name, item.stem, texts)
        # id-based inside text_c10
        for ent in d.iterdir():
            if ent.is_dir() and ent.name.isdigit():
                texts = [p.read_text().strip() for p in sorted(ent.glob("*.txt")) if p.read_text().strip()]
                store_id(ent.name, texts)
            elif ent.is_file() and ent.suffix == ".txt" and ent.stem.isdigit():
                texts = [ln.strip() for ln in ent.read_text().splitlines() if ln.strip()]
                store_id(ent.stem, texts)

    # ---- text: <class>/<stem>.txt and top-level <id>.txt
    for d in text_roots:
        for cls_dir in d.iterdir():
            if cls_dir.is_dir():
                for f in cls_dir.glob("*.txt"):
                    texts = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
                    store_rel(cls_dir.name, f.stem, texts)
            elif cls_dir.is_file() and cls_dir.suffix == ".txt" and cls_dir.stem.isdigit():
                texts = [ln.strip() for ln in cls_dir.read_text().splitlines() if ln.strip()]
                store_id(cls_dir.stem, texts)

    return by_rel, by_id

def main(cub_root, captions_root, out_root):
    cub_root = Path(cub_root)
    caps_root = Path(captions_root)
    out_root  = Path(out_root)

    img_src_root = cub_root / "images"
    out_img_dir  = out_root / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    id2rel, id2cls, clsid2name = build_cub_indexes(cub_root)
    by_rel, by_id = index_captions(caps_root)

    captions_json, labels_json = {}, {}

    # helper to derive (class, stem) from rel path (case-insensitive match)
    def rel_to_cls_stem(rel_path_str: str):
        rel = Path(rel_path_str)
        return rel.parent.name, rel.stem

    for img_id, rel in tqdm(sorted(id2rel.items()), desc="Preparing"):
        src = img_src_root / rel
        if not src.is_file():  # skip missing
            continue

        # destination name: "<id>__<basename>"
        dst_name = f"{img_id:05d}__{Path(rel).name}"
        dst_path = out_img_dir / dst_name
        if not dst_path.exists():
            shutil.copy2(src, dst_path)

        # label
        species = clsid2name[id2cls[img_id]]
        labels_json[dst_name] = species

        # captions: try by (class, stem) then by id
        cls_name, stem = rel_to_cls_stem(rel)
        caps = by_rel.get((cls_name.lower(), stem.lower()))
        if not caps:
            caps = by_id.get(img_id, [])
        if caps:
            captions_json[dst_name] = caps

    (out_root / "captions.json").write_text(json.dumps(captions_json, indent=2))
    (out_root / "labels.json").write_text(json.dumps(labels_json, indent=2))
    print(f"Images written: {len(labels_json):,}")
    print(f"Images with captions: {len(captions_json):,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub_root", required=True)
    ap.add_argument("--captions_root", required=True)
    ap.add_argument("--out_root", default="embedding/data")
    args = ap.parse_args()
    main(args.cub_root, args.captions_root, args.out_root)
