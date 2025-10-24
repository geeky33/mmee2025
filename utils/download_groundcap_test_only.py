# utils/download_groundcap_test_only.py
from datasets import load_dataset, Image as HFImage
from pathlib import Path
import shutil, json

# 1) Load only the test split
ds = load_dataset("daniel3303/GroundCap", split="test")

# 2) Force "image" column to keep file paths (no decoding)
ds = ds.cast_column("image", HFImage(decode=False))

# 3) Output dir
out_dir = Path.home() / "Datasets/GroundCap_test"
img_out = out_dir / "images"
img_out.mkdir(parents=True, exist_ok=True)

print(f"📦 Saving {len(ds)} GroundCap test samples to {out_dir}")

annotations = []
for i, sample in enumerate(ds):
    img_info = sample["image"]
    caption = sample["caption"]
    objects = sample.get("objects", [])

    # Primary path (when decode=False)
    src_path = img_info.get("path") if isinstance(img_info, dict) else None

    # File name decision
    if src_path:
        fname = Path(src_path).name
    else:
        # Fallback: if decoded somehow, synthesize a name
        fname = f"frame_{i:05d}.jpg"

    dst_path = img_out / fname

    # Save/copy image
    if src_path:
        # Copy from HF cache path
        shutil.copy2(src_path, dst_path)
    else:
        # Fallback: image is decoded; save PIL image object
        pil_img = sample["image"]
        pil_img.save(dst_path, format="JPEG")

    # Collect minimal annotation (relative name only)
    annotations.append({
        "image": fname,
        "caption": caption,
        "objects": objects
    })

# 4) Write annotations JSON
out_json = out_dir / "groundcap_test_annotations.json"
with out_json.open("w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(" Done! Test split saved under:", out_dir)
print(f"   - images/: {len(list(img_out.glob('*.jpg')))} files")
print(f"   - {out_json.name}: {out_json.stat().st_size/1024:.1f} KB")
