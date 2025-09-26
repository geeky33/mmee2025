#!/usr/bin/env python3
"""
Quick sanity checker for the authenticity GT files used by the Streamlit UI.

It verifies (for a given dataset folder like data/cub-200 or data/ms-coco):
- That both files exist:
    - captions_with_noise.json
    - bad_caption_gt.json
- Total number of GT entries and total positives
- Key format overlap (by relpath and by basename) between GT and captions
- For overlapping keys, checks that the number of flags equals the number of captions
- Prints a few sample items

Usage:
  python tools/check_gt.py --root data/cub-200
  python tools/check_gt.py --root data/ms-coco -n 3
"""

import argparse
import json
from pathlib import Path

def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def as_caption_map(obj):
    """
    Normalize captions file to: Dict[str, List[str]]
    Accepts dict {key: [cap1, cap2, ...]} or series-like.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, list):
                out[str(k)] = [str(x) for x in v]
            elif v is None:
                out[str(k)] = []
            else:
                out[str(k)] = [str(v)]
        return out
    # Fallback: unknown format
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Dataset folder (e.g., data/cub-200 or data/ms-coco)")
    ap.add_argument("-n", "--num-samples", type=int, default=2,
                    help="How many sample entries to print")
    args = ap.parse_args()

    root = Path(args.root)
    p_caps = root / "captions_with_noise.json"
    p_gt   = root / "bad_caption_gt.json"

    print(f"Dataset root          : {root.resolve()}")
    print(f"Captions (with noise) : {p_caps}  (exists: {p_caps.exists()})")
    print(f"GT flags              : {p_gt}  (exists: {p_gt.exists()})")

    caps_raw = load_json(p_caps)
    gt_raw   = load_json(p_gt)

    if caps_raw is None or gt_raw is None:
        print("\n❌ Missing file(s). The UI won’t show authenticity metrics until both exist.")
        return

    caps = as_caption_map(caps_raw)
    gt   = {str(k): list(v) for k, v in gt_raw.items()}

    print("\n--- Basic stats ---")
    print(f"Captions entries     : {len(caps)}")
    print(f"GT entries           : {len(gt)}")
    total_pos = sum(sum(int(x) for x in flags) for flags in gt.values())
    print(f"Total positives (GT) : {total_pos}")

    # Overlap checks (by exact key and by basename)
    cap_keys = set(caps.keys())
    gt_keys  = set(gt.keys())

    overlap_exact = cap_keys & gt_keys
    # also try basename mapping
    caps_by_base = {Path(k).name: k for k in cap_keys}
    gt_basenames = {Path(k).name for k in gt_keys}
    overlap_by_basename = gt_basenames & set(caps_by_base.keys())

    print("\n--- Key overlap ---")
    print(f"Exact-key overlap           : {len(overlap_exact)}")
    print(f"Basename-only overlap       : {len(overlap_by_basename)} "
          f"(useful if GT uses 'img.jpg' but captions use 'images/img.jpg')")

    # Length consistency for overlapping keys (exact-key first)
    def count_mismatches(overlap, use_basename=False):
        bad = 0
        examples = []
        for gk in list(overlap)[:5000]:  # safeguard
            ck = caps_by_base[gk] if use_basename else gk
            # If we came from basenames, translate gk (a basename) to the real captions key
            if use_basename:
                ck = caps_by_base[gk]
                # find the matching GT full key that has this basename
                # (there might be multiple, pick the first)
                candidates = [full for full in gt if Path(full).name == gk]
                if not candidates:
                    continue
                gk_full = candidates[0]
                flags = gt[gk_full]
            else:
                flags = gt[gk]
            caps_list = caps.get(ck, [])
            if len(flags) != len(caps_list):
                bad += 1
                if len(examples) < 5:
                    examples.append((gk if not use_basename else f"(basename) {gk}",
                                     len(flags), len(caps_list)))
        return bad, examples

    bad_exact, ex_exact = count_mismatches(overlap_exact, use_basename=False)
    bad_base,  ex_base  = count_mismatches(overlap_by_basename, use_basename=True)

    print("\n--- Length consistency (flags vs #captions) ---")
    print(f"Exact-key mismatches        : {bad_exact}")
    if ex_exact:
        for k, lf, lc in ex_exact:
            print(f"  • {k}: flags={lf}, captions={lc}")

    print(f"Basename-key mismatches     : {bad_base}")
    if ex_base:
        for k, lf, lc in ex_base:
            print(f"  • {k}: flags={lf}, captions={lc}")

    # Print a few sample entries
    print("\n--- Sample entries ---")
    printed = 0
    for k in gt:
        if printed >= args.num_samples:
            break
        caps_key = k if k in caps else Path(k).name if Path(k).name in caps_by_base else None
        cap_list = caps.get(caps_key if caps_key in caps else caps_by_base.get(Path(k).name, ""), [])
        flags = gt[k]
        print(f"* {k}")
        print(f"  flags  ({len(flags)}): {flags[:min(10, len(flags))]}{' ...' if len(flags)>10 else ''}")
        if cap_list:
            print(f"  caps   ({len(cap_list)}): {cap_list[:min(3, len(cap_list))]}{' ...' if len(cap_list)>3 else ''}")
        else:
            print("  caps   : (no captions found under the same key)")
        printed += 1

    print("\n✅ Done. If overlap is healthy and positives > 0, "
          "the Streamlit UI can show Precision/Recall/F1, confusion matrix, and examples.")

if __name__ == "__main__":
    main()
