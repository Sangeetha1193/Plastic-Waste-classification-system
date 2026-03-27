"""
Convert TACO COCO-style batch annotations (batch_1 ... batch_N) to YOLO detection format.

Each batch has its own annotations.json. The same file_name can appear in different batches
with different image dimensions and boxes — images are keyed by (batch_id, file_name).

Expected layout:
  taco_data/
    batch_1/annotations.json  (+ image files next to json when downloaded)
    batch_2/...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    if wn <= 0 or hn <= 0:
        return None
    return clamp01(xc), clamp01(yc), clamp01(wn), clamp01(hn)


def stem_for_batch_image(batch_id: int, file_name: str) -> str:
    base = Path(file_name).stem
    return f"batch_{batch_id:02d}_{base}"


def load_categories(first_json: dict) -> list[str]:
    cats = sorted(first_json["categories"], key=lambda c: c["id"])
    ids = [c["id"] for c in cats]
    if ids != list(range(len(cats))):
        raise SystemExit(
            "Unexpected category ids: expected contiguous 0..N-1. "
            "Adjust mapping in this script if your export differs."
        )
    return [c["name"] for c in cats]


def main() -> None:
    p = argparse.ArgumentParser(description="TACO batches (COCO JSON) → YOLO labels + data.yaml")
    p.add_argument(
        "--taco-data",
        type=Path,
        default=None,
        help="Path to TACO data dir containing batch_1, batch_2, ...",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dataset root (will contain images/, labels/, data.yaml)",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction for training (default 0.8)",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction for validation (default 0.1)",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction for test (default 0.1)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for train/val/test split")
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image files from each batch folder into the YOLO dataset when present",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    taco_data = args.taco_data or (repo_root.parent / "TACO" / "TACO" / "data")
    out_root = args.out or (repo_root / "data" / "taco_yolo")

    if not taco_data.is_dir():
        raise SystemExit(f"TACO data directory not found: {taco_data}")

    batch_dirs = sorted(
        d for d in taco_data.iterdir() if d.is_dir() and d.name.startswith("batch_")
    )
    if not batch_dirs:
        raise SystemExit(f"No batch_* folders under {taco_data}")

    records: list[dict] = []
    categories: list[str] | None = None

    for bdir in batch_dirs:
        ann_path = bdir / "annotations.json"
        if not ann_path.is_file():
            continue
        try:
            batch_id = int(bdir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if categories is None:
            categories = load_categories(data)
        else:
            other = load_categories(data)
            if other != categories:
                raise SystemExit(f"Category mismatch in {ann_path}")

        img_by_id = {im["id"]: im for im in data["images"]}
        anns_by_img: dict[int, list] = {}
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for im in data["images"]:
            iid = im["id"]
            stem = stem_for_batch_image(batch_id, im["file_name"])
            src_img = bdir / im["file_name"]
            records.append(
                {
                    "batch_id": batch_id,
                    "stem": stem,
                    "ext": Path(im["file_name"]).suffix or ".jpg",
                    "src_image": src_img,
                    "width": int(im["width"]),
                    "height": int(im["height"]),
                    "annotations": anns_by_img.get(iid, []),
                }
            )

    assert categories is not None
    nc = len(categories)

    tr, vr, ter = args.train_ratio, args.val_ratio, args.test_ratio
    if abs(tr + vr + ter - 1.0) > 1e-6:
        raise SystemExit(f"train-ratio + val-ratio + test-ratio must sum to 1.0 (got {tr + vr + ter})")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n = len(records)
    n_test = int(n * ter + 0.5)
    n_val = int(n * vr + 0.5)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise SystemExit(f"Split yields non-positive train count (n={n}, train={n_train})")

    for sub in ("images", "labels"):
        tree = out_root / sub
        if tree.is_dir():
            shutil.rmtree(tree)

    train_dir_img = out_root / "images" / "train"
    train_dir_lbl = out_root / "labels" / "train"
    val_dir_img = out_root / "images" / "val"
    val_dir_lbl = out_root / "labels" / "val"
    test_dir_img = out_root / "images" / "test"
    test_dir_lbl = out_root / "labels" / "test"
    for d in (
        train_dir_img,
        train_dir_lbl,
        val_dir_img,
        val_dir_lbl,
        test_dir_img,
        test_dir_lbl,
    ):
        d.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    n_skip_box = 0
    n_missing_img = 0
    n_copied = 0

    for idx, rec in enumerate(records):
        if idx < n_test:
            img_dir, lbl_dir = test_dir_img, test_dir_lbl
        elif idx < n_test + n_val:
            img_dir, lbl_dir = val_dir_img, val_dir_lbl
        else:
            img_dir, lbl_dir = train_dir_img, train_dir_lbl

        stem = rec["stem"]
        ext = rec["ext"]
        out_img = img_dir / f"{stem}{ext}"
        out_lbl = lbl_dir / f"{stem}.txt"

        lines: list[str] = []
        for ann in rec["annotations"]:
            cid = int(ann["category_id"])
            if cid < 0 or cid >= nc:
                n_skip_box += 1
                continue
            yolo = coco_bbox_to_yolo(ann["bbox"], rec["width"], rec["height"])
            if yolo is None:
                n_skip_box += 1
                continue
            xc, yc, wn, hn = yolo
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
            n_lines += 1

        out_lbl.write_text("".join(lines), encoding="utf-8")

        src = rec["src_image"]
        if args.copy_images:
            if src.is_file():
                shutil.copy2(src, out_img)
                n_copied += 1
            else:
                n_missing_img += 1

    yaml_names = "\n".join(f"  {i}: {name}" for i, name in enumerate(categories))
    data_yaml = f"""# TACO → YOLO (generated by scripts/convert_taco_to_yolo.py)
# Split: train {n_train} / val {n_val} / test {n_test} (seed={args.seed})
path: {out_root.as_posix()}
train: images/train
val: images/val
test: images/test
nc: {nc}
names:
{yaml_names}
"""
    (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"Batches used: {len(batch_dirs)}")
    print(f"Images (unique batch+file): {len(records)}")
    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")
    print(f"Label lines written: {n_lines}  (skipped boxes: {n_skip_box})")
    print(f"Output: {out_root}")
    if args.copy_images:
        print(f"Images copied: {n_copied}  missing source files: {n_missing_img}")
    else:
        print("Images not copied; use --copy-images after downloading images into each batch_* folder.")


if __name__ == "__main__":
    main()
