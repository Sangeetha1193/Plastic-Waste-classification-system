"""
Remap YOLO class IDs from one taxonomy to another.

Default usage for this project:
  python scripts/remap_yolo_classes.py

It reads:
  data/taco_yolo/data.yaml
  data/class_maps/map_17_fixed.csv

And writes:
  data/taco_yolo_17/{images,labels,data.yaml}
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml


def load_yaml_names(yaml_path: Path) -> dict[int, str]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: v for i, v in enumerate(names)}


def load_map_csv(path: Path) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    target_order: list[str] = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            src = row[0].strip()
            dst = row[1].strip()
            if not src or not dst:
                continue
            mapping[src] = dst
            if dst not in seen:
                seen.add(dst)
                target_order.append(dst)
    return mapping, target_order


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap YOLO labels to reduced classes.")
    parser.add_argument("--src", default="data/taco_yolo", help="Source YOLO dataset root")
    parser.add_argument("--out", default="data/taco_yolo_17", help="Output YOLO dataset root")
    parser.add_argument(
        "--map-csv",
        default="data/class_maps/map_17_fixed.csv",
        help="CSV mapping: source_class_name,target_class_name",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output folder if exists")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    map_csv = Path(args.map_csv)

    src_yaml = src / "data.yaml"
    if not src_yaml.is_file():
        raise SystemExit(f"Missing source YAML: {src_yaml}")
    if not map_csv.is_file():
        raise SystemExit(f"Missing mapping CSV: {map_csv}")

    src_names = load_yaml_names(src_yaml)
    src_cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    name_to_id = {v: k for k, v in src_names.items()}

    name_map, target_names = load_map_csv(map_csv)
    missing = sorted([n for n in src_names.values() if n not in name_map])
    if missing:
        raise SystemExit(
            "Mapping CSV does not cover all source classes. Missing: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    target_id = {n: i for i, n in enumerate(target_names)}

    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {out}. Use --overwrite to recreate.")
        shutil.rmtree(out)

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy images as-is
    for split in ("train", "val", "test"):
        img_src = src / "images" / split
        img_out = out / "images" / split
        if not img_src.is_dir():
            raise SystemExit(f"Missing image split dir: {img_src}")
        for p in img_src.iterdir():
            if p.is_file():
                shutil.copy2(p, img_out / p.name)

    # Remap labels
    lines_in = 0
    lines_out = 0
    for split in ("train", "val", "test"):
        lbl_src = src / "labels" / split
        lbl_out = out / "labels" / split
        if not lbl_src.is_dir():
            raise SystemExit(f"Missing label split dir: {lbl_src}")
        for txt in lbl_src.glob("*.txt"):
            out_lines = []
            for line in txt.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                src_name = src_names[cls]
                dst_name = name_map[src_name]
                out_cls = target_id[dst_name]
                out_lines.append(f"{out_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                lines_in += 1
                lines_out += 1
            (lbl_out / txt.name).write_text("".join(out_lines), encoding="utf-8")

    # Write output YAML
    out_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(target_names),
        "names": {i: n for i, n in enumerate(target_names)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(out_yaml, sort_keys=False), encoding="utf-8")

    # Also write/update a project-level taco.yaml for evaluation scripts.
    project_yaml = {
        "path": "./" + str(out).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(target_names),
        "names": {i: n for i, n in enumerate(target_names)},
    }
    Path("data/taco.yaml").write_text(yaml.safe_dump(project_yaml, sort_keys=False), encoding="utf-8")

    print(f"Source classes: {len(src_names)}")
    print(f"Target classes: {len(target_names)}")
    print(f"Labels remapped: {lines_out}/{lines_in}")
    print(f"Output dataset: {out}")
    print("Updated: data/taco.yaml")


if __name__ == "__main__":
    main()

