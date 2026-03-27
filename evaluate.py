from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO


def _local_data_yaml(
    original_yaml_path: Path,
    local_root: Path,
    out_yaml_path: Path,
) -> Path:
    """Create a temp YAML that points the dataset `path:` to the local folder."""
    with open(original_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(local_root.resolve())

    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return out_yaml_path


def evaluate_split(model: YOLO, data_yaml: str, split: str, *, conf: float, iou: float) -> None:
    print(f"\nRunning evaluation on {split} set...\n")
    # Ultralytics will build confusion-matrix axes based on the provided `classes` filter.
    # Your labels contain IDs beyond `nc=28`, and Ultralytics ignores those, which can
    # lead to an incorrect (smaller) confusion-matrix size. Forcing the classes
    # range avoids the out-of-bounds error.
    nc = int(getattr(model.model, "nc", len(model.names)))
    classes_filter = list(range(nc))
    metrics = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        classes=classes_filter,
        plots=True,
        project="outputs",
        name=f"evaluation_{split}",
    )

    print("\n" + "=" * 45)
    print("         EVALUATION RESULTS")
    print("=" * 45)
    print(f"  mAP@0.5:         {metrics.box.map50:.3f}  ({metrics.box.map50 * 100:.1f}%)")
    print(f"  mAP@0.5:0.95:    {metrics.box.map:.3f}  ({metrics.box.map * 100:.1f}%)")
    print(f"  Precision:       {metrics.box.mp:.3f}")
    print(f"  Recall:          {metrics.box.mr:.3f}")
    print("=" * 45)

    # Per-class breakdown
    print(f"\nPer-class AP@0.5 ({split}):")
    print(f"{'Class':<25} {'AP':>8}")
    print("-" * 35)

    # metrics.box.ap50 is indexed by class_id
    ap50 = metrics.box.ap50
    for class_id in range(len(ap50)):
        cls_name = model.names[class_id] if isinstance(model.names, dict) else model.names[class_id]
        ap = float(ap50[class_id])
        bar = "█" * int(ap * 20)
        print(f"  {cls_name:<23} {ap:.3f}  {bar}")

    print(f"\nPlots saved to: outputs/evaluation_{split}/")
    print("  - confusion_matrix.png")
    print("  - PR_curve.png")
    print("  - F1_curve.png")


def main() -> None:
    model = YOLO("models/best.pt")

    # 17-class dataset (used for training)
    original_yaml_path = Path("data/taco_yolo_17/data.yaml")
    local_root = Path("data/taco_yolo_17")

    # Write a temp YAML with corrected `path:` so Ultralytics can find images locally.
    data_yaml_tmp = Path("outputs/evaluation/data_local_17.yaml")
    data_yaml = _local_data_yaml(original_yaml_path, local_root, data_yaml_tmp)

    # Evaluate both splits for the "val/test set" request.
    evaluate_split(model, str(data_yaml), "val", conf=0.25, iou=0.5)
    evaluate_split(model, str(data_yaml), "test", conf=0.25, iou=0.5)


if __name__ == "__main__":
    main()

