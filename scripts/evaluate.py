from __future__ import annotations

import json

from ultralytics import YOLO


def evaluate(data_yaml: str = "data/taco.yaml", model_path: str = "models/best_taco.pt"):
    """
    Evaluate YOLO weights on the `test` split defined in the dataset YAML.

    Expected YAML fields:
      - path
      - train / val / test (relative to `path`)
    """

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split="test")

    print("\n========== Evaluation Results ==========")
    print(f"mAP@0.5       : {float(metrics.box.map50):.4f}  (target: >0.50)")
    print(f"mAP@0.5:0.95  : {float(metrics.box.map):.4f}  (target: >0.30)")
    print(f"Precision     : {float(metrics.box.mp):.4f}")
    print(f"Recall        : {float(metrics.box.mr):.4f}")
    print("=========================================\n")

    # Per-class breakdown (class_id 0..nc-1 aligned with metrics.box.ap50)
    print(f"{'Class':<25} {'AP@0.5':>8}")
    print("-" * 36)

    names = model.names  # usually dict[int,str]
    ap50 = metrics.box.ap50
    for i in range(len(ap50)):
        class_name = names[i] if isinstance(names, dict) else names[i]
        ap_val = float(ap50[i])
        flag = " <-- low" if ap_val < 0.3 else ""
        print(f"  {class_name:<23} {ap_val:.4f}{flag}")

    report = {
        "map50": round(float(metrics.box.map50), 4),
        "map": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    with open("data/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Report saved → data/eval_report.json")


if __name__ == "__main__":
    evaluate()

