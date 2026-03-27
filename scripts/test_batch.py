from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

from ultralytics import YOLO


def batch_test(
    image_dir: str | Path = "data/test_samples",
    model_path: str | Path = "models/best_taco.pt",
    conf: float = 0.25,
):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    model = YOLO(str(model_path))

    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    # Skip saved outputs (keeps your folder clean)
    images = [p for p in images if not p.name.startswith("result_")]

    if not images:
        print(f"No images found in {image_dir} (only result_* files?)")
        return

    print(f"Testing {len(images)} images...\n")

    summary: defaultdict[str, int] = defaultdict(int)
    all_results: list[dict] = []

    for img_path in images:
        results = model(str(img_path), conf=conf)
        r = results[0]

        detections = []
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf_val = float(box.conf)
            summary[cls_name] += 1
            detections.append({"class": cls_name, "confidence": round(conf_val, 3)})

        # Save annotated output next to the input image
        r.save(filename=str(image_dir / f"result_{img_path.name}"))
        all_results.append({"image": img_path.name, "detections": detections})
        print(f"  {img_path.name}: {len(detections)} detection(s)")

    print(f"\n--- Summary across {len(images)} images ---")
    for cls, count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {cls:<25} {count} detection(s)")

    out_json = Path("data/test_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {out_json}")


if __name__ == "__main__":
    batch_test()

