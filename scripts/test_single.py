"""
Quick single-image inference + print recycling tips.

Usage:
  python scripts/test_single.py path/to/image.jpg
"""

import sys
from pathlib import Path

from ultralytics import YOLO


RECYCLING_TIPS = {
    "bottle": "Rinse and place in blue recycling bin",
    "plastic_bag": "Return to supermarket plastic bag drop-off",
    "plastic_container": "Remove food residue, then blue bin",
    "can": "Rinse and crush — blue recycling bin",
    "carton": "Flatten and place in blue bin",
    "paper": "Dry paper only — blue recycling bin",
    "paper_bag": "Blue recycling bin",
    "cardboard": "Flatten — blue recycling bin",
    "glass_jar": "Rinse — green glass bank",
    "bottle_cap": "Remove from bottle — grey bin",
    "cigarette": "Grey general waste bin",
    "food_waste": "Brown compost bin",
    "styrofoam": "Grey general waste — not recyclable",
    "battery": "Hazardous waste drop-off point",
    "broken_glass": "Wrap safely — grey general waste",
}


def get_tip(class_name: str) -> str:
    for key, tip in RECYCLING_TIPS.items():
        if key in class_name.lower():
            return tip
    return "Check local recycling guidelines"


def run_inference(
    image_path: str | Path,
    model_path: str | Path = "models/best_taco.pt",
    conf: float = 0.25,
    save: bool = True,
):
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(str(model_path))
    results = model(str(image_path), conf=conf)
    r = results[0]

    print(f"\nImage : {image_path}")
    print(f"Found : {len(r.boxes)} object(s)\n")
    print(f"{'Class':<22} {'Confidence':>10}  Recycling tip")
    print("-" * 70)

    for box in r.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf_val = float(box.conf)
        tip = get_tip(cls_name)
        print(f"{cls_name:<22} {conf_val:>9.1%}  {tip}")

    if save:
        out_dir = Path("data/test_samples")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("result_" + image_path.name)
        # Ultralytics handles drawing + saving to the provided filename.
        r.save(filename=str(out_path))
        print(f"\nAnnotated image saved → {out_path}")

    return r


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "data/test_samples/plastic_bottle.jpg"
    run_inference(img)

