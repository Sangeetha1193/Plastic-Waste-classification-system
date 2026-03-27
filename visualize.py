from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize YOLO predictions on random test images.")
    p.add_argument("--weights", default="models/best.pt", help="Path to model weights (.pt).")
    p.add_argument("--test-dir", default="data/taco_yolo/Images/test", help="Folder containing test images.")
    p.add_argument("--n", type=int, default=5, help="Number of random images to sample.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open windows; save annotated images to outputs/visualize/ instead.",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=1200,
        help="Resize display/saved image if wider than this.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    test_dir = Path(args.test_dir)
    all_images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.jpeg")) + list(test_dir.rglob("*.png"))
    if not all_images:
        raise FileNotFoundError(f"No images found under '{test_dir}'.")

    images = random.sample(all_images, min(args.n, len(all_images)))

    out_dir = Path("outputs/visualize")
    if args.no_gui:
        out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        results = model(str(img_path), conf=args.conf)
        annotated = results[0].plot()  # numpy array with boxes drawn (BGR)

        h, w = annotated.shape[:2]
        if w > args.max_width:
            scale = args.max_width / w
            annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

        if args.no_gui:
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)
            print(f"Saved: {out_path}")
            continue

        cv2.imshow(f"Prediction - {img_path.name}", annotated)
        key = cv2.waitKey(0)  # press any key for next image
        if key == ord("q"):  # press q to quit
            break

    if not args.no_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

