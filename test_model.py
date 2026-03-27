from pathlib import Path
import random

from ultralytics import YOLO


model = YOLO("models/best.pt")

# Point this at any folder of test images.
# Use your TACO test set or grab a few waste photos from Google.
TEST_IMAGES_DIR = Path("data/taco_yolo/Images/test")

# Get 10 random samples.
all_images = list(TEST_IMAGES_DIR.rglob("*.jpg")) + list(TEST_IMAGES_DIR.rglob("*.png"))

if not all_images:
    raise FileNotFoundError(
        f"No test images found in '{TEST_IMAGES_DIR}'. "
        "Add .jpg/.png files and run again."
    )

samples = random.sample(all_images, min(10, len(all_images)))

print(f"Running inference on {len(samples)} images...\n")

for img_path in samples:
    # conf=0.25 means show predictions >25% confidence.
    results = model(str(img_path), conf=0.25)

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"{img_path.name}: no detections")
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0]]
            print(f"{img_path.name}: {cls_name} ({conf:.0%}) @ [{x1},{y1},{x2},{y2}]")

# Save annotated images to outputs/.
output_dir = Path("outputs/predictions")
output_dir.mkdir(parents=True, exist_ok=True)

model(samples, conf=0.25, save=True, project="outputs", name="predictions")
print("\nAnnotated images saved to outputs/predictions/")
