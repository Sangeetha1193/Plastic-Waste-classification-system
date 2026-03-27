from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.25))
IOU_THRESH = float(os.environ.get("IOU_THRESH", 0.45))
MAX_IMG_SIZE = int(os.environ.get("MAX_IMG_SIZE", 10 * 1024 * 1024))  # 10 MB
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recycling tips
# NOTE: one entry per TACO class name (must match `model.names` values)
# This app expects the reduced 17-class model.
# ---------------------------------------------------------------------------

RECYCLING_TIPS = {
    "Aluminium foil": {
        "tip": "Scrunch into a ball and place in metal recycling bin.",
        "bin": "Metal (grey bin)",
        "colour": "#C0C0C0",
    },
    "Other": {
        "tip": "If unsure, place in general waste to avoid contaminating recycling.",
        "bin": "General waste (black bin)",
        "colour": "#888888",
    },
    "Plastic bottle": {
        "tip": "Rinse, remove cap, and place in plastic recycling.",
        "bin": "Plastic recycling (blue bin)",
        "colour": "#4488FF",
    },
    "Glass bottle": {
        "tip": "Rinse, remove any cap/lid, and place in glass recycling bank/bin.",
        "bin": "Glass (green bank)",
        "colour": "#44AA44",
    },
    "Plastic bottle cap": {
        "tip": "Separate from bottle. Plastic caps generally go in plastic recycling if your council accepts them.",
        "bin": "Plastic recycling (blue bin)",
        "colour": "#4488FF",
    },
    "Metal bottle cap": {
        "tip": "Separate from bottle. Metal caps go in metal recycling.",
        "bin": "Metal (grey bin)",
        "colour": "#C0C0C0",
    },
    "Can": {
        "tip": "Rinse and place in metal recycling. No need to crush.",
        "bin": "Metal (grey bin)",
        "colour": "#C0C0C0",
    },
    "Carton": {
        "tip": "Empty, rinse, flatten. Most councils accept in dry recycling.",
        "bin": "Dry recycling (blue bin)",
        "colour": "#4488FF",
    },
    "Cup": {
        "tip": "Check for recycling symbol. Many takeaway cups are not recyclable kerbside.",
        "bin": "General waste (black bin)",
        "colour": "#888888",
    },
    "Plastic lid": {
        "tip": "Metal lids go in metal recycling. Plastic lids go in plastic recycling if accepted locally.",
        "bin": "Check material",
        "colour": "#AAAAAA",
    },
    "Paper": {
        "tip": "Keep dry. Place in paper/card recycling. Remove plastic windows.",
        "bin": "Paper (blue bin)",
        "colour": "#4488FF",
    },
    "Plastic film": {
        "tip": "Clean and ensure it is not contaminated. Many locations require drop-off for film/plastic bags.",
        "bin": "Plastic bag / film drop-off",
        "colour": "#4488FF",
    },
    "Wrapper": {
        "tip": "If it is not a recognised recyclable wrapper, place in general waste to avoid contamination.",
        "bin": "General waste (black bin)",
        "colour": "#888888",
    },
    "Plastic container": {
        "tip": "Rinse clean, check recycling symbol, and place in plastic recycling.",
        "bin": "Plastic (blue bin)",
        "colour": "#4488FF",
    },
    "Pop tab": {
        "tip": "Attach to can or collect separately for metal recycling.",
        "bin": "Metal (grey bin)",
        "colour": "#C0C0C0",
    },
    "Straw": {
        "tip": "Plastic straws are typically not recyclable kerbside. Dispose responsibly.",
        "bin": "General waste (black bin)",
        "colour": "#888888",
    },
    "Styrofoam piece": {
        "tip": "Not usually kerbside recyclable. Some councils have EPS drop-off points.",
        "bin": "General waste / EPS drop-off",
        "colour": "#888888",
    },
}

DEFAULT_TIP = {
    "tip": "Check your local council recycling guidelines for this item.",
    "bin": "Check locally",
    "colour": "#AAAAAA",
}

# ---------------------------------------------------------------------------
# App + model init
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMG_SIZE

log.info("Loading model from %s ...", MODEL_PATH)
model = YOLO(MODEL_PATH)
log.info("Model ready — %d classes", len(model.names))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


def read_image_from_request(file_storage) -> np.ndarray:
    """Read a FileStorage object into a numpy BGR image."""
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Check the file is a valid image.")
    return img


def cls_id_to_name(cls_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    try:
        return str(names[cls_id])
    except Exception:
        return str(cls_id)


def build_prediction(box, img_w: int, img_h: int) -> dict:
    """Turn a single YOLO box into a clean JSON-serialisable dict."""
    x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0]]
    conf = round(float(box.conf[0]), 4)
    cls_id = int(box.cls[0])
    cls_name = cls_id_to_name(cls_id)

    tip_data = RECYCLING_TIPS.get(cls_name, DEFAULT_TIP)

    return {
        "class_id": cls_id,
        "class_name": cls_name,
        "confidence": conf,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": round(x2 - x1, 2),
            "height": round(y2 - y1, 2),
            # Normalised 0–1 (useful for frontend scaling)
            "x1_norm": round(x1 / img_w, 4),
            "y1_norm": round(y1 / img_h, 4),
            "x2_norm": round(x2 / img_w, 4),
            "y2_norm": round(y2 / img_h, 4),
        },
        "recycling": {
            "tip": tip_data["tip"],
            "bin": tip_data["bin"],
            "colour": tip_data["colour"],
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the web UI."""
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_PATH,
            "classes": len(model.names),
        }
    )


@app.route("/metrics")
def metrics():
    """Return model metadata. Plug in your real eval numbers here."""
    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        import json

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)

    return jsonify(
        {
            "model": "YOLO",
            "dataset": "TACO (17-class)",
            "mAP50": "run evaluate.py to populate",
            "precision": "run evaluate.py to populate",
            "recall": "run evaluate.py to populate",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a multipart image upload and return detections.

    Request:
        POST /predict
        Content-Type: multipart/form-data
        Body: image=<file>
        Optional query params:
            conf  (float, default 0.25)
            iou   (float, default 0.45)
    """
    if "image" not in request.files:
        return (
            jsonify(
                {"success": False, "error": "No 'image' field in request. Use multipart/form-data with key 'image'."}
            ),
            400,
        )

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return (
            jsonify(
                {"success": False, "error": f"File type not allowed. Accepted: {', '.join(sorted(ALLOWED_EXT))}"}
            ),
            400,
        )

    try:
        conf = float(request.args.get("conf", CONF_THRESH))
        iou = float(request.args.get("iou", IOU_THRESH))
    except ValueError:
        return jsonify({"success": False, "error": "conf and iou must be floats between 0 and 1."}), 400

    if not (0 < conf < 1) or not (0 < iou < 1):
        return jsonify({"success": False, "error": "conf and iou must be between 0 and 1 (exclusive)."}), 400

    try:
        img = read_image_from_request(file)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    img_h, img_w = img.shape[:2]

    try:
        t0 = time.perf_counter()
        results = model(img, conf=conf, iou=iou, verbose=False)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        log.exception("Inference failed")
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500

    predictions = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            predictions.append(build_prediction(box, img_w, img_h))

    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return jsonify(
        {
            "success": True,
            "image_size": {"width": img_w, "height": img_h},
            "inference_ms": elapsed,
            "count": len(predictions),
            "predictions": predictions,
        }
    )


# ---------------------------------------------------------------------------
# Static files (optional)
# ---------------------------------------------------------------------------


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info("Starting server on http://0.0.0.0:%d  debug=%s", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
