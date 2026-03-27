from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

# Use non-interactive backend so script works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    in_json = "data/test_results.json"
    with open(in_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    class_confs: dict[str, list[float]] = defaultdict(list)

    for r in results:
        for d in r["detections"]:
            class_confs[d["class"]].append(d["confidence"])

    classes = sorted(class_confs.keys())
    if not classes:
        raise SystemExit("No detections found in data/test_results.json")

    avg_confs = [sum(class_confs[c]) / len(class_confs[c]) for c in classes]
    counts = [len(class_confs[c]) for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1 — detection counts per class
    ax1.barh(classes, counts, color="#1D9E75")
    ax1.set_xlabel("Detection count")
    ax1.set_title("Detections per class")
    ax1.invert_yaxis()

    # Chart 2 — average confidence per class
    colors = [
        "#1D9E75" if c >= 0.6 else "#EF9F27" if c >= 0.4 else "#E24B4A"
        for c in avg_confs
    ]
    ax2.barh(classes, avg_confs, color=colors)
    ax2.set_xlabel("Average confidence")
    ax2.set_title("Avg confidence per class")
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax2.invert_yaxis()

    plt.tight_layout()
    out_png = "data/results_chart.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved -> {out_png}")


if __name__ == "__main__":
    main()

