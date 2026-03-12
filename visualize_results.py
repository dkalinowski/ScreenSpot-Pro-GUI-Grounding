#!/usr/bin/env python3
"""
Visualize ScreenSpot-Pro evaluation results by drawing ground-truth bounding boxes
and predicted click points on each image.

Usage:
    python visualize_results.py --result results/Qwen/Qwen3-VL-2B-Instruct.json \
                                --output_dir visualized_results
"""

import argparse
import json
import os
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ScreenSpot-Pro results on images.")
    parser.add_argument("--result", type=str, required=True, help="Path to the result JSON file.")
    parser.add_argument("--output_dir", type=str, default="visualized_results",
                        help="Directory to save annotated images.")
    parser.add_argument("--bbox_color_correct", type=str, default="green",
                        help="Color for bbox when prediction is correct.")
    parser.add_argument("--bbox_color_wrong", type=str, default="red",
                        help="Color for bbox when prediction is wrong.")
    parser.add_argument("--pred_color", type=str, default="blue",
                        help="Color for the predicted click point.")
    parser.add_argument("--gt_center_color", type=str, default="green",
                        help="Color for the ground-truth center point.")
    parser.add_argument("--line_width", type=int, default=3, help="Line width for bbox rectangle.")
    parser.add_argument("--point_radius", type=int, default=10, help="Radius for click point circles.")
    return parser.parse_args()


def draw_crosshair(draw, x, y, radius, color, width=2):
    """Draw a crosshair marker at (x, y)."""
    draw.line([(x - radius, y), (x + radius, y)], fill=color, width=width)
    draw.line([(x, y - radius), (x, y + radius)], fill=color, width=width)


def annotate_image(entry, args):
    """Draw bbox and predicted point on a single image and save it."""
    img_path = entry["img_path"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(args.result), "../../", img_path)
    img_path = os.path.normpath(img_path)

    if not os.path.exists(img_path):
        print(f"WARNING: Image not found: {img_path}")
        return

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Determine colors based on correctness
    is_correct = entry.get("correctness", "wrong") == "correct"
    bbox_color = args.bbox_color_correct if is_correct else args.bbox_color_wrong

    # Draw ground-truth bounding box
    bbox = entry.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=args.line_width)
        # Draw GT center point
        gt_cx = (x1 + x2) / 2
        gt_cy = (y1 + y2) / 2
        r = args.point_radius
        draw.ellipse([gt_cx - r, gt_cy - r, gt_cx + r, gt_cy + r],
                      fill=args.gt_center_color, outline="white", width=1)

    # Draw predicted click point
    pred = entry.get("pred")
    if pred and len(pred) == 2:
        px, py = pred
        r = args.point_radius
        draw.ellipse([px - r, py - r, px + r, py + r],
                      fill=args.pred_color, outline="white", width=1)
        # Draw crosshair for visibility
        draw_crosshair(draw, px, py, r + 6, args.pred_color, width=2)

    # Draw label text
    label_parts = []
    correctness = entry.get("correctness", "unknown")
    label_parts.append(correctness.upper())
    prompt = entry.get("prompt_to_evaluate", "")
    if prompt:
        max_len = 80
        label_parts.append(prompt[:max_len] + ("..." if len(prompt) > max_len else ""))

    label = " | ".join(label_parts)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Draw text background
    text_bbox = draw.textbbox((10, 10), label, font=font)
    padding = 4
    draw.rectangle(
        [text_bbox[0] - padding, text_bbox[1] - padding,
         text_bbox[2] + padding, text_bbox[3] + padding],
        fill="black"
    )
    draw.text((10, 10), label, fill="white", font=font)

    # Draw legend
    legend_y = text_bbox[3] + 12
    legend_items = [
        (bbox_color, "GT bbox"),
        (args.gt_center_color, "GT center"),
        (args.pred_color, "Predicted"),
    ]
    for color, text in legend_items:
        draw.ellipse([10, legend_y, 22, legend_y + 12], fill=color, outline="white")
        draw.text((28, legend_y - 2), text, fill="white", font=font)
        legend_y += 20

    # Build output path preserving some structure
    app = entry.get("application", "unknown")
    task_filename = entry.get("task_filename", "unknown")
    orig_basename = os.path.splitext(os.path.basename(entry["img_path"]))[0]
    idx = entry.get("_index", 0)
    out_name = f"{idx:04d}_{correctness}_{app}_{orig_basename}.png"
    out_dir = os.path.join(args.output_dir, task_filename)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    img.save(out_path)
    return out_path


def main():
    args = parse_args()

    with open(args.result, "r") as f:
        data = json.load(f)

    details = data.get("details", [])
    if not details:
        print("No 'details' found in the result file.")
        return

    print(f"Processing {len(details)} entries...")
    os.makedirs(args.output_dir, exist_ok=True)

    correct = 0
    wrong = 0
    skipped = 0

    for i, entry in enumerate(details):
        entry["_index"] = i
        result = annotate_image(entry, args)
        if result is None:
            skipped += 1
        elif entry.get("correctness") == "correct":
            correct += 1
        else:
            wrong += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(details)}")

    print(f"\nDone! Saved {correct + wrong} images to '{args.output_dir}/'")
    print(f"  Correct: {correct}, Wrong: {wrong}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
