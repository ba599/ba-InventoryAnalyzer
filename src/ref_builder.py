import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from src.grid_detector import detect_cells, crop_icon_region


def build_references(
    images: list[np.ndarray],
    item_order: list[str],
    start_id: str,
    output_dir: Path,
) -> int:
    """Build reference icon images from in-game screenshots.

    Args:
        images: List of BGR grid screenshot images.
        item_order: Ordered list of all material_ids.
        start_id: material_id of the first item in the first screenshot.
        output_dir: Directory to save reference images.

    Returns:
        Number of reference images created.
    """
    start_idx = item_order.index(start_id)
    current_idx = start_idx
    count = 0

    for img in images:
        cells = detect_cells(img)
        for cell in cells:
            if current_idx >= len(item_order):
                break
            material_id = item_order[current_idx]
            current_idx += 1
            if material_id is None:
                continue
            icon = crop_icon_region(img, cell)
            out_path = output_dir / f"{material_id}.png"
            cv2.imwrite(str(out_path), icon)
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Build reference images from in-game screenshots")
    parser.add_argument("--images", nargs="+", required=True, help="Screenshot image paths")
    parser.add_argument("--start-id", required=True, help="Material ID of the first visible item")
    parser.add_argument("--order", default="item_order.json", help="Path to item order JSON")
    parser.add_argument("--output", default="references", help="Output directory for reference images")
    args = parser.parse_args()

    order_path = Path(args.order)
    with open(order_path, "r", encoding="utf-8") as f:
        item_order = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping")
            continue
        images.append(img)

    count = build_references(images, item_order, args.start_id, output_dir)
    print(f"Created {count} reference images in {output_dir}/")


if __name__ == "__main__":
    main()
