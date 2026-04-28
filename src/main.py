import argparse
from pathlib import Path
import cv2
from src.core.pipeline import load_item_order, process_image_streaming
from src.grid_detector import detect_cells, crop_icon_region, crop_text_region
from src.item_matcher import ItemMatcher
from src.count_ocr_backend import build_backend
from src.json_updater import load_json, update_owned_materials, save_json


def process_screenshots_sequential(
    images: list,
    item_order: list[str | None],
    reader,
    confidence_threshold: float = 0.9,
    start_id: str | None = None,
    matcher: "ItemMatcher | None" = None,
) -> dict[str, tuple[int, float]]:
    """Process screenshots sequentially.

    Modes:
    - matcher given: uses streaming pipeline (auto-scan + 7-match consensus)
    - start_id given (no matcher): first cell of first image = start_id

    Returns:
        {material_id: (quantity, confidence)}
    """
    results: dict[str, tuple[int, float]] = {}

    if matcher is not None:
        # Auto-scan mode: use streaming pipeline
        for img in images:
            for cell_result in process_image_streaming(img, item_order, matcher, reader):
                mid = cell_result.material_id
                if mid not in results:
                    results[mid] = (cell_result.quantity, cell_result.confidence)
                    flag = "[CHECK] " if cell_result.confidence < confidence_threshold else ""
                    print(f"  {flag}{mid}: {cell_result.quantity} ({cell_result.confidence:.2f})")
        return results

    # Manual start_id mode: walk cells sequentially
    current_idx = None
    step = 1

    for img in images:
        cells = detect_cells(img)
        if not cells:
            print("  Warning: No cells detected, skipping image")
            continue

        print(f"  Detected {len(cells)} cells")

        if current_idx is None:
            if start_id is not None and start_id in item_order:
                current_idx = item_order.index(start_id)
            else:
                print(f"  Error: start_id '{start_id}' not found in item order")
                return results

        for cell in cells:
            if not (0 <= current_idx < len(item_order)):
                break

            material_id = item_order[current_idx]

            if material_id is None:
                current_idx += step
                continue

            current_idx += step

            text_img = crop_text_region(img, cell)
            result = reader.read_quantity(text_img)

            if result is not None:
                qty, conf = result
                if material_id not in results:
                    results[material_id] = (qty, conf)
                    flag = "[CHECK] " if conf < confidence_threshold else ""
                    print(f"  {flag}{material_id}: {qty} ({conf:.2f})")
                else:
                    print(f"  {material_id}: duplicate (keeping {results[material_id][0]})")
            else:
                print(f"  {material_id}: OCR failed, skipping")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Blue Archive inventory screenshots")
    parser.add_argument("--images", nargs="+", required=True, help="Screenshot image paths")
    parser.add_argument("--json", required=True, help="Justin Planner JSON path")
    parser.add_argument("--order", default="item_order.json", help="Item order JSON path")
    parser.add_argument("--start-id", help="Start material ID (skips template matching, processes images sequentially)")
    parser.add_argument("--refs", default="references", help="Reference images directory (unused with --start-id)")
    parser.add_argument("--output", help="Output JSON path (default: overwrite input)")
    parser.add_argument("--check", help="Answer JSON path for accuracy check")
    parser.add_argument("--confidence-threshold", type=float, default=0.9, help="Confidence threshold for [CHECK] flag (default: 0.9)")
    args = parser.parse_args()

    item_order, name_map = load_item_order(Path(args.order))
    reader = build_backend()
    data = load_json(Path(args.json))

    valid_items = [x for x in item_order if x is not None]
    print(f"Item order: {len(valid_items)} items ({len(item_order)} total with skips)")

    # Load images
    loaded_images = []
    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_path}")
            continue
        loaded_images.append((img_path, img))

    if not loaded_images:
        print("No valid images provided")
        return

    # Process
    all_results: dict[str, tuple[int, float]] = {}
    images = [img for _, img in loaded_images]

    if args.start_id:
        # Direct sequential: first cell = start_id
        print(f"\nSequential mode: starting from {args.start_id}")
        results = process_screenshots_sequential(
            images, item_order, reader, args.confidence_threshold,
            start_id=args.start_id,
        )
    else:
        # Auto-scan: find first matching item via template matching
        matcher = ItemMatcher(Path(args.refs))
        print(f"\nAuto-scan mode: loaded {len(matcher.references)} references")
        results = process_screenshots_sequential(
            images, item_order, reader, args.confidence_threshold,
            matcher=matcher,
        )

    all_results.update(results)

    # Summary
    qty_only = {k: v[0] for k, v in all_results.items()}
    print(f"\nTotal items recognized: {len(all_results)}")

    # Collect all items needing manual review (both reasons)
    existing = data.get("owned_materials", {})
    review_items: dict[str, list[str]] = {}  # mid -> list of reasons

    for mid, (qty, conf) in all_results.items():
        reasons = []
        if conf < args.confidence_threshold:
            reasons.append(f"low confidence ({conf:.2f})")
        if mid in existing:
            old_val = int(existing[mid]) if existing[mid] else 0
            if abs(qty - old_val) >= 100:
                reasons.append(f"deviation {qty - old_val:+d} (was {old_val})")
        if reasons:
            review_items[mid] = reasons

    if review_items:
        print(f"\n=== Manual Review Required: {len(review_items)} items ===")
        for mid, reasons in review_items.items():
            qty = qty_only[mid]
            print(f"  [REVIEW] {mid}: {qty} - {', '.join(reasons)}")

    # Accuracy check
    if args.check:
        from src.accuracy_checker import load_answer, compare_results, print_report
        answer_data = load_answer(Path(args.check))
        report = compare_results(qty_only, answer_data)
        print_report(report)

    # Update JSON
    update_owned_materials(data, qty_only)

    output_path = Path(args.output) if args.output else Path(args.json)
    save_json(data, output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
