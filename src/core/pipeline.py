import json
from pathlib import Path

import numpy as np

from src.grid_detector import CellInfo, detect_cells, crop_icon_region, crop_text_region
from src.item_matcher import ItemMatcher
from src.ocr_reader import OcrReader


def load_item_order(path: Path) -> list[str | None]:
    """Load item order JSON. Supports null entries for items to skip."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_start_cell(
    cells: list[CellInfo],
    image: np.ndarray,
    item_order: list[str | None],
    matcher: ItemMatcher,
    min_score: float = 0.6,
) -> tuple[int, str, int] | None:
    """Scan cells from top-left to find first item matching a reference.

    Returns:
        (cell_index, material_id, item_order_index) or None if no match found.
    """
    for i, cell in enumerate(cells):
        icon = crop_icon_region(image, cell)
        matched_id, score = matcher.match_with_score(icon)
        if matched_id is not None and score >= min_score and matched_id in item_order:
            return i, matched_id, item_order.index(matched_id)
    return None


def process_single_image(
    image: np.ndarray,
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: OcrReader,
    min_match_score: float = 0.6,
) -> tuple[dict[str, tuple[int, float]], dict[str, np.ndarray]]:
    """Process a single screenshot independently.

    Auto-scans to find start position, then processes cells sequentially.

    Returns:
        (results, cell_images)
        results: {material_id: (quantity, confidence)}
        cell_images: {material_id: cropped_cell_image} for review display
    """
    results: dict[str, tuple[int, float]] = {}
    cell_images: dict[str, np.ndarray] = {}

    cells = detect_cells(image)
    if not cells:
        return results, cell_images

    match = find_start_cell(cells, image, item_order, matcher, min_match_score)
    if match is None:
        return results, cell_images

    cell_start, _, current_idx = match

    for cell in cells[cell_start:]:
        if current_idx >= len(item_order):
            break

        material_id = item_order[current_idx]
        current_idx += 1

        if material_id is None:
            continue

        text_img = crop_text_region(image, cell)
        result = reader.read_quantity(text_img)

        if result is not None:
            qty, conf = result
            if material_id not in results:
                results[material_id] = (qty, conf)
                cell_crop = image[cell.y:cell.y + cell.h, cell.x:cell.x + cell.w].copy()
                cell_images[material_id] = cell_crop

    return results, cell_images


def process_all_images(
    images: list[np.ndarray],
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: OcrReader,
) -> tuple[dict[str, tuple[int, float]], dict[str, np.ndarray]]:
    """Process multiple screenshots independently and merge results.

    Each image is auto-scanned for its start position. Duplicate items
    keep the first occurrence.

    Returns:
        (merged_results, merged_cell_images)
    """
    all_results: dict[str, tuple[int, float]] = {}
    all_cell_images: dict[str, np.ndarray] = {}

    for image in images:
        results, cell_images = process_single_image(
            image, item_order, matcher, reader
        )
        for mid, val in results.items():
            if mid not in all_results:
                all_results[mid] = val
                all_cell_images[mid] = cell_images[mid]

    return all_results, all_cell_images
