import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.grid_detector import detect_cells, crop_icon_region, crop_text_region
from src.item_matcher import ItemMatcher
from src.count_ocr_backend import CountOcrBackend


@dataclass
class CellResult:
    """Result from processing a single cell."""
    material_id: str
    quantity: int
    confidence: float
    cell_image: np.ndarray = field(compare=False, repr=False)


def load_item_order(path: Path) -> tuple[list[str | None], dict[str, str]]:
    """Load item order JSON.

    Supports two formats:
    - Legacy flat list: ["100", "101", null, ...]
    - Dict format: [{"100": "이름", "101": "이름", ...}]
      Keys become material IDs (ordered). Null-valued entries are skipped.

    Returns:
        (ordered_ids, name_map)
        ordered_ids: list of material_id strings (None for skip slots)
        name_map: {material_id: display_name} for UI display
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return [], {}

    # Dict format: [{key: name_or_null, ...}]
    if isinstance(data[0], dict):
        item_dict = data[0]
        ordered_ids: list[str | None] = []
        name_map: dict[str, str] = {}
        for key, name in item_dict.items():
            if name is None:
                ordered_ids.append(None)
            else:
                ordered_ids.append(key)
                name_map[key] = name
        return ordered_ids, name_map

    # Legacy flat list: ["100", "101", null, ...]
    name_map = {}
    return data, name_map


def process_image_streaming(
    image: np.ndarray,
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: CountOcrBackend,
    min_match_score: float = 0.6,
    consensus_count: int = 7,
    min_consensus: int = 3,
) -> Iterator[CellResult]:
    """Process a single screenshot, yielding results one cell at a time.

    Phase 1 (matching): Match cells until `consensus_count` trackable items
    are found. Use consensus voting to determine start position and direction.

    Phase 2 (index-walking): Assign remaining cells by walking item_order.
    Stop when a null (dump) slot is encountered.

    Yields:
        CellResult for each successfully processed cell.
    """
    cells = detect_cells(image)
    if not cells:
        return

    # Build lookup: material_id -> index in item_order
    order_index: dict[str, int] = {}
    for idx, mid in enumerate(item_order):
        if mid is not None and mid not in order_index:
            order_index[mid] = idx

    # Phase 1: Match cells to build consensus
    fwd_votes: Counter[int] = Counter()
    rev_votes: Counter[int] = Counter()
    matched_cells: list[tuple[int, str, float]] = []  # (cell_idx, material_id, score)
    trackable_count = 0

    phase1_last_cell_idx = -1
    for cell_idx, cell in enumerate(cells):
        phase1_last_cell_idx = cell_idx
        icon = crop_icon_region(image, cell)
        matched_id, score = matcher.match_with_score(icon)

        if matched_id is None or score < min_match_score:
            continue
        if matched_id not in order_index:
            continue  # untrackable item — skip

        j = order_index[matched_id]
        fwd_votes[j - cell_idx] += 1
        rev_votes[j + cell_idx] += 1
        matched_cells.append((cell_idx, matched_id, score))
        trackable_count += 1

        if trackable_count >= consensus_count:
            break

    # Determine consensus
    fwd_best = fwd_votes.most_common(1)[0] if fwd_votes else (0, 0)
    rev_best = rev_votes.most_common(1)[0] if rev_votes else (0, 0)

    if fwd_best[1] >= rev_best[1]:
        best_offset, count = fwd_best
        reversed_order = False
    else:
        best_offset, count = rev_best
        reversed_order = True

    if count < min_consensus:
        return

    step = -1 if reversed_order else 1

    # Phase 1 results: yield matched cells that align with consensus
    seen: set[str] = set()
    for ci, mid, score in matched_cells:
        if reversed_order:
            expected_offset = order_index[mid] + ci
        else:
            expected_offset = order_index[mid] - ci
        if expected_offset != best_offset:
            continue  # outlier vote — skip
        if mid in seen:
            continue

        text_img = crop_text_region(image, cells[ci])
        result = reader.read_quantity(text_img)
        if result is not None:
            qty, conf = result
            cell = cells[ci]
            cell_crop = image[cell.y:cell.y + cell.h, cell.x:cell.x + cell.w].copy()
            seen.add(mid)
            yield CellResult(mid, qty, conf, cell_crop)

    # Phase 2: Walk item_order for remaining cells (no matching)
    if reversed_order:
        current_order_idx = best_offset - (phase1_last_cell_idx + 1)
    else:
        current_order_idx = best_offset + (phase1_last_cell_idx + 1)

    for ci in range(phase1_last_cell_idx + 1, len(cells)):
        if not (0 <= current_order_idx < len(item_order)):
            break

        mid = item_order[current_order_idx]
        if mid is None:
            break  # dump boundary — stop

        if mid not in seen:
            text_img = crop_text_region(image, cells[ci])
            result = reader.read_quantity(text_img)
            if result is not None:
                qty, conf = result
                cell = cells[ci]
                cell_crop = image[cell.y:cell.y + cell.h, cell.x:cell.x + cell.w].copy()
                seen.add(mid)
                yield CellResult(mid, qty, conf, cell_crop)

        current_order_idx += step


@dataclass
class ImageProgress:
    """Progress marker emitted when starting a new image."""
    current: int  # 1-based
    total: int


def process_all_images_streaming(
    images: list[np.ndarray],
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: CountOcrBackend,
) -> Iterator[ImageProgress | CellResult]:
    """Process multiple images, yielding progress and results as a stream.

    Yields:
        ImageProgress when starting each image.
        CellResult for each processed cell (deduped across images).
    """
    seen: set[str] = set()
    total = len(images)

    for i, image in enumerate(images):
        yield ImageProgress(current=i + 1, total=total)

        for cell_result in process_image_streaming(image, item_order, matcher, reader):
            if cell_result.material_id not in seen:
                seen.add(cell_result.material_id)
                yield cell_result


