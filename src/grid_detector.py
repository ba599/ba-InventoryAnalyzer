from dataclasses import dataclass
import statistics
import cv2
import numpy as np


@dataclass
class CellInfo:
    """Detected grid cell with position and region info."""
    x: int
    y: int
    w: int
    h: int
    row: int
    col: int


def _deduplicate_rects(
    rects: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.5,
    containment_thresh: float = 0.7,
) -> list[tuple[int, int, int, int]]:
    """Remove duplicate/overlapping rectangles, keeping the largest by area.

    A rectangle is dropped if:
    - Its IoU with an already-kept rect exceeds iou_thresh (near-duplicate), OR
    - More than containment_thresh of its area is covered by an already-kept rect
      (it is mostly contained within a larger cell).
    """
    if not rects:
        return rects
    # Sort by area descending so largest (most complete) rects are kept first
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for x1, y1, w1, h1 in rects:
        is_dup = False
        for kx, ky, kw, kh in kept:
            ix = max(x1, kx)
            iy = max(y1, ky)
            iw = min(x1 + w1, kx + kw) - ix
            ih = min(y1 + h1, ky + kh) - iy
            if iw > 0 and ih > 0:
                inter = iw * ih
                area1 = w1 * h1
                union = area1 + kw * kh - inter
                iou = inter / union if union > 0 else 0
                # Drop near-duplicates or rects mostly inside a larger rect
                containment = inter / area1 if area1 > 0 else 0
                if iou > iou_thresh or containment > containment_thresh:
                    is_dup = True
                    break
        if not is_dup:
            kept.append((x1, y1, w1, h1))
    return kept


def detect_cells(image: np.ndarray, min_cell_ratio: float = 0.6) -> list[CellInfo]:
    """Detect item cells in a grid screenshot.

    Args:
        image: BGR screenshot of the grid area.
        min_cell_ratio: Minimum width/height ratio vs median cell size to keep (filters partial cells).

    Returns:
        List of CellInfo sorted top-left to bottom-right.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to handle varying backgrounds
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10
    )

    # Use RETR_LIST to find all contours (RETR_EXTERNAL misses cells nested inside
    # the overall image border on real screenshots)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding rects and filter by reasonable cell size
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.002  # cell should be at least 0.2% of image
    max_area = img_area * 0.1   # cell should be at most 10% of image

    raw_rects: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / h if h > 0 else 0
        # Cells are roughly square (aspect ratio 0.7 ~ 1.4)
        if min_area < area < max_area and 0.7 < aspect < 1.4:
            raw_rects.append((x, y, w, h))

    if not raw_rects:
        return []

    # Remove duplicate/overlapping contours (real images produce many overlapping
    # contours for the same cell due to nested contour levels)
    deduped_rects = _deduplicate_rects(raw_rects, iou_thresh=0.5)

    if not deduped_rects:
        return []

    # Calculate median cell size to filter partial cells
    widths = [r[2] for r in deduped_rects]
    heights = [r[3] for r in deduped_rects]
    median_w = statistics.median(widths)
    median_h = statistics.median(heights)

    # Filter out partial cells
    full_rects = [
        (x, y, w, h) for x, y, w, h in deduped_rects
        if w >= median_w * min_cell_ratio and h >= median_h * min_cell_ratio
    ]

    if not full_rects:
        return []

    # Assign row/col by clustering Y positions then sorting X within each row
    # Group by Y: cells within half-median-height are on the same row
    y_tolerance = median_h * 0.5
    sorted_by_y = sorted(full_rects, key=lambda r: r[1])

    rows_grouped: list[list[tuple[int, int, int, int]]] = []
    current_row: list[tuple[int, int, int, int]] = [sorted_by_y[0]]

    for rect in sorted_by_y[1:]:
        current_row_median_y = statistics.median([r[1] for r in current_row])
        if abs(rect[1] - current_row_median_y) <= y_tolerance:
            current_row.append(rect)
        else:
            rows_grouped.append(current_row)
            current_row = [rect]
    rows_grouped.append(current_row)

    # Sort each row by X
    for row_rects in rows_grouped:
        row_rects.sort(key=lambda r: r[0])

    # Extrapolate missing cells using detected spacing pattern
    cells = _extrapolate_grid(rows_grouped, median_w, median_h, image.shape[1], min_cell_ratio, image)

    return cells


def _extrapolate_grid(
    rows_grouped: list[list[tuple[int, int, int, int]]],
    cell_w: float,
    cell_h: float,
    img_width: int,
    min_cell_ratio: float,
    image: np.ndarray | None = None,
) -> list[CellInfo]:
    """Fill in missing cells by extrapolating from detected cells' spacing pattern.

    When some cells aren't detected by contour analysis (e.g., borders merge with
    grid edges), we use the regular spacing of detected cells to predict the full grid.
    Extrapolated cells are validated by checking image content variance.
    """
    # Calculate x-step from consecutive detected cells across all rows
    x_steps = []
    for row_rects in rows_grouped:
        for i in range(1, len(row_rects)):
            step = row_rects[i][0] - row_rects[i - 1][0]
            # Only consider reasonable steps (0.8x ~ 2x cell width)
            if cell_w * 0.8 < step < cell_w * 2:
                x_steps.append(step)

    if not x_steps:
        # Can't extrapolate, return detected cells as-is
        cells = []
        for row_idx, row_rects in enumerate(rows_grouped):
            for col_idx, (x, y, w, h) in enumerate(row_rects):
                cells.append(CellInfo(x=x, y=y, w=w, h=h, row=row_idx, col=col_idx))
        return cells

    step_x = int(statistics.median(x_steps))
    cell_w_int = int(cell_w)
    cell_h_int = int(cell_h)

    # Collect detected cell x-positions to distinguish detected vs extrapolated
    detected_xs: set[int] = set()
    for row_rects in rows_grouped:
        for r in row_rects:
            detected_xs.add(r[0])

    all_cells = []
    for row_idx, row_rects in enumerate(rows_grouped):
        if not row_rects:
            continue

        # Use median Y of detected cells in this row
        row_y = int(statistics.median([r[1] for r in row_rects]))

        # Extrapolate leftward from the leftmost detected cell
        leftmost_x = row_rects[0][0]
        row_xs = []
        x = leftmost_x
        while x >= 0:
            row_xs.insert(0, x)
            x -= step_x

        # Extrapolate rightward from the rightmost detected cell
        rightmost_x = row_rects[-1][0]
        x = rightmost_x + step_x
        while x + cell_w_int <= img_width:
            row_xs.append(x)
            x += step_x

        # Also include all detected positions (in case step calculation skipped some)
        for r in row_rects:
            if r[0] not in row_xs:
                row_xs.append(r[0])
        row_xs.sort()

        # Filter: keep detected cells unconditionally; validate extrapolated cells
        for col_idx, x in enumerate(row_xs):
            if x < 0 or x + cell_w_int * min_cell_ratio > img_width:
                continue

            # Detected cells are always valid
            is_detected = any(abs(x - dx) < step_x * 0.3 for dx in detected_xs)

            if not is_detected and image is not None:
                # Validate extrapolated cell: check if region has meaningful content
                cy = max(0, row_y)
                cx = max(0, x)
                crop = image[cy:min(cy + cell_h_int, image.shape[0]),
                             cx:min(cx + cell_w_int, image.shape[1])]
                if crop.size == 0:
                    continue
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                # Empty/background cells have low variance; real items have high variance
                if np.std(gray_crop) < 15:
                    continue

            all_cells.append(CellInfo(
                x=x, y=row_y, w=cell_w_int, h=cell_h_int,
                row=row_idx, col=col_idx,
            ))

    # Re-assign col indices after filtering
    rows_dict: dict[int, list[CellInfo]] = {}
    for cell in all_cells:
        rows_dict.setdefault(cell.row, []).append(cell)

    result = []
    for row_idx in sorted(rows_dict.keys()):
        row_cells = sorted(rows_dict[row_idx], key=lambda c: c.x)
        for col_idx, cell in enumerate(row_cells):
            cell.col = col_idx
            result.append(cell)

    return result


def crop_icon_region(image: np.ndarray, cell: CellInfo) -> np.ndarray:
    """Crop the icon area (top ~78%) from a cell."""
    icon_h = int(cell.h * 0.78)
    return image[cell.y:cell.y + icon_h, cell.x:cell.x + cell.w].copy()


def crop_text_region(image: np.ndarray, cell: CellInfo) -> np.ndarray:
    """Crop the quantity text area (bottom-right of cell).

    Takes bottom ~22% vertically and right ~65% horizontally to avoid
    tier badges (T2, T3, etc.) in the bottom-left of equipment items.
    """
    text_y = cell.y + int(cell.h * 0.78)
    text_x = cell.x + int(cell.w * 0.35)
    return image[text_y:cell.y + cell.h, text_x:cell.x + cell.w].copy()
