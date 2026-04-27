# GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add web (Flask) and desktop (PySide6) GUIs to the existing CLI inventory analyzer, sharing a common core pipeline.

**Architecture:** Extract processing logic from `main.py` into `src/core/` (pipeline + review). Web and desktop UIs import core and provide 3-step flow: Input → Review → Result. Each image is processed independently via auto-scan.

**Tech Stack:** Flask (web), PySide6 (desktop), existing OpenCV/EasyOCR/numpy stack.

---

### File Structure

```
src/
  core/
    __init__.py
    pipeline.py      # NEW: process_single_image(), process_all_images()
    review.py         # NEW: find_review_items(), ReviewItem dataclass
  web/
    __init__.py
    app.py            # NEW: Flask routes
    static/
      style.css       # NEW: minimal styling
      app.js          # NEW: paste handling, review flow, copy
    templates/
      index.html      # NEW: single-page 3-step UI
  desktop/
    __init__.py
    app.py            # NEW: PySide6 main window
  main.py             # MODIFY: use core.pipeline
  grid_detector.py    # unchanged
  ocr_reader.py       # unchanged
  item_matcher.py     # unchanged
  json_updater.py     # unchanged
tests/
  test_pipeline.py    # NEW
  test_review.py      # NEW
```

---

### Task 1: Create `src/core/pipeline.py`

**Files:**
- Create: `src/core/__init__.py`
- Create: `src/core/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Create `src/core/__init__.py`**

```python
# empty
```

- [ ] **Step 2: Write failing test for `process_single_image`**

Create `tests/test_pipeline.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.core.pipeline import process_single_image


def _make_cell_info(x, y, w, h, row, col):
    from src.grid_detector import CellInfo
    return CellInfo(x=x, y=y, w=w, h=h, row=row, col=col)


def test_process_single_image_returns_results_and_cell_images():
    """process_single_image returns {mid: (qty, conf)} and {mid: cell_crop}."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101", "102"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
    ]

    # First cell matches item "100" at index 0
    mock_matcher.match_with_score.return_value = ("100", 0.9)

    # OCR returns quantities for each cell
    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (200, 0.80),
        (100, 0.50),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results, cell_images = process_single_image(
            fake_image, item_order, mock_matcher, mock_reader
        )

    assert "100" in results
    assert "101" in results
    assert "102" in results
    assert results["100"] == (500, 0.95)
    assert results["101"] == (200, 0.80)
    assert results["102"] == (100, 0.50)
    # cell_images should have crops for all processed items
    assert "100" in cell_images
    assert "101" in cell_images
    assert "102" in cell_images


def test_process_single_image_no_match_returns_empty():
    """When no cell matches a reference, return empty results."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(0, 0, 80, 80, 0, 0)]
    mock_matcher.match_with_score.return_value = (None, -1.0)

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]):
        results, cell_images = process_single_image(
            fake_image, item_order, mock_matcher, mock_reader
        )

    assert results == {}
    assert cell_images == {}


def test_process_single_image_skips_null_items():
    """Null entries in item_order are skipped."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", None, "102"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
    ]
    mock_matcher.match_with_score.return_value = ("100", 0.9)
    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (999, 0.99),  # null slot - will be read but skipped
        (100, 0.50),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results, cell_images = process_single_image(
            fake_image, item_order, mock_matcher, mock_reader
        )

    assert "100" in results
    assert "102" in results
    assert None not in results
    assert len(results) == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.pipeline'`

- [ ] **Step 4: Implement `process_single_image` and `process_all_images`**

Create `src/core/pipeline.py`:

```python
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
        text_img = crop_text_region(image, cell)
        result = reader.read_quantity(text_img)

        if material_id is None:
            current_idx += 1
            continue

        if result is not None:
            qty, conf = result
            if material_id not in results:
                results[material_id] = (qty, conf)
                # Store full cell crop for review display
                cell_crop = image[cell.y:cell.y + cell.h, cell.x:cell.x + cell.w].copy()
                cell_images[material_id] = cell_crop

        current_idx += 1

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/core/__init__.py src/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: extract core pipeline from main.py"
```

---

### Task 2: Create `src/core/review.py`

**Files:**
- Create: `src/core/review.py`
- Test: `tests/test_review.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_review.py`:

```python
import numpy as np
from src.core.review import find_review_items, ReviewItem


def test_low_confidence_flagged():
    results = {"100": (500, 0.50)}
    existing = {"100": "500"}
    cell_images = {"100": np.zeros((80, 80, 3), dtype=np.uint8)}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert len(items) == 1
    assert items[0].material_id == "100"
    assert items[0].ocr_qty == 500
    assert "low confidence" in items[0].reasons[0]


def test_large_deviation_flagged():
    results = {"100": (500, 0.95)}
    existing = {"100": "1500"}
    cell_images = {"100": np.zeros((80, 80, 3), dtype=np.uint8)}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert len(items) == 1
    assert "deviation" in items[0].reasons[0]


def test_both_reasons_flagged():
    results = {"100": (500, 0.50)}
    existing = {"100": "1500"}
    cell_images = {"100": np.zeros((80, 80, 3), dtype=np.uint8)}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert len(items) == 1
    assert len(items[0].reasons) == 2


def test_no_review_needed():
    results = {"100": (500, 0.95)}
    existing = {"100": "510"}
    cell_images = {"100": np.zeros((80, 80, 3), dtype=np.uint8)}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert len(items) == 0


def test_review_item_has_cell_image():
    results = {"100": (500, 0.50)}
    existing = {"100": "500"}
    cell_img = np.ones((80, 80, 3), dtype=np.uint8) * 128
    cell_images = {"100": cell_img}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert items[0].cell_image is cell_img
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_review.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `review.py`**

Create `src/core/review.py`:

```python
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReviewItem:
    """An OCR result that needs manual review."""
    material_id: str
    ocr_qty: int
    confidence: float
    reasons: list[str]
    cell_image: np.ndarray


def find_review_items(
    results: dict[str, tuple[int, float]],
    existing_materials: dict[str, str],
    cell_images: dict[str, np.ndarray],
    threshold: float = 0.7,
    deviation_limit: int = 100,
) -> list[ReviewItem]:
    """Identify items needing manual review.

    Flags items with:
    - OCR confidence below threshold
    - Quantity change >= deviation_limit from existing value

    Returns:
        List of ReviewItem, one per flagged material.
    """
    items: list[ReviewItem] = []

    for mid, (qty, conf) in results.items():
        reasons: list[str] = []

        if conf < threshold:
            reasons.append(f"low confidence ({conf:.2f})")

        if mid in existing_materials:
            old_val = int(existing_materials[mid]) if existing_materials[mid] else 0
            if abs(qty - old_val) >= deviation_limit:
                reasons.append(f"deviation {qty - old_val:+d} (was {old_val})")

        if reasons:
            items.append(ReviewItem(
                material_id=mid,
                ocr_qty=qty,
                confidence=conf,
                reasons=reasons,
                cell_image=cell_images[mid],
            ))

    return items
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_review.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/review.py tests/test_review.py
git commit -m "feat: add review logic for flagging uncertain OCR results"
```

---

### Task 3: Update `src/main.py` to use core

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Run existing tests to confirm baseline**

Run: `python -m pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 2: Refactor `main.py` to use `core.pipeline` and `core.review`**

Replace `src/main.py` with:

```python
import argparse
import json
from pathlib import Path
import cv2
from src.core.pipeline import (
    load_item_order,
    find_start_cell,
    process_all_images,
    process_single_image,
)
from src.core.review import find_review_items
from src.item_matcher import ItemMatcher
from src.ocr_reader import OcrReader
from src.json_updater import load_json, update_owned_materials, save_json


def process_screenshots_sequential(
    images: list,
    item_order: list[str | None],
    reader: OcrReader,
    confidence_threshold: float = 0.7,
    start_id: str | None = None,
    matcher: "ItemMatcher | None" = None,
) -> dict[str, tuple[int, float]]:
    """Process screenshots sequentially (legacy CLI mode with start_id support).

    Modes:
    - start_id given: first cell of first image = start_id
    - matcher given (no start_id): auto-scan cells to find first known item

    Returns:
        {material_id: (quantity, confidence)}
    """
    from src.grid_detector import detect_cells, crop_icon_region, crop_text_region

    results: dict[str, tuple[int, float]] = {}
    current_idx = None

    for img in images:
        cells = detect_cells(img)
        if not cells:
            print("  Warning: No cells detected, skipping image")
            continue

        print(f"  Detected {len(cells)} cells")
        cell_start = 0

        if current_idx is None:
            if matcher is not None:
                match = find_start_cell(cells, img, item_order, matcher)
                if match is None:
                    print("  Warning: No matching item found, skipping image")
                    continue
                cell_start, matched_id, current_idx = match
                print(f"  Cell {cell_start}: matched {matched_id}")
            elif start_id is not None and start_id in item_order:
                current_idx = item_order.index(start_id)
            else:
                print(f"  Error: start_id '{start_id}' not found in item order")
                return results

        for cell in cells[cell_start:]:
            if current_idx >= len(item_order):
                break

            material_id = item_order[current_idx]
            text_img = crop_text_region(img, cell)
            result = reader.read_quantity(text_img)

            if material_id is None:
                current_idx += 1
                if result is not None:
                    print(f"  [SKIP] position {current_idx - 1}: qty={result[0]}")
                continue

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

            current_idx += 1

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
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Confidence threshold for [CHECK] flag (default: 0.7)")
    args = parser.parse_args()

    item_order = load_item_order(Path(args.order))
    reader = OcrReader()
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
    images = [img for _, img in loaded_images]

    if args.start_id:
        print(f"\nSequential mode: starting from {args.start_id}")
        results = process_screenshots_sequential(
            images, item_order, reader, args.confidence_threshold,
            start_id=args.start_id,
        )
        all_results = results
    else:
        matcher = ItemMatcher(Path(args.refs))
        print(f"\nAuto-scan mode: loaded {len(matcher.references)} references")
        results = process_screenshots_sequential(
            images, item_order, reader, args.confidence_threshold,
            matcher=matcher,
        )
        all_results = results

    # Summary
    qty_only = {k: v[0] for k, v in all_results.items()}
    print(f"\nTotal items recognized: {len(all_results)}")

    # Review items
    existing = data.get("owned_materials", {})
    review_items: dict[str, list[str]] = {}

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
```

- [ ] **Step 3: Run all tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 4: Commit**

```bash
git add src/main.py
git commit -m "refactor: update main.py to import from core.pipeline"
```

---

### Task 4: Web GUI — Flask app

**Files:**
- Create: `src/web/__init__.py`
- Create: `src/web/app.py`
- Create: `src/web/templates/index.html`
- Create: `src/web/static/style.css`
- Create: `src/web/static/app.js`

- [ ] **Step 1: Create `src/web/__init__.py`**

```python
# empty
```

- [ ] **Step 2: Create `src/web/app.py` — Flask routes**

```python
import base64
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

from src.core.pipeline import load_item_order, process_all_images
from src.core.review import find_review_items
from src.item_matcher import ItemMatcher
from src.ocr_reader import OcrReader
from src.json_updater import update_owned_materials

app = Flask(__name__)

# Initialize once (heavy objects)
_reader: OcrReader | None = None
_matcher: ItemMatcher | None = None
_item_order: list[str | None] | None = None


def _get_reader() -> OcrReader:
    global _reader
    if _reader is None:
        _reader = OcrReader()
    return _reader


def _get_matcher() -> ItemMatcher:
    global _matcher
    if _matcher is None:
        _matcher = ItemMatcher(Path("references"))
    return _matcher


def _get_item_order() -> list[str | None]:
    global _item_order
    if _item_order is None:
        _item_order = load_item_order(Path("item_order.json"))
    return _item_order


def _decode_image(data_url: str) -> np.ndarray:
    """Decode a base64 data URL to an OpenCV image."""
    header, b64data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_image(img: np.ndarray) -> str:
    """Encode an OpenCV image to a base64 data URL."""
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Receive images + JSON, run pipeline, return results + review items."""
    payload = request.get_json()
    image_data_urls: list[str] = payload["images"]
    justin_json_text: str = payload["json_text"]

    try:
        justin_data = json.loads(justin_json_text)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    existing_materials = justin_data.get("owned_materials", {})

    # Decode images
    images = []
    for url in image_data_urls:
        img = _decode_image(url)
        if img is not None:
            images.append(img)

    if not images:
        return jsonify({"error": "No valid images"}), 400

    # Process
    item_order = _get_item_order()
    matcher = _get_matcher()
    reader = _get_reader()

    results, cell_images = process_all_images(images, item_order, matcher, reader)

    if not results:
        return jsonify({"error": "No items detected in any image"}), 400

    # Find review items
    review_items = find_review_items(
        results, existing_materials, cell_images, threshold=0.7
    )

    # Build response
    # Separate confirmed vs pending-review results
    review_mids = {item.material_id for item in review_items}
    confirmed = {mid: qty for mid, (qty, _) in results.items() if mid not in review_mids}

    review_list = []
    for item in review_items:
        review_list.append({
            "material_id": item.material_id,
            "ocr_qty": item.ocr_qty,
            "confidence": item.confidence,
            "reasons": item.reasons,
            "cell_image": _encode_image(item.cell_image),
        })

    return jsonify({
        "confirmed": confirmed,
        "review_items": review_list,
        "total_items": len(results),
    })


@app.route("/finalize", methods=["POST"])
def finalize():
    """Apply confirmed + reviewed quantities to the JSON and return full JSON."""
    payload = request.get_json()
    justin_json_text: str = payload["json_text"]
    confirmed: dict[str, int] = payload["confirmed"]
    reviewed: dict[str, int] = payload["reviewed"]

    try:
        justin_data = json.loads(justin_json_text)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    # Merge confirmed + reviewed
    all_updates = {}
    for mid, qty in confirmed.items():
        all_updates[mid] = int(qty)
    for mid, qty in reviewed.items():
        all_updates[mid] = int(qty)

    update_owned_materials(justin_data, all_updates)

    return jsonify({
        "json_text": json.dumps(justin_data, indent=2, ensure_ascii=False),
    })


def run():
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Create `src/web/templates/index.html`**

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blue Archive Inventory Analyzer</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <!-- Step 1: Input -->
    <div id="step-input" class="step active">
        <h1>Blue Archive Inventory Analyzer</h1>

        <div class="section">
            <h2>1. Justin Planner JSON</h2>
            <textarea id="json-input" placeholder="justin163.json 내용을 붙여넣으세요..." rows="6"></textarea>
        </div>

        <div class="section">
            <h2>2. Screenshots (Ctrl+V)</h2>
            <div id="paste-area" tabindex="0">
                Ctrl+V로 스크린샷을 붙여넣으세요
            </div>
            <div id="thumbnails"></div>
        </div>

        <button id="btn-analyze" disabled>분석 시작</button>
        <div id="loading" class="hidden">분석 중...</div>
    </div>

    <!-- Step 2: Review -->
    <div id="step-review" class="step hidden">
        <h1>검수</h1>
        <div id="review-progress"></div>
        <div id="review-cell-image"></div>
        <div id="review-form">
            <div id="review-material-id"></div>
            <input type="text" id="review-input" autocomplete="off" />
            <div id="review-reasons"></div>
            <div class="review-actions">
                Enter: 확정 / <button id="btn-skip">Skip</button>
            </div>
        </div>
    </div>

    <!-- Step 3: Result -->
    <div id="step-result" class="step hidden">
        <h1>결과</h1>
        <textarea id="result-json" rows="20" readonly></textarea>
        <button id="btn-copy">복사</button>
    </div>

    <script src="{{ url_for('static', filename='app.js') }}"></script>
</body>
</html>
```

- [ ] **Step 4: Create `src/web/static/style.css`**

```css
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 640px;
    margin: 0 auto;
    padding: 20px;
    background: #1a1a2e;
    color: #e0e0e0;
}

h1 { margin-bottom: 20px; color: #fff; }
h2 { margin-bottom: 8px; font-size: 14px; color: #aaa; }

.section { margin-bottom: 16px; }

textarea, input[type="text"] {
    width: 100%;
    background: #16213e;
    color: #e0e0e0;
    border: 1px solid #334;
    border-radius: 4px;
    padding: 8px;
    font-family: monospace;
    font-size: 14px;
}

textarea:focus, input[type="text"]:focus {
    outline: none;
    border-color: #4a9eff;
}

#paste-area {
    border: 2px dashed #334;
    border-radius: 8px;
    padding: 40px;
    text-align: center;
    color: #666;
    cursor: pointer;
    margin-bottom: 8px;
}

#paste-area:focus, #paste-area.has-images {
    border-color: #4a9eff;
}

#thumbnails {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

#thumbnails .thumb {
    position: relative;
    width: 80px;
    height: 80px;
}

#thumbnails .thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 4px;
}

#thumbnails .thumb .remove {
    position: absolute;
    top: -6px;
    right: -6px;
    background: #e74c3c;
    color: #fff;
    border: none;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    cursor: pointer;
    font-size: 12px;
    line-height: 20px;
}

button {
    background: #4a9eff;
    color: #fff;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    font-size: 14px;
    cursor: pointer;
    margin-top: 8px;
}

button:disabled { opacity: 0.4; cursor: not-allowed; }
button:hover:not(:disabled) { background: #3a8eef; }

.hidden { display: none !important; }
.step { display: none; }
.step.active { display: block; }

/* Step 2: Review */
#review-progress {
    text-align: center;
    font-size: 18px;
    margin-bottom: 16px;
    color: #aaa;
}

#review-cell-image {
    text-align: center;
    margin-bottom: 16px;
}

#review-cell-image img {
    max-width: 100%;
    max-height: 300px;
    image-rendering: pixelated;
    border-radius: 4px;
    border: 2px solid #4a9eff;
}

#review-form { text-align: center; }

#review-material-id {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 8px;
}

#review-input {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    max-width: 200px;
    margin: 0 auto 8px;
    display: block;
}

#review-reasons {
    color: #e74c3c;
    font-size: 13px;
    margin-bottom: 12px;
}

.review-actions {
    color: #666;
    font-size: 13px;
}

.review-actions button {
    font-size: 13px;
    padding: 4px 12px;
    background: #555;
}

/* Step 3: Result */
#result-json {
    height: 400px;
    resize: vertical;
}

#btn-copy.copied {
    background: #27ae60;
}

#loading {
    text-align: center;
    padding: 20px;
    color: #4a9eff;
}
```

- [ ] **Step 5: Create `src/web/static/app.js`**

```javascript
(function () {
    // State
    let images = [];          // base64 data URLs
    let jsonText = "";
    let confirmed = {};       // {mid: qty}
    let reviewItems = [];     // from server
    let reviewIndex = 0;
    let reviewed = {};        // {mid: qty} user-confirmed values

    // Elements
    const stepInput = document.getElementById("step-input");
    const stepReview = document.getElementById("step-review");
    const stepResult = document.getElementById("step-result");
    const jsonInput = document.getElementById("json-input");
    const pasteArea = document.getElementById("paste-area");
    const thumbnails = document.getElementById("thumbnails");
    const btnAnalyze = document.getElementById("btn-analyze");
    const loading = document.getElementById("loading");
    const reviewProgress = document.getElementById("review-progress");
    const reviewCellImage = document.getElementById("review-cell-image");
    const reviewMaterialId = document.getElementById("review-material-id");
    const reviewInput = document.getElementById("review-input");
    const reviewReasons = document.getElementById("review-reasons");
    const btnSkip = document.getElementById("btn-skip");
    const resultJson = document.getElementById("result-json");
    const btnCopy = document.getElementById("btn-copy");

    // Step navigation
    function showStep(stepEl) {
        document.querySelectorAll(".step").forEach(s => s.classList.remove("active"));
        stepEl.classList.add("active");
    }

    // Update analyze button state
    function updateAnalyzeButton() {
        btnAnalyze.disabled = !(images.length > 0 && jsonInput.value.trim());
    }

    // Paste handler — works on the paste area and globally
    function handlePaste(e) {
        const items = e.clipboardData?.items;
        if (!items) return;

        for (const item of items) {
            if (item.type.startsWith("image/")) {
                e.preventDefault();
                const blob = item.getAsFile();
                const reader = new FileReader();
                reader.onload = function (ev) {
                    images.push(ev.target.result);
                    renderThumbnails();
                    updateAnalyzeButton();
                };
                reader.readAsDataURL(blob);
                return;
            }
        }
    }

    document.addEventListener("paste", handlePaste);
    jsonInput.addEventListener("input", updateAnalyzeButton);

    // Render thumbnails
    function renderThumbnails() {
        thumbnails.innerHTML = "";
        images.forEach((src, idx) => {
            const div = document.createElement("div");
            div.className = "thumb";
            div.innerHTML = `
                <img src="${src}" alt="screenshot ${idx + 1}" />
                <button class="remove" data-idx="${idx}">&times;</button>
            `;
            thumbnails.appendChild(div);
        });

        if (images.length > 0) {
            pasteArea.classList.add("has-images");
            pasteArea.textContent = `${images.length}장 추가됨 (Ctrl+V로 더 추가)`;
        } else {
            pasteArea.classList.remove("has-images");
            pasteArea.textContent = "Ctrl+V로 스크린샷을 붙여넣으세요";
        }
    }

    // Remove thumbnail
    thumbnails.addEventListener("click", function (e) {
        if (e.target.classList.contains("remove")) {
            const idx = parseInt(e.target.dataset.idx);
            images.splice(idx, 1);
            renderThumbnails();
            updateAnalyzeButton();
        }
    });

    // Analyze
    btnAnalyze.addEventListener("click", async function () {
        jsonText = jsonInput.value.trim();
        btnAnalyze.disabled = true;
        loading.classList.remove("hidden");

        try {
            const resp = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    images: images,
                    json_text: jsonText,
                }),
            });

            const data = await resp.json();

            if (!resp.ok) {
                alert(data.error || "Analysis failed");
                return;
            }

            confirmed = data.confirmed;
            reviewItems = data.review_items;
            reviewIndex = 0;
            reviewed = {};

            if (reviewItems.length > 0) {
                showStep(stepReview);
                showReviewItem();
            } else {
                finalize();
            }
        } catch (err) {
            alert("Error: " + err.message);
        } finally {
            loading.classList.add("hidden");
            btnAnalyze.disabled = false;
        }
    });

    // Review flow
    function showReviewItem() {
        const item = reviewItems[reviewIndex];
        reviewProgress.textContent = `${reviewIndex + 1} / ${reviewItems.length}`;
        reviewCellImage.innerHTML = `<img src="${item.cell_image}" />`;
        reviewMaterialId.textContent = item.material_id;
        reviewInput.value = item.ocr_qty;
        reviewReasons.textContent = item.reasons.join(", ");
        reviewInput.focus();
        reviewInput.select();
    }

    function confirmCurrentReview() {
        const item = reviewItems[reviewIndex];
        const val = parseInt(reviewInput.value);
        if (isNaN(val)) {
            reviewInput.focus();
            reviewInput.select();
            return;
        }
        reviewed[item.material_id] = val;
        advanceReview();
    }

    function skipCurrentReview() {
        const item = reviewItems[reviewIndex];
        reviewed[item.material_id] = item.ocr_qty;
        advanceReview();
    }

    function advanceReview() {
        reviewIndex++;
        if (reviewIndex < reviewItems.length) {
            showReviewItem();
        } else {
            finalize();
        }
    }

    reviewInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            e.preventDefault();
            confirmCurrentReview();
        }
    });

    btnSkip.addEventListener("click", skipCurrentReview);

    // Finalize
    async function finalize() {
        try {
            const resp = await fetch("/finalize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    json_text: jsonText,
                    confirmed: confirmed,
                    reviewed: reviewed,
                }),
            });

            const data = await resp.json();

            if (!resp.ok) {
                alert(data.error || "Finalize failed");
                return;
            }

            resultJson.value = data.json_text;
            showStep(stepResult);
        } catch (err) {
            alert("Error: " + err.message);
        }
    }

    // Copy
    btnCopy.addEventListener("click", async function () {
        try {
            await navigator.clipboard.writeText(resultJson.value);
            btnCopy.textContent = "복사 완료!";
            btnCopy.classList.add("copied");
            setTimeout(() => {
                btnCopy.textContent = "복사";
                btnCopy.classList.remove("copied");
            }, 2000);
        } catch {
            resultJson.select();
            document.execCommand("copy");
        }
    });
})();
```

- [ ] **Step 6: Add Flask to requirements**

Append to `requirements.txt`:

```
flask>=3.0.0
```

- [ ] **Step 7: Manual test — run the web app**

Run: `python -m src.web.app`
Expected: Flask dev server starts at http://localhost:5000

- [ ] **Step 8: Commit**

```bash
git add src/web/ requirements.txt
git commit -m "feat: add web GUI with Flask (paste, review, copy flow)"
```

---

### Task 5: Desktop GUI — PySide6 app

**Files:**
- Create: `src/desktop/__init__.py`
- Create: `src/desktop/app.py`

- [ ] **Step 1: Create `src/desktop/__init__.py`**

```python
# empty
```

- [ ] **Step 2: Create `src/desktop/app.py`**

```python
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.pipeline import load_item_order, process_all_images
from src.core.review import ReviewItem, find_review_items
from src.item_matcher import ItemMatcher
from src.json_updater import update_owned_materials
from src.ocr_reader import OcrReader


def _cv2_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert OpenCV BGR image to QPixmap."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class AnalyzeWorker(QThread):
    """Background thread for OCR processing."""
    finished = Signal(dict, dict)  # results, cell_images
    error = Signal(str)

    def __init__(self, images, item_order, matcher, reader):
        super().__init__()
        self.images = images
        self.item_order = item_order
        self.matcher = matcher
        self.reader = reader

    def run(self):
        try:
            results, cell_images = process_all_images(
                self.images, self.item_order, self.matcher, self.reader
            )
            self.finished.emit(results, cell_images)
        except Exception as e:
            self.error.emit(str(e))


class InputPage(QWidget):
    analyze_requested = Signal(str, list)  # json_text, [np.ndarray images]

    def __init__(self):
        super().__init__()
        self._images: list[np.ndarray] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("1. Justin Planner JSON"))
        self.json_input = QTextEdit()
        self.json_input.setPlaceholderText("justin163.json 내용을 붙여넣으세요...")
        self.json_input.setMaximumHeight(150)
        layout.addWidget(self.json_input)

        layout.addWidget(QLabel("2. Screenshots (Ctrl+V)"))
        self.paste_label = QLabel("Ctrl+V로 스크린샷을 붙여넣으세요")
        self.paste_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.paste_label.setStyleSheet(
            "border: 2px dashed #555; border-radius: 8px; padding: 30px; color: #888;"
        )
        layout.addWidget(self.paste_label)

        self.thumb_layout = QHBoxLayout()
        self.thumb_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(self.thumb_layout)

        self.btn_analyze = QPushButton("분석 시작")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._on_analyze)
        layout.addWidget(self.btn_analyze)

        self.loading_label = QLabel("분석 중...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        layout.addWidget(self.loading_label)

        layout.addStretch()

    def keyPressEvent(self, event):
        if event.matches(event.key() == Qt.Key.Key_V and event.modifiers() == Qt.KeyboardModifier.ControlModifier):
            self._paste_image()
        super().keyPressEvent(event)

    def _paste_image(self):
        clipboard = QApplication.clipboard()
        img = clipboard.image()
        if img.isNull():
            return

        # Convert QImage to numpy array
        img = img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        ptr = img.bits()
        arr = np.array(ptr, dtype=np.uint8).reshape(h, w, 3)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self._images.append(bgr)
        self._update_thumbnails()
        self._update_button()

    def _update_thumbnails(self):
        # Clear existing
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, img in enumerate(self._images):
            thumb = QLabel()
            pix = _cv2_to_qpixmap(img)
            thumb.setPixmap(pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio))
            self.thumb_layout.addWidget(thumb)

        count = len(self._images)
        if count > 0:
            self.paste_label.setText(f"{count}장 추가됨 (Ctrl+V로 더 추가)")
        else:
            self.paste_label.setText("Ctrl+V로 스크린샷을 붙여넣으세요")

    def _update_button(self):
        has_json = bool(self.json_input.toPlainText().strip())
        has_images = len(self._images) > 0
        self.btn_analyze.setEnabled(has_json and has_images)

    def _on_analyze(self):
        json_text = self.json_input.toPlainText().strip()
        self.analyze_requested.emit(json_text, self._images.copy())


class ReviewPage(QWidget):
    review_completed = Signal(dict)  # {mid: qty}

    def __init__(self):
        super().__init__()
        self._items: list[ReviewItem] = []
        self._index = 0
        self._reviewed: dict[str, int] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 18px; color: #aaa;")
        layout.addWidget(self.progress_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        self.mid_label = QLabel()
        self.mid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mid_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.mid_label)

        self.qty_input = QLineEdit()
        self.qty_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qty_input.setStyleSheet("font-size: 24px; font-weight: bold; max-width: 200px;")
        self.qty_input.setMaximumWidth(200)
        self.qty_input.returnPressed.connect(self._confirm)
        layout.addWidget(self.qty_input, alignment=Qt.AlignmentFlag.AlignCenter)

        self.reasons_label = QLabel()
        self.reasons_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reasons_label.setStyleSheet("color: #e74c3c; font-size: 13px;")
        layout.addWidget(self.reasons_label)

        actions = QHBoxLayout()
        actions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("Enter: 확정")
        hint.setStyleSheet("color: #666; font-size: 13px;")
        actions.addWidget(hint)
        btn_skip = QPushButton("Skip")
        btn_skip.clicked.connect(self._skip)
        actions.addWidget(btn_skip)
        layout.addLayout(actions)

        layout.addStretch()

    def set_items(self, items: list[ReviewItem]):
        self._items = items
        self._index = 0
        self._reviewed = {}
        self._show_current()

    def _show_current(self):
        item = self._items[self._index]
        self.progress_label.setText(f"{self._index + 1} / {len(self._items)}")

        pix = _cv2_to_qpixmap(item.cell_image)
        scaled = pix.scaled(
            300, 300,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.image_label.setPixmap(scaled)

        self.mid_label.setText(item.material_id)
        self.qty_input.setText(str(item.ocr_qty))
        self.reasons_label.setText(", ".join(item.reasons))
        self.qty_input.setFocus()
        self.qty_input.selectAll()

    def _confirm(self):
        item = self._items[self._index]
        try:
            val = int(self.qty_input.text())
        except ValueError:
            self.qty_input.selectAll()
            return
        self._reviewed[item.material_id] = val
        self._advance()

    def _skip(self):
        item = self._items[self._index]
        self._reviewed[item.material_id] = item.ocr_qty
        self._advance()

    def _advance(self):
        self._index += 1
        if self._index < len(self._items):
            self._show_current()
        else:
            self.review_completed.emit(self._reviewed)


class ResultPage(QWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("결과"))

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.result_text)

        self.btn_copy = QPushButton("복사")
        self.btn_copy.clicked.connect(self._copy)
        layout.addWidget(self.btn_copy)

    def set_result(self, json_text: str):
        self.result_text.setPlainText(json_text)

    def _copy(self):
        QApplication.clipboard().setText(self.result_text.toPlainText())
        self.btn_copy.setText("복사 완료!")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.btn_copy.setText("복사"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blue Archive Inventory Analyzer")
        self.setMinimumSize(500, 600)

        self._json_text = ""
        self._confirmed: dict[str, int] = {}
        self._worker: AnalyzeWorker | None = None

        # Lazy-loaded heavy objects
        self._reader: OcrReader | None = None
        self._matcher: ItemMatcher | None = None
        self._item_order: list[str | None] | None = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.input_page = InputPage()
        self.review_page = ReviewPage()
        self.result_page = ResultPage()

        self.stack.addWidget(self.input_page)
        self.stack.addWidget(self.review_page)
        self.stack.addWidget(self.result_page)

        self.input_page.analyze_requested.connect(self._on_analyze)
        self.review_page.review_completed.connect(self._on_review_done)

    def keyPressEvent(self, event):
        """Handle Ctrl+V globally for image pasting on InputPage."""
        if (
            event.key() == Qt.Key.Key_V
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and self.stack.currentWidget() is self.input_page
        ):
            self.input_page._paste_image()
        else:
            super().keyPressEvent(event)

    def _get_reader(self) -> OcrReader:
        if self._reader is None:
            self._reader = OcrReader()
        return self._reader

    def _get_matcher(self) -> ItemMatcher:
        if self._matcher is None:
            self._matcher = ItemMatcher(Path("references"))
        return self._matcher

    def _get_item_order(self) -> list[str | None]:
        if self._item_order is None:
            self._item_order = load_item_order(Path("item_order.json"))
        return self._item_order

    def _on_analyze(self, json_text: str, images: list[np.ndarray]):
        try:
            self._justin_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Error", f"Invalid JSON: {e}")
            return

        self._json_text = json_text
        self.input_page.loading_label.show()
        self.input_page.btn_analyze.setEnabled(False)

        self._worker = AnalyzeWorker(
            images,
            self._get_item_order(),
            self._get_matcher(),
            self._get_reader(),
        )
        self._worker.finished.connect(self._on_analyze_done)
        self._worker.error.connect(self._on_analyze_error)
        self._worker.start()

    def _on_analyze_done(self, results: dict, cell_images: dict):
        self.input_page.loading_label.hide()

        if not results:
            QMessageBox.warning(self, "Error", "No items detected")
            self.input_page.btn_analyze.setEnabled(True)
            return

        existing = self._justin_data.get("owned_materials", {})
        review_items = find_review_items(results, existing, cell_images)

        review_mids = {item.material_id for item in review_items}
        self._confirmed = {mid: qty for mid, (qty, _) in results.items() if mid not in review_mids}

        if review_items:
            self.review_page.set_items(review_items)
            self.stack.setCurrentWidget(self.review_page)
        else:
            self._finalize({})

    def _on_analyze_error(self, msg: str):
        self.input_page.loading_label.hide()
        self.input_page.btn_analyze.setEnabled(True)
        QMessageBox.warning(self, "Error", msg)

    def _on_review_done(self, reviewed: dict[str, int]):
        self._finalize(reviewed)

    def _finalize(self, reviewed: dict[str, int]):
        all_updates = {}
        for mid, qty in self._confirmed.items():
            all_updates[mid] = qty
        for mid, qty in reviewed.items():
            all_updates[mid] = qty

        update_owned_materials(self._justin_data, all_updates)
        result_text = json.dumps(self._justin_data, indent=2, ensure_ascii=False)
        self.result_page.set_result(result_text)
        self.stack.setCurrentWidget(self.result_page)


def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Add PySide6 to requirements**

Append to `requirements.txt`:

```
PySide6>=6.6.0
```

- [ ] **Step 4: Manual test — run the desktop app**

Run: `python -m src.desktop.app`
Expected: PySide6 window opens with input page

- [ ] **Step 5: Commit**

```bash
git add src/desktop/ requirements.txt
git commit -m "feat: add desktop GUI with PySide6 (paste, review, copy flow)"
```

---

### Task 6: Integration test & polish

**Files:**
- All files created in Tasks 1-5

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Test web app end-to-end**

Run: `python -m src.web.app`
Manual verification:
1. Open http://localhost:5000
2. Paste JSON text → paste screenshot → click "분석 시작"
3. Review items appear one-by-one → Enter/Skip
4. Result JSON shown → "복사" copies to clipboard

- [ ] **Step 3: Test desktop app end-to-end**

Run: `python -m src.desktop.app`
Manual verification:
1. Window opens
2. Paste JSON → Ctrl+V screenshot → click "분석 시작"
3. Review items one-by-one → Enter/Skip
4. Result JSON → "복사" works

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: integration verification complete"
```
