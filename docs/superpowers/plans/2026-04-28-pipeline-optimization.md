# Pipeline Optimization & Progressive UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce analysis time by matching only 7 items for consensus, then OCR-only for the rest; show results progressively in the UI.

**Architecture:** Single-pass pipeline function replaces `find_start_cell` + `process_single_image`. Worker emits per-item signals. UI streams review items as they arrive instead of waiting for full completion.

**Tech Stack:** Python, PySide6 (Qt), OpenCV, numpy

---

### Task 1: Single-Pass Pipeline Core — `process_image_streaming`

**Files:**
- Modify: `src/core/pipeline.py`
- Test: `tests/test_pipeline.py`

This task replaces the two-function approach with a single generator that yields results one cell at a time.

- [ ] **Step 1: Write failing test — 7-match consensus then OCR-only**

Add to `tests/test_pipeline.py`:

```python
from src.core.pipeline import process_image_streaming, CellResult


def test_process_image_streaming_matches_only_first_7_trackable():
    """After 7 trackable matches, remaining cells use index walking (no matcher calls)."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = [str(i) for i in range(100, 115)]  # 15 trackable items

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(10)]

    # First 7 cells match items 100-106 (consensus: forward, offset=100 in item_order)
    match_returns = [(str(100 + i), 0.9) for i in range(7)]
    mock_matcher.match_with_score.side_effect = match_returns

    # All 10 cells get OCR
    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(10)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    assert len(results) == 10
    # Matcher called exactly 7 times (not 10)
    assert mock_matcher.match_with_score.call_count == 7
    # Reader called 10 times (all cells get OCR)
    assert mock_reader.read_quantity.call_count == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_process_image_streaming_matches_only_first_7_trackable -v`
Expected: FAIL — `process_image_streaming` and `CellResult` don't exist yet.

- [ ] **Step 3: Write failing test — dump boundary stops processing**

Add to `tests/test_pipeline.py`:

```python
def test_process_image_streaming_stops_at_dump():
    """When item_order hits null (dump), stop yielding results."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    # 5 trackable, then dump (null), then more items
    item_order = ["100", "101", "102", "103", "104", None, "200", "201"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(8)]

    match_returns = [(str(100 + i), 0.9) for i in range(5)]
    # After 5 matches we don't have 7, but we've run out of trackable items before dump
    # In this case, consensus from 5 matches (>=3) should still work
    mock_matcher.match_with_score.side_effect = match_returns

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(5)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    # Only 5 results — stopped at dump
    assert len(results) == 5
    result_mids = [r.material_id for r in results]
    assert "200" not in result_mids
    assert "201" not in result_mids
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_process_image_streaming_stops_at_dump -v`
Expected: FAIL

- [ ] **Step 5: Write failing test — leading untrackable items are skipped**

Add to `tests/test_pipeline.py`:

```python
def test_process_image_streaming_skips_leading_untrackable():
    """Cells with icons not in item_order are skipped during matching phase."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = [None, None, None, "100", "101", "102", "103", "104", "105", "106"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    # 10 cells: first 3 are untrackable, then 7 trackable
    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(10)]

    # First 3 cells match unknown IDs, next 7 match trackable items
    match_returns = [
        ("UNKNOWN_1", 0.9), ("UNKNOWN_2", 0.9), ("UNKNOWN_3", 0.9),
        ("100", 0.9), ("101", 0.9), ("102", 0.9), ("103", 0.9),
        ("104", 0.9), ("105", 0.9), ("106", 0.9),
    ]
    mock_matcher.match_with_score.side_effect = match_returns

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(7)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    # 7 trackable results, not 10
    assert len(results) == 7
    result_mids = [r.material_id for r in results]
    assert "UNKNOWN_1" not in result_mids
    assert "100" in result_mids
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_process_image_streaming_skips_leading_untrackable -v`
Expected: FAIL

- [ ] **Step 7: Implement `CellResult` dataclass and `process_image_streaming` generator**

In `src/core/pipeline.py`, add after imports:

```python
from dataclasses import dataclass


@dataclass
class CellResult:
    """Result from processing a single cell."""
    material_id: str
    quantity: int
    confidence: float
    cell_image: np.ndarray
```

Add the new function (keep existing functions for now — they'll be updated in Task 3):

```python
def process_image_streaming(
    image: np.ndarray,
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: CountOcrBackend,
    min_match_score: float = 0.6,
    consensus_count: int = 7,
    min_consensus: int = 3,
):
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

    cell_idx = 0
    for cell_idx, cell in enumerate(cells):
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
    # Determine the item_order index for the cell after the last matched cell
    last_matched_cell_idx = cell_idx
    if reversed_order:
        current_order_idx = best_offset - (last_matched_cell_idx + 1)
    else:
        current_order_idx = best_offset + (last_matched_cell_idx + 1)

    for ci in range(last_matched_cell_idx + 1, len(cells)):
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
```

- [ ] **Step 8: Run all new tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v -k "streaming"`
Expected: All 3 new tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: add single-pass streaming pipeline with 7-match consensus and dump boundary"
```

---

### Task 2: `process_all_images_streaming` Wrapper

**Files:**
- Modify: `src/core/pipeline.py`
- Test: `tests/test_pipeline.py`

A generator that iterates over multiple images and yields `(image_index, total_images, CellResult)`.

- [ ] **Step 1: Write failing test**

Add to `tests/test_pipeline.py`:

```python
from src.core.pipeline import process_all_images_streaming, ImageProgress


def test_process_all_images_streaming_yields_progress_and_results():
    """Iterating multiple images yields progress markers and cell results."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101", "102"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
    ]

    match_returns = [(str(100 + i), 0.9) for i in range(3)]
    mock_matcher.match_with_score.side_effect = match_returns * 2  # 2 images

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(6)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        events = list(process_all_images_streaming(
            [fake_image, fake_image], item_order, mock_matcher, mock_reader
        ))

    # Filter progress events
    progress_events = [e for e in events if isinstance(e, ImageProgress)]
    cell_events = [e for e in events if isinstance(e, CellResult)]

    assert len(progress_events) == 2
    assert progress_events[0].current == 1
    assert progress_events[0].total == 2
    assert progress_events[1].current == 2

    # Second image should not re-yield already-seen material_ids
    assert len(cell_events) == 3  # only from first image (second has same items)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_process_all_images_streaming_yields_progress_and_results -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Add to `src/core/pipeline.py`:

```python
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
):
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_pipeline.py -v -k "streaming"`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: add multi-image streaming pipeline with deduplication"
```

---

### Task 3: Update Existing Functions and Exports

**Files:**
- Modify: `src/core/pipeline.py`
- Test: `tests/test_pipeline.py`

Keep `process_all_images` working (used by existing code) by reimplementing it on top of the streaming version.

- [ ] **Step 1: Update `process_all_images` to use streaming internally**

In `src/core/pipeline.py`, replace the existing `process_all_images` function body:

```python
def process_all_images(
    images: list[np.ndarray],
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: CountOcrBackend,
) -> tuple[dict[str, tuple[int, float]], dict[str, np.ndarray]]:
    """Process multiple screenshots and merge results.

    Wraps the streaming pipeline for backward compatibility.
    """
    all_results: dict[str, tuple[int, float]] = {}
    all_cell_images: dict[str, np.ndarray] = {}

    for event in process_all_images_streaming(images, item_order, matcher, reader):
        if isinstance(event, CellResult):
            all_results[event.material_id] = (event.quantity, event.confidence)
            all_cell_images[event.material_id] = event.cell_image

    return all_results, all_cell_images
```

Also remove `process_single_image` and `find_start_cell` — they are superseded. Update the test file to remove or adapt tests that reference them.

- [ ] **Step 2: Update existing tests**

In `tests/test_pipeline.py`, update the import and existing tests to use the new API. The existing `test_process_single_image_*` tests should be rewritten to test `process_image_streaming` instead:

```python
# Replace the old import:
# from src.core.pipeline import process_single_image
# With:
from src.core.pipeline import process_image_streaming, process_all_images_streaming, CellResult, ImageProgress
```

Update `test_process_single_image_returns_results_and_cell_images`:

```python
def test_process_image_streaming_returns_results():
    """process_image_streaming yields CellResult for each matched cell."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101", "102"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
    ]

    match_returns = [("100", 0.9), ("101", 0.9), ("102", 0.9)]
    mock_matcher.match_with_score.side_effect = match_returns

    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (200, 0.80),
        (100, 0.50),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    mids = {r.material_id for r in results}
    assert "100" in mids
    assert "101" in mids
    assert "102" in mids
    r100 = next(r for r in results if r.material_id == "100")
    assert r100.quantity == 500
    assert r100.confidence == 0.95
```

Update `test_process_single_image_no_match_returns_empty`:

```python
def test_process_image_streaming_no_match_returns_empty():
    """When no cell matches a reference, yield nothing."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(0, 0, 80, 80, 0, 0)]
    mock_matcher.match_with_score.return_value = (None, -1.0)

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    assert results == []
```

Update `test_process_single_image_skips_null_items`:

```python
def test_process_image_streaming_skips_null_items():
    """Null entries in item_order are skipped."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", None, "102", "103", "104"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
        _make_cell_info(240, 0, 80, 80, 0, 3),
    ]

    match_returns = [("100", 0.9), ("100", 0.9), ("102", 0.9), ("103", 0.9)]
    mock_matcher.match_with_score.side_effect = match_returns

    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (200, 0.80),
        (300, 0.90),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    mids = [r.material_id for r in results]
    assert "100" in mids
    assert "102" in mids
    assert None not in mids
```

- [ ] **Step 3: Run all pipeline tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/core/pipeline.py tests/test_pipeline.py
git commit -m "refactor: replace legacy pipeline functions with streaming implementation"
```

---

### Task 4: Progressive UI — AnalyzeWorker Signals

**Files:**
- Modify: `src/desktop/app.py`

Update `AnalyzeWorker` to emit streaming signals instead of one bulk `finished` signal.

- [ ] **Step 1: Update AnalyzeWorker**

In `src/desktop/app.py`, replace the `AnalyzeWorker` class:

```python
from src.core.pipeline import (
    load_item_order,
    process_all_images_streaming,
    CellResult,
    ImageProgress,
)


class AnalyzeWorker(QThread):
    """Background thread for streaming OCR processing."""
    progress = Signal(int, int)           # (current_image, total_images)
    item_ready = Signal(str, int, float, object)  # (material_id, qty, confidence, cell_image)
    finished = Signal()
    error = Signal(str)

    def __init__(self, images, item_order, matcher, reader):
        super().__init__()
        self.images = images
        self.item_order = item_order
        self.matcher = matcher
        self.reader = reader

    def run(self):
        try:
            for event in process_all_images_streaming(
                self.images, self.item_order, self.matcher, self.reader
            ):
                if isinstance(event, ImageProgress):
                    self.progress.emit(event.current, event.total)
                elif isinstance(event, CellResult):
                    self.item_ready.emit(
                        event.material_id,
                        event.quantity,
                        event.confidence,
                        event.cell_image,
                    )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.desktop.app import AnalyzeWorker; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/desktop/app.py
git commit -m "feat: update AnalyzeWorker to emit streaming signals"
```

---

### Task 5: Progressive UI — Analysis Page Layout

**Files:**
- Modify: `src/desktop/app.py`

Add progress bar and placeholder to the UI. Repurpose the flow so analysis results stream in.

- [ ] **Step 1: Add QProgressBar import and update InputPage**

At the top of `src/desktop/app.py`, add `QProgressBar` to the import:

```python
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
```

- [ ] **Step 2: Create AnalyzingPage widget**

Add a new page class after `InputPage` in `src/desktop/app.py`:

```python
class AnalyzingPage(QWidget):
    """Shows progress bar during analysis, with placeholder for review items."""

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Top: progress bar
        self.progress_label = QLabel("분석 중 0/0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Bottom: placeholder / review area
        self.placeholder = QLabel("분석 중입니다")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet(
            "font-size: 18px; color: #888; padding: 40px;"
        )
        self.placeholder.setMinimumHeight(300)
        layout.addWidget(self.placeholder)

        layout.addStretch()

    def update_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"분석 중 {current}/{total}")

    def reset(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(1)
        self.progress_label.setText("분석 중 0/0")
        self.placeholder.show()
```

- [ ] **Step 3: Commit**

```bash
git add src/desktop/app.py
git commit -m "feat: add AnalyzingPage with progress bar and placeholder"
```

---

### Task 6: Progressive UI — Wire Up MainWindow

**Files:**
- Modify: `src/desktop/app.py`

Connect the streaming signals to the new UI flow.

- [ ] **Step 1: Update MainWindow to use AnalyzingPage and streaming flow**

In `MainWindow.__init__`, add the analyzing page to the stack:

```python
self.analyzing_page = AnalyzingPage()

# Insert after input_page, before review_page in the stack
self.stack.addWidget(self.input_page)    # index 0
self.stack.addWidget(self.analyzing_page) # index 1
self.stack.addWidget(self.review_page)    # index 2
self.stack.addWidget(self.result_page)    # index 3
```

Add instance variables for accumulating streaming results:

```python
self._pending_review: list[ReviewItem] = []
self._reviewing = False
```

- [ ] **Step 2: Update `_on_analyze` to show AnalyzingPage**

Replace the method:

```python
def _on_analyze(self, json_text: str, images: list[np.ndarray]):
    try:
        self._justin_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        QMessageBox.warning(self, "Error", f"Invalid JSON: {e}")
        return

    self._json_text = json_text
    self._confirmed = {}
    self._pending_review = []
    self._reviewing = False

    self.analyzing_page.reset()
    self.stack.setCurrentWidget(self.analyzing_page)

    self._worker = AnalyzeWorker(
        images,
        self._get_item_order(),
        self._get_matcher(),
        self._get_reader(),
    )
    self._worker.progress.connect(self._on_progress)
    self._worker.item_ready.connect(self._on_item_ready)
    self._worker.finished.connect(self._on_analyze_done)
    self._worker.error.connect(self._on_analyze_error)
    self._worker.start()
```

- [ ] **Step 3: Add streaming signal handlers**

Add these methods to `MainWindow`:

```python
def _on_progress(self, current: int, total: int):
    self.analyzing_page.update_progress(current, total)

def _on_item_ready(self, material_id: str, qty: int, confidence: float, cell_image):
    existing = self._justin_data.get("owned_materials", {})

    # Check if this item needs review (same logic as find_review_items)
    reasons: list[str] = []
    if confidence < 0.9:
        reasons.append(f"low confidence ({confidence:.2f})")
    if material_id in existing:
        old_val = int(existing[material_id]) if existing[material_id] else 0
        if abs(qty - old_val) >= 100:
            reasons.append(f"deviation {qty - old_val:+d} (was {old_val})")

    if reasons:
        item = ReviewItem(
            material_id=material_id,
            ocr_qty=qty,
            confidence=confidence,
            reasons=reasons,
            cell_image=cell_image,
        )
        self._pending_review.append(item)
        self._try_show_review()
    else:
        self._confirmed[material_id] = qty
```

- [ ] **Step 4: Add review queue management**

```python
def _try_show_review(self):
    """Show next review item if not already reviewing."""
    if self._reviewing or not self._pending_review:
        return

    self._reviewing = True
    # Show first pending item — ReviewPage handles one-at-a-time display
    self.review_page.set_items(self._pending_review, self._name_map)
    self.analyzing_page.placeholder.hide()
    # Embed review page content area — but we keep analyzing page visible for progress bar
    # Simpler approach: switch to review page (progress bar is less important than review)
    self.stack.setCurrentWidget(self.review_page)
```

- [ ] **Step 5: Update `_on_analyze_done` and `_on_review_done`**

```python
def _on_analyze_done(self):
    if not self._confirmed and not self._pending_review:
        QMessageBox.warning(self, "Error", "No items detected")
        self.input_page.btn_analyze.setEnabled(True)
        self.stack.setCurrentWidget(self.input_page)
        return

    if not self._reviewing and not self._pending_review:
        # No review needed — go straight to results
        self._finalize({})
    elif not self._reviewing and self._pending_review:
        # Analysis done, pending reviews not yet shown
        self._try_show_review()
    # else: reviewing in progress — _on_review_done will handle finalization

def _on_review_done(self, reviewed: dict[str, int]):
    self._reviewing = False
    self._pending_review = []

    if self._worker is not None and self._worker.isRunning():
        # Analysis still running — go back to analyzing page, wait for more items
        self.stack.setCurrentWidget(self.analyzing_page)
        self.analyzing_page.placeholder.show()
        self.analyzing_page.placeholder.setText("분석 중입니다")
    else:
        # Analysis done — finalize
        self._finalize(reviewed)

def _on_analyze_error(self, msg: str):
    self.input_page.btn_analyze.setEnabled(True)
    self.stack.setCurrentWidget(self.input_page)
    QMessageBox.warning(self, "Error", msg)
```

- [ ] **Step 6: Remove old `loading_label` usage from InputPage**

In `InputPage._init_ui`, remove the `self.loading_label` creation and the `layout.addWidget(self.loading_label)` line. Also remove `self.input_page.loading_label.show()` and `self.input_page.loading_label.hide()` from MainWindow (these are now handled by AnalyzingPage).

- [ ] **Step 7: Update ResultPage text to be selectable**

The existing `QTextEdit` with `setReadOnly(True)` already supports text selection, Ctrl+A, and drag. No code change needed — just verify this works during manual testing.

- [ ] **Step 8: Verify the app starts without errors**

Run: `python -c "from src.desktop.app import MainWindow; print('OK')"`
Expected: OK

- [ ] **Step 9: Commit**

```bash
git add src/desktop/app.py
git commit -m "feat: progressive UI with streaming analysis and inline review"
```

---

### Task 7: Remove Dead Code and Update Imports

**Files:**
- Modify: `src/core/pipeline.py`
- Modify: `src/desktop/app.py`

- [ ] **Step 1: Remove `find_start_cell` and `process_single_image` from pipeline.py**

These functions are fully superseded by `process_image_streaming`. Remove them.

Also update the `process_all_images` function if it still references old functions (it was already updated in Task 3, but verify).

- [ ] **Step 2: Clean up unused imports in `src/desktop/app.py`**

Remove the import of `process_all_images` if no longer used directly (AnalyzeWorker now uses `process_all_images_streaming`).

Also remove the import of `find_review_items` from `src/core/review` — review logic is now inline in `_on_item_ready`.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/core/pipeline.py src/desktop/app.py
git commit -m "refactor: remove superseded pipeline functions and unused imports"
```
