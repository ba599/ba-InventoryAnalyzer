# YOLO Backend Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YOLO를 기본 OCR 백엔드로 정식 통합하고, EasyOCR을 선택적 대안으로 유지하며, CLI/GUI 모두에서 백엔드를 전환할 수 있게 한다.

**Architecture:** `CountOcrBackend` ABC를 정의하고, 기존 `OcrReader`와 `YoloOcrReader`가 이를 상속하게 한다. 팩토리 함수로 백엔드를 생성하며, CLI에는 `--backend` 플래그를, GUI에는 YOLO 기본값을 적용한다. YOLO 사용 시 confidence threshold 기본값을 0.9로 올린다.

**Tech Stack:** Python 3.12, onnxruntime, easyocr (optional), PySide6, pytest

---

## File Structure

```
src/
  count_ocr_backend.py     # NEW — ABC + factory function
  ocr_reader.py            # MODIFY — OcrReader inherits CountOcrBackend
  yolo_ocr_reader.py       # MODIFY — YoloOcrReader inherits CountOcrBackend
  core/
    pipeline.py            # MODIFY — type hints OcrReader → CountOcrBackend
    review.py              # (no change)
  main.py                  # MODIFY — --backend flag, factory, threshold logic
  desktop/
    app.py                 # MODIFY — YOLO default, lazy backend init
tests/
  test_count_ocr_backend.py  # NEW — ABC + factory tests
  test_ocr_reader.py         # MODIFY — add backend_name test
  test_yolo_ocr_reader.py    # MODIFY — add backend_name test
requirements.txt             # MODIFY — easyocr optional
```

---

### Task 1: Create CountOcrBackend ABC and factory

**Files:**
- Create: `src/count_ocr_backend.py`
- Create: `tests/test_count_ocr_backend.py`

- [ ] **Step 1: Write failing tests for the ABC and factory**

```python
# tests/test_count_ocr_backend.py
import numpy as np
import pytest

from src.count_ocr_backend import CountOcrBackend, build_backend


class TestCountOcrBackendABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            CountOcrBackend()

    def test_subclass_must_implement_read_quantity(self):
        class Incomplete(CountOcrBackend):
            @property
            def backend_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_backend_name(self):
        class Incomplete(CountOcrBackend):
            def read_quantity(self, image):
                return None

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_works(self):
        class Dummy(CountOcrBackend):
            @property
            def backend_name(self) -> str:
                return "dummy"

            def read_quantity(self, image: np.ndarray):
                return (42, 0.99)

        d = Dummy()
        assert d.backend_name == "dummy"
        assert d.read_quantity(np.zeros((10, 10, 3), dtype=np.uint8)) == (42, 0.99)


class TestBuildBackend:
    def test_build_yolo_backend(self):
        backend = build_backend("yolo")
        assert backend.backend_name == "yolo"

    def test_build_easyocr_backend(self):
        backend = build_backend("easyocr")
        assert backend.backend_name == "easyocr"

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            build_backend("unknown")

    def test_default_is_yolo(self):
        backend = build_backend()
        assert backend.backend_name == "yolo"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_count_ocr_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.count_ocr_backend'`

- [ ] **Step 3: Implement CountOcrBackend ABC and build_backend factory**

```python
# src/count_ocr_backend.py
"""Abstract base class for OCR backends and factory function."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CountOcrBackend(ABC):
    """Common interface for quantity-reading OCR backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for this backend (e.g. 'yolo', 'easyocr')."""

    @abstractmethod
    def read_quantity(self, image: np.ndarray) -> tuple[int, float] | None:
        """Read quantity from a cell's text region image.

        Returns:
            (quantity, confidence) tuple, or None if recognition fails.
        """


def build_backend(backend_type: str = "yolo", **kwargs) -> CountOcrBackend:
    """Create an OCR backend instance by name.

    Args:
        backend_type: 'yolo' or 'easyocr'.
        **kwargs: Forwarded to backend constructor (e.g. model_path for yolo).

    Returns:
        A CountOcrBackend instance.

    Raises:
        ValueError: If backend_type is unknown.
    """
    if backend_type == "yolo":
        from src.yolo_ocr_reader import YoloOcrReader
        return YoloOcrReader(**kwargs)
    elif backend_type == "easyocr":
        from src.ocr_reader import OcrReader
        return OcrReader()
    else:
        raise ValueError(f"Unknown backend: {backend_type!r}. Use 'yolo' or 'easyocr'.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_count_ocr_backend.py -v`
Expected: ABC tests PASS. `TestBuildBackend` tests will still FAIL because `OcrReader` and `YoloOcrReader` don't inherit from `CountOcrBackend` yet — that's expected, they're fixed in Tasks 2-3.

- [ ] **Step 5: Commit**

```bash
git add src/count_ocr_backend.py tests/test_count_ocr_backend.py
git commit -m "feat: add CountOcrBackend ABC and build_backend factory"
```

---

### Task 2: Make OcrReader inherit from CountOcrBackend

**Files:**
- Modify: `src/ocr_reader.py:1-8` (imports + class declaration)
- Modify: `tests/test_ocr_reader.py` (add backend_name test)

- [ ] **Step 1: Add backend_name test to existing test file**

Add this test method to `tests/test_ocr_reader.py`, inside `TestOcrReader`:

```python
def test_backend_name(self):
    assert self.reader.backend_name == "easyocr"
```

Also add an import and isinstance check:

```python
from src.count_ocr_backend import CountOcrBackend

# Add inside TestOcrReader:
def test_is_count_ocr_backend(self):
    assert isinstance(self.reader, CountOcrBackend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ocr_reader.py::TestOcrReader::test_backend_name -v`
Expected: FAIL — `AttributeError: 'OcrReader' object has no attribute 'backend_name'`

- [ ] **Step 3: Modify OcrReader to inherit from CountOcrBackend**

Change the top of `src/ocr_reader.py`:

**Old (lines 1-8):**
```python
import re
import cv2
import numpy as np
import easyocr


class OcrReader:
    def __init__(self):
```

**New:**
```python
import re
import cv2
import numpy as np
import easyocr

from src.count_ocr_backend import CountOcrBackend


class OcrReader(CountOcrBackend):
    @property
    def backend_name(self) -> str:
        return "easyocr"

    def __init__(self):
```

No other changes needed — `read_quantity` already matches the ABC signature.

- [ ] **Step 4: Run all OcrReader tests**

Run: `python -m pytest tests/test_ocr_reader.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/ocr_reader.py tests/test_ocr_reader.py
git commit -m "feat: OcrReader inherits CountOcrBackend"
```

---

### Task 3: Make YoloOcrReader inherit from CountOcrBackend

**Files:**
- Modify: `src/yolo_ocr_reader.py:1-6,124` (imports + class declaration)
- Modify: `tests/test_yolo_ocr_reader.py` (add backend_name test)

- [ ] **Step 1: Add backend_name test**

Add to `tests/test_yolo_ocr_reader.py`:

```python
from src.count_ocr_backend import CountOcrBackend

class TestYoloOcrReaderInterface:
    """Tests that don't require the ONNX model file."""

    def test_is_count_ocr_backend(self):
        assert issubclass(YoloOcrReader, CountOcrBackend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_yolo_ocr_reader.py::TestYoloOcrReaderInterface -v`
Expected: FAIL — `AssertionError`

- [ ] **Step 3: Modify YoloOcrReader to inherit from CountOcrBackend**

**Old (lines 3-4, 124-125):**
```python
from __future__ import annotations
# ... other imports ...

class YoloOcrReader:
    """ONNX YOLO-based digit detector with the same ``read_quantity`` interface as OcrReader."""
```

**New:**
```python
from __future__ import annotations
# ... other imports ...

from src.count_ocr_backend import CountOcrBackend

class YoloOcrReader(CountOcrBackend):
    """ONNX YOLO-based digit detector for inventory quantity reading."""

    @property
    def backend_name(self) -> str:
        return "yolo"
```

Insert `backend_name` property right after the class-level `_lock` attribute (line 129), before `__init__`.

- [ ] **Step 4: Run all YoloOcrReader tests**

Run: `python -m pytest tests/test_yolo_ocr_reader.py -v`
Expected: ALL PASS (model-dependent tests skipped if model not present)

- [ ] **Step 5: Run build_backend tests from Task 1**

Run: `python -m pytest tests/test_count_ocr_backend.py -v`
Expected: ALL PASS — now both backends inherit from ABC, factory tests should pass.

- [ ] **Step 6: Commit**

```bash
git add src/yolo_ocr_reader.py tests/test_yolo_ocr_reader.py
git commit -m "feat: YoloOcrReader inherits CountOcrBackend"
```

---

### Task 4: Update pipeline.py type hints

**Files:**
- Modify: `src/core/pipeline.py:3,7,37-42,89-94`

- [ ] **Step 1: Run existing pipeline tests (baseline)**

Run: `python -m pytest tests/ -k pipeline -v`
Expected: PASS (or identify which tests exist)

- [ ] **Step 2: Update imports and type hints in pipeline.py**

**Old (line 7):**
```python
from src.ocr_reader import OcrReader
```

**New:**
```python
from src.count_ocr_backend import CountOcrBackend
```

**Old `process_single_image` signature (lines 37-42):**
```python
def process_single_image(
    image: np.ndarray,
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: OcrReader,
    min_match_score: float = 0.6,
) -> tuple[dict[str, tuple[int, float]], dict[str, np.ndarray]]:
```

**New:**
```python
def process_single_image(
    image: np.ndarray,
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: CountOcrBackend,
    min_match_score: float = 0.6,
) -> tuple[dict[str, tuple[int, float]], dict[str, np.ndarray]]:
```

**Old `process_all_images` signature (lines 89-94):**
```python
def process_all_images(
    images: list[np.ndarray],
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: OcrReader,
) -> tuple[dict[str, tuple[int, float]], dict[str, np.ndarray]]:
```

**New:**
```python
def process_all_images(
    images: list[np.ndarray],
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: CountOcrBackend,
) -> tuple[dict[str, tuple[int, float]], dict[str, np.ndarray]]:
```

- [ ] **Step 3: Run all tests to verify nothing breaks**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/pipeline.py
git commit -m "refactor: pipeline uses CountOcrBackend type hints"
```

---

### Task 5: Update main.py — add --backend flag and threshold logic

**Files:**
- Modify: `src/main.py:1-9,11-18,85-98`

- [ ] **Step 1: Update imports in main.py**

**Old (lines 5-6):**
```python
from src.ocr_reader import OcrReader
from src.json_updater import load_json, update_owned_materials, save_json
```

**New:**
```python
from src.count_ocr_backend import CountOcrBackend, build_backend
from src.json_updater import load_json, update_owned_materials, save_json
```

- [ ] **Step 2: Update process_screenshots_sequential signature**

**Old (lines 11-18):**
```python
def process_screenshots_sequential(
    images: list,
    item_order: list[str | None],
    reader: OcrReader,
    confidence_threshold: float = 0.7,
    start_id: str | None = None,
    matcher: "ItemMatcher | None" = None,
) -> dict[str, tuple[int, float]]:
```

**New:**
```python
def process_screenshots_sequential(
    images: list,
    item_order: list[str | None],
    reader: CountOcrBackend,
    confidence_threshold: float = 0.7,
    start_id: str | None = None,
    matcher: "ItemMatcher | None" = None,
) -> dict[str, tuple[int, float]]:
```

- [ ] **Step 3: Update main() — add --backend arg, factory call, threshold logic**

**Old (lines 86-98):**
```python
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
```

**New:**
```python
# Default confidence threshold per backend
_DEFAULT_THRESHOLD = {"yolo": 0.9, "easyocr": 0.7}


def main():
    parser = argparse.ArgumentParser(description="Analyze Blue Archive inventory screenshots")
    parser.add_argument("--images", nargs="+", required=True, help="Screenshot image paths")
    parser.add_argument("--json", required=True, help="Justin Planner JSON path")
    parser.add_argument("--order", default="item_order.json", help="Item order JSON path")
    parser.add_argument("--start-id", help="Start material ID (skips template matching, processes images sequentially)")
    parser.add_argument("--refs", default="references", help="Reference images directory (unused with --start-id)")
    parser.add_argument("--output", help="Output JSON path (default: overwrite input)")
    parser.add_argument("--check", help="Answer JSON path for accuracy check")
    parser.add_argument("--backend", default="yolo", choices=["yolo", "easyocr"], help="OCR backend (default: yolo)")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Confidence threshold for [CHECK] flag (default: 0.9 for yolo, 0.7 for easyocr)")
    args = parser.parse_args()

    # Resolve confidence threshold
    if args.confidence_threshold is None:
        args.confidence_threshold = _DEFAULT_THRESHOLD[args.backend]

    item_order = load_item_order(Path(args.order))
    reader = build_backend(args.backend)
    print(f"Backend: {reader.backend_name}")
```

- [ ] **Step 4: Run to verify CLI parses correctly**

Run: `python -m src.main --help`
Expected: Shows `--backend {yolo,easyocr}` option in help text.

- [ ] **Step 5: Commit**

```bash
git add src/main.py
git commit -m "feat: add --backend flag to CLI, YOLO default with 0.9 threshold"
```

---

### Task 6: Update desktop/app.py — YOLO default

**Files:**
- Modify: `src/desktop/app.py:26,282-283,318-322`

- [ ] **Step 1: Update imports**

**Old (line 26):**
```python
from src.ocr_reader import OcrReader
```

**New:**
```python
from src.count_ocr_backend import CountOcrBackend, build_backend
```

- [ ] **Step 2: Update MainWindow instance variable type hint**

**Old (line 282):**
```python
self._reader: OcrReader | None = None
```

**New:**
```python
self._reader: CountOcrBackend | None = None
```

- [ ] **Step 3: Update _get_reader to use factory**

**Old (lines 318-320):**
```python
def _get_reader(self) -> OcrReader:
    if self._reader is None:
        self._reader = OcrReader()
    return self._reader
```

**New:**
```python
def _get_reader(self) -> CountOcrBackend:
    if self._reader is None:
        self._reader = build_backend("yolo")
    return self._reader
```

- [ ] **Step 4: Update find_review_items threshold**

In `_on_analyze_done` (line 363), the `find_review_items` call uses default `threshold=0.7`. Update to `0.9`:

**Old (line 363):**
```python
review_items = find_review_items(results, existing, cell_images)
```

**New:**
```python
review_items = find_review_items(results, existing, cell_images, threshold=0.9)
```

- [ ] **Step 5: Commit**

```bash
git add src/desktop/app.py
git commit -m "feat: desktop app uses YOLO backend by default with 0.9 threshold"
```

---

### Task 7: Make easyocr an optional dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `src/ocr_reader.py:4` (lazy import)

- [ ] **Step 1: Update requirements.txt**

**Old:**
```
opencv-python>=4.8.0
easyocr>=1.7.0
numpy>=1.24.0
pytest>=7.0.0
PySide6>=6.6.0
onnxruntime>=1.16.0
```

**New:**
```
opencv-python>=4.8.0
numpy>=1.24.0
pytest>=7.0.0
PySide6>=6.6.0
onnxruntime>=1.16.0
# Optional: install easyocr to use --backend easyocr
# easyocr>=1.7.0
```

- [ ] **Step 2: Add lazy import guard in OcrReader**

**Old (line 4 of `src/ocr_reader.py`):**
```python
import easyocr
```

**New — move import into `__init__`:**

Remove the top-level `import easyocr` line. Change `__init__`:

**Old:**
```python
    def __init__(self):
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
```

**New:**
```python
    def __init__(self):
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "easyocr is required for the EasyOCR backend. "
                "Install it with: pip install easyocr>=1.7.0"
            )
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
```

Also update the `read_quantity` method — `easyocr` is used only via `self._reader`, so no other changes needed.

- [ ] **Step 3: Run tests to verify YOLO tests still pass without easyocr**

Run: `python -m pytest tests/test_yolo_ocr_reader.py tests/test_count_ocr_backend.py -v -k "not easyocr"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/ocr_reader.py
git commit -m "chore: make easyocr optional dependency, YOLO is default"
```

---

### Task 8: Run full test suite and fix any issues

**Files:**
- All modified files from Tasks 1-7

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Test CLI with both backends**

Run: `python -m src.main --help`
Verify: `--backend {yolo,easyocr}` is shown, default is `yolo`.

- [ ] **Step 3: Fix any failures found**

If any test fails, fix the root cause and re-run.

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: resolve integration test issues"
```
