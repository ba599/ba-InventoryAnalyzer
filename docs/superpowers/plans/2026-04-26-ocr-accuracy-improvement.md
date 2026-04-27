# OCR 정확도 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 장비 아이템 OCR 정확도를 81%→90%+로 개선하고, 저신뢰 결과를 플래그하며, answer.json 기반 자동 검증을 추가한다.

**Architecture:** ocr_reader.py의 전처리를 색상 마스크 기반으로 강화하고, confidence를 반환 타입에 추가. main.py에서 confidence 표시 및 [CHECK] 플래그. accuracy_checker.py로 answer.json 대비 자동 비교.

**Tech Stack:** Python 3.13, OpenCV (cv2), EasyOCR/PaddleOCR, numpy, pytest

---

## File Structure

| 파일 | 역할 | 변경 |
|------|------|------|
| `src/ocr_reader.py` | OCR 전처리 + 수량 읽기 | 수정: 전처리 강화, confidence 반환 |
| `src/main.py` | CLI 엔트리포인트 | 수정: confidence 표시, --check, --confidence-threshold |
| `src/accuracy_checker.py` | 정답 비교 | 신규 |
| `tests/test_ocr_reader.py` | OcrReader 테스트 | 수정: 새 반환 타입 |
| `tests/test_accuracy_checker.py` | accuracy_checker 테스트 | 신규 |

---

### Task 1: ocr_reader.py 전처리 강화 + confidence 반환

**Files:**
- Modify: `src/ocr_reader.py`
- Modify: `tests/test_ocr_reader.py`

- [ ] **Step 1: 기존 테스트가 통과하는지 확인**

Run: `.venv313/Scripts/python.exe -m pytest tests/test_ocr_reader.py -v`
Expected: 4 tests PASS

- [ ] **Step 2: test_ocr_reader.py에 새 반환 타입 테스트 추가**

`tests/test_ocr_reader.py`에 다음 테스트를 추가:

```python
def test_read_quantity_returns_tuple_with_confidence(self):
    """read_quantity should return (qty, confidence) tuple or None."""
    img = np.zeros((40, 100, 3), dtype=np.uint8)
    cv2.putText(img, "x500", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    result = self.reader.read_quantity(img)
    if result is not None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        qty, conf = result
        assert isinstance(qty, int)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv313/Scripts/python.exe -m pytest tests/test_ocr_reader.py::TestOcrReader::test_read_quantity_returns_tuple_with_confidence -v`
Expected: FAIL (현재 read_quantity는 int | None을 반환)

- [ ] **Step 4: preprocess() 강화 및 read_quantity() 반환 타입 변경**

`src/ocr_reader.py`를 다음과 같이 수정:

```python
import re
import cv2
import numpy as np

# Try PaddleOCR first, fall back to EasyOCR
try:
    from paddleocr import PaddleOCR
    _OCR_ENGINE = "paddle"
except ImportError:
    import easyocr
    _OCR_ENGINE = "easyocr"

# Font color in equipment item text regions (BGR)
_FONT_COLOR_BGR = np.array([99, 70, 45], dtype=np.uint8)
_FONT_TOLERANCE = 30


class OcrReader:
    def __init__(self):
        if _OCR_ENGINE == "paddle":
            self._reader = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        else:
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def parse_quantity(self, text: str) -> int | None:
        """Parse 'x1234' format text into an integer."""
        text = text.strip()
        match = re.search(r"[xX](\d+)", text)
        if match:
            return int(match.group(1))
        # Try pure digits as fallback
        match = re.match(r"(\d+)$", text)
        if match:
            return int(match.group(1))
        return None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess text region for better OCR accuracy.

        Uses color-based text mask targeting font color #2D4663,
        with Otsu fallback if color mask captures too few pixels.
        """
        # Color mask: extract pixels near #2D4663 (BGR: 99, 70, 45)
        lower = np.clip(_FONT_COLOR_BGR.astype(int) - _FONT_TOLERANCE, 0, 255).astype(np.uint8)
        upper = np.clip(_FONT_COLOR_BGR.astype(int) + _FONT_TOLERANCE, 0, 255).astype(np.uint8)
        mask = cv2.inRange(image, lower, upper)

        # Check if mask captured enough text pixels (at least 2% of image)
        mask_ratio = np.count_nonzero(mask) / mask.size
        if mask_ratio >= 0.02:
            # Invert: font pixels become black (0) on white background (255)
            processed = cv2.bitwise_not(mask)
        else:
            # Fallback: grayscale + Otsu
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Upscale if too small
        if processed.shape[0] < 80:
            scale = 80 / processed.shape[0]
            processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return processed

    def read_quantity(self, image: np.ndarray) -> tuple[int, float] | None:
        """Read quantity from a cell's text region image.

        Returns:
            (quantity, confidence) tuple, or None if OCR fails.
        """
        processed = self.preprocess(image)

        if _OCR_ENGINE == "paddle":
            result = self._reader.ocr(processed, cls=False)
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    conf = float(line[1][1])
                    qty = self.parse_quantity(text)
                    if qty is not None:
                        return (qty, conf)
        else:
            results = self._reader.readtext(processed, allowlist="xX0123456789")
            for bbox, text, conf in results:
                qty = self.parse_quantity(text)
                if qty is not None:
                    return (qty, float(conf))

        return None
```

- [ ] **Step 5: 기존 테스트 수정 — read_from_synthetic_image의 타입 체크**

`tests/test_ocr_reader.py`의 `test_read_from_synthetic_image`를 수정:

```python
def test_read_from_synthetic_image(self):
    # Create an image with clear "x500" text
    img = np.zeros((40, 100, 3), dtype=np.uint8)
    cv2.putText(img, "x500", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    result = self.reader.read_quantity(img)
    # OCR on synthetic may not be perfect; just verify it returns a tuple or None
    if result is not None:
        qty, conf = result
        assert isinstance(qty, int)
        assert isinstance(conf, float)
```

- [ ] **Step 6: 모든 테스트 통과 확인**

Run: `.venv313/Scripts/python.exe -m pytest tests/test_ocr_reader.py -v`
Expected: 5 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/ocr_reader.py tests/test_ocr_reader.py
git commit -m "feat: enhance OCR preprocessing with color mask and confidence return"
```

---

### Task 2: accuracy_checker.py 신규 작성

**Files:**
- Create: `src/accuracy_checker.py`
- Create: `tests/test_accuracy_checker.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_accuracy_checker.py`:

```python
import pytest
from src.accuracy_checker import compare_results


class TestAccuracyChecker:
    def test_all_correct(self):
        ocr = {"100": 2375, "101": 1475}
        answer = {"owned_materials": {"100": "2375", "101": "1475"}}
        result = compare_results(ocr, answer)
        assert result["total"] == 2
        assert result["correct"] == 2
        assert result["errors"] == []

    def test_some_errors(self):
        ocr = {"100": 2375, "101": 9999}
        answer = {"owned_materials": {"100": "2375", "101": "1475"}}
        result = compare_results(ocr, answer)
        assert result["total"] == 2
        assert result["correct"] == 1
        assert len(result["errors"]) == 1
        err = result["errors"][0]
        assert err["id"] == "101"
        assert err["ocr"] == 9999
        assert err["expected"] == 1475

    def test_missing_in_answer(self):
        """OCR found an item not in answer — it is skipped (not counted as error)."""
        ocr = {"100": 2375, "UNKNOWN": 999}
        answer = {"owned_materials": {"100": "2375"}}
        result = compare_results(ocr, answer)
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["skipped"] == 1

    def test_accuracy_percentage(self):
        ocr = {"100": 2375, "101": 9999, "102": 701}
        answer = {"owned_materials": {"100": "2375", "101": "1475", "102": "701"}}
        result = compare_results(ocr, answer)
        assert result["accuracy"] == pytest.approx(66.67, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv313/Scripts/python.exe -m pytest tests/test_accuracy_checker.py -v`
Expected: FAIL (모듈 없음)

- [ ] **Step 3: 구현**

`src/accuracy_checker.py`:

```python
import json
from pathlib import Path


def load_answer(path: Path) -> dict:
    """Load answer.json ground truth file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_results(ocr_results: dict[str, int], answer_data: dict) -> dict:
    """Compare OCR results against answer ground truth.

    Args:
        ocr_results: {material_id: quantity} from OCR.
        answer_data: Full answer.json data with "owned_materials" key.

    Returns:
        Dict with keys: total, correct, errors (list of dicts), skipped, accuracy.
    """
    answer = answer_data.get("owned_materials", {})
    correct = 0
    errors = []
    skipped = 0

    for material_id, ocr_qty in ocr_results.items():
        if material_id not in answer:
            skipped += 1
            continue
        expected = int(answer[material_id])
        if ocr_qty == expected:
            correct += 1
        else:
            errors.append({
                "id": material_id,
                "ocr": ocr_qty,
                "expected": expected,
            })

    total = correct + len(errors)
    accuracy = (correct / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "errors": errors,
        "skipped": skipped,
        "accuracy": round(accuracy, 2),
    }


def print_report(result: dict) -> None:
    """Print a human-readable accuracy report."""
    print(f"\n=== Accuracy Report ===")
    print(f"Total: {result['total']}, Correct: {result['correct']}, "
          f"Errors: {len(result['errors'])}, Skipped: {result['skipped']}")
    print(f"Accuracy: {result['accuracy']}%")

    if result["errors"]:
        print(f"\nErrors:")
        for err in result["errors"]:
            print(f"  {err['id']}: OCR={err['ocr']} Expected={err['expected']}")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv313/Scripts/python.exe -m pytest tests/test_accuracy_checker.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/accuracy_checker.py tests/test_accuracy_checker.py
git commit -m "feat: add accuracy checker for answer.json comparison"
```

---

### Task 3: main.py에 confidence 표시 + --check 플래그 통합

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: process_screenshots_sequential() 수정**

`read_quantity()`가 이제 `tuple[int, float] | None`을 반환하므로 이에 맞게 수정:

```python
def process_screenshots_sequential(
    images: list,
    item_order: list[str | None],
    start_id: str,
    reader: OcrReader,
    confidence_threshold: float = 0.7,
) -> dict[str, tuple[int, float]]:
    """Process screenshots sequentially from a known start_id.

    Returns:
        {material_id: (quantity, confidence)}
    """
    if start_id not in item_order:
        print(f"  Error: start_id '{start_id}' not found in item order")
        return {}

    start_idx = item_order.index(start_id)
    current_idx = start_idx
    results: dict[str, tuple[int, float]] = {}

    for img in images:
        cells = detect_cells(img)
        if not cells:
            print("  Warning: No cells detected, skipping image")
            continue

        print(f"  Detected {len(cells)} cells")

        for cell in cells:
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
```

- [ ] **Step 2: process_screenshot_matched() 수정**

```python
def process_screenshot_matched(
    image,
    item_order: list[str | None],
    matcher: ItemMatcher,
    reader: OcrReader,
    confidence_threshold: float = 0.7,
) -> dict[str, tuple[int, float]]:
    """Process a single screenshot using template matching for first item."""
    cells = detect_cells(image)
    if not cells:
        print("  Warning: No cells detected")
        return {}

    first_icon = crop_icon_region(image, cells[0])
    start_id = matcher.match(first_icon)
    if start_id is None:
        print("  Warning: Could not match first item")
        return {}

    if start_id not in item_order:
        print(f"  Warning: Matched ID '{start_id}' not found in item order")
        return {}

    start_idx = item_order.index(start_id)
    print(f"  First item: {start_id} (index {start_idx})")

    results = {}
    for i, cell in enumerate(cells):
        idx = start_idx + i
        if idx >= len(item_order):
            break

        material_id = item_order[idx]
        if material_id is None:
            continue

        text_img = crop_text_region(image, cell)
        result = reader.read_quantity(text_img)

        if result is not None:
            qty, conf = result
            results[material_id] = (qty, conf)
            flag = "[CHECK] " if conf < confidence_threshold else ""
            print(f"  {flag}{material_id}: {qty} ({conf:.2f})")
        else:
            print(f"  {material_id}: OCR failed, skipping")

    return results
```

- [ ] **Step 3: main() 함수에 --check, --confidence-threshold 인자 추가 및 결과 처리 수정**

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

    if args.start_id:
        print(f"\nSequential mode: starting from {args.start_id}")
        images = [img for _, img in loaded_images]
        results = process_screenshots_sequential(
            images, item_order, args.start_id, reader, args.confidence_threshold
        )
        all_results.update(results)
    else:
        matcher = ItemMatcher(Path(args.refs))
        print(f"Loaded {len(matcher.references)} reference images")
        for img_path, img in loaded_images:
            print(f"\nProcessing: {img_path}")
            results = process_screenshot_matched(
                img, item_order, matcher, reader, args.confidence_threshold
            )
            for mid, val in results.items():
                if mid in all_results:
                    print(f"  Duplicate: {mid} (keeping {all_results[mid][0]})")
                else:
                    all_results[mid] = val

    # Summary
    qty_only = {k: v[0] for k, v in all_results.items()}
    check_count = sum(1 for _, (_, conf) in all_results.items() if conf < args.confidence_threshold)
    print(f"\nTotal items recognized: {len(all_results)}")
    if check_count:
        print(f"Items needing review: {check_count}")

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
```

- [ ] **Step 4: 전체 테스트 실행**

Run: `.venv313/Scripts/python.exe -m pytest tests/ -v`
Expected: 모든 테스트 PASS

- [ ] **Step 5: Commit**

```bash
git add src/main.py
git commit -m "feat: add confidence display, CHECK flag, and --check accuracy verification"
```

---

### Task 4: 장비 스크린샷으로 정확도 검증

**Files:**
- None (실행 테스트만)

- [ ] **Step 1: 그룹2 (장비) 실행**

Run:
```bash
.venv313/Scripts/python.exe -m src.main \
  --images docs/input-image/10.png docs/input-image/11.png docs/input-image/12.png \
  --json docs/input-image/justin163.json \
  --start-id T2_Hat \
  --output docs/output/result_equipment.json \
  --check answer.json
```

Expected: accuracy report 출력. 목표 90%+ (73/81 이상).

- [ ] **Step 2: 그룹1 (일반) 회귀 테스트**

Run:
```bash
.venv313/Scripts/python.exe -m src.main \
  --images docs/input-image/01.png docs/input-image/02.png docs/input-image/03.png docs/input-image/04.png docs/input-image/05.png docs/input-image/06.png docs/input-image/07.png \
  --json docs/input-image/justin163.json \
  --start-id 100 \
  --output docs/output/result_general.json \
  --check answer.json
```

Expected: accuracy 99% 유지. 일반 아이템에서는 색상 마스크가 fallback(Otsu)으로 동작할 수 있음 — 정확도 저하 없어야 함.

- [ ] **Step 3: 결과 분석 및 필요시 tolerance 튜닝**

색상 마스크 tolerance (현재 ±30)가 부족하거나 과도한 경우:
- 오류 패턴 확인: 여전히 첫 자리 누락이면 tolerance 확대 또는 upscale 추가 증가
- 새로운 오류 유형이면 fallback 조건 조정
- `_FONT_TOLERANCE` 값을 조정하고 재실행

- [ ] **Step 4: Commit (튜닝 변경이 있는 경우)**

```bash
git add src/ocr_reader.py
git commit -m "fix: tune color mask tolerance for equipment text"
```
