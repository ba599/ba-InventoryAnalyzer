# YOLO Digit Detection A/B Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EasyOCR과 YOLO digit detection 백엔드를 동일 이미지셋으로 동시 실행하여 정확도를 비교하는 A/B 테스트 스크립트를 만든다.

**Architecture:** 기존 코드를 변경하지 않고, `YoloOcrReader`를 별도 파일로 구현한다. `compare_backends.py` 스크립트에서 두 리더를 동시에 실행하고 `answer.json` 기준으로 정확도를 비교 리포트로 출력한다.

**Tech Stack:** onnxruntime (ONNX 모델 추론), OpenCV (전처리), NumPy

**Spec:** `docs/superpowers/specs/2026-04-28-yolo-ab-test-design.md`
**Reference:** `docs/YOLO_DIGIT_OCR_REFERENCE.md`

**Prerequisites:** 프로젝트가 git 레포가 아니므로, 실행 전 `git init` + 초기 커밋이 필요하다. 워크트리를 만들어 격리된 환경에서 작업한다.

---

## File Structure

```
models/                                    # 신규 디렉토리
  YOLO26m_BA_AUTO_CAL_digt_v1.onnx        # docs/에서 이동
src/
  ocr_reader.py                            # 변경 없음
  yolo_ocr_reader.py                       # 신규 — YOLO digit detection 리더
  compare_backends.py                      # 신규 — A/B 비교 스크립트
tests/
  test_yolo_ocr_reader.py                  # 신규 — YOLO 리더 단위 테스트
  test_compare_backends.py                 # 신규 — 비교 스크립트 테스트
requirements.txt                           # 수정 — onnxruntime 추가
```

---

## Task 0: Git 초기화 및 워크트리 설정

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Git 초기화 및 초기 커밋**

```bash
cd C:\Users\admin\Documents\workspace\test\ba-InventoryAnalyzer-master
git init
git add -A
git commit -m "chore: initial commit"
```

- [ ] **Step 2: 워크트리 브랜치 생성 및 체크아웃**

```bash
git branch feat/yolo-ab-test
git worktree add ../ba-InventoryAnalyzer-yolo-ab-test feat/yolo-ab-test
```

이후 모든 작업은 `../ba-InventoryAnalyzer-yolo-ab-test/` 디렉토리에서 진행한다.

---

## Task 1: 프로젝트 설정 (모델 파일 이동 + 의존성)

**Files:**
- Create: `models/` 디렉토리
- Move: `docs/YOLO26m_BA_AUTO_CAL_digt_v1.onnx` → `models/YOLO26m_BA_AUTO_CAL_digt_v1.onnx`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: 모델 파일 이동**

```bash
mkdir models
mv docs/YOLO26m_BA_AUTO_CAL_digt_v1.onnx models/
```

- [ ] **Step 2: .gitignore에 모델 파일 제외 추가**

`.gitignore` 파일 끝에 추가:

```
# ONNX model files (large binary)
models/*.onnx
```

- [ ] **Step 3: requirements.txt에 onnxruntime 추가**

`requirements.txt`에 한 줄 추가:

```
onnxruntime>=1.16.0
```

최종 `requirements.txt`:
```
opencv-python>=4.8.0
easyocr>=1.7.0
numpy>=1.24.0
pytest>=7.0.0
PySide6>=6.6.0
onnxruntime>=1.16.0
```

- [ ] **Step 4: onnxruntime 설치**

```bash
pip install onnxruntime>=1.16.0
```

- [ ] **Step 5: 커밋**

```bash
git add requirements.txt .gitignore
git commit -m "chore: add onnxruntime dependency, move ONNX model to models/"
```

---

## Task 2: YoloOcrReader — 유틸리티 함수 (전처리/후처리)

**Files:**
- Create: `src/yolo_ocr_reader.py`
- Create: `tests/test_yolo_ocr_reader.py`

이 태스크에서는 YOLO 파이프라인의 개별 유틸리티 함수들을 TDD로 구현한다.

### 2-1: letterbox 리사이즈

- [ ] **Step 1: letterbox 테스트 작성**

`tests/test_yolo_ocr_reader.py`:

```python
import numpy as np
import pytest
from src.yolo_ocr_reader import letterbox


class TestLetterbox:
    def test_output_shape_matches_target(self):
        img = np.zeros((30, 80, 3), dtype=np.uint8)
        result = letterbox(img, target_h=64, target_w=160, pad_value=114)
        assert result.shape == (64, 160, 3)

    def test_preserves_aspect_ratio(self):
        """A 30x80 image scaled to fit 64x160 → scale=0.8, resized to 24x64, centered."""
        img = np.full((30, 80, 3), 200, dtype=np.uint8)
        result = letterbox(img, target_h=64, target_w=160, pad_value=0)
        # Top/bottom padding area should be pad_value (0)
        assert result[0, 80, 0] == 0  # top padding region, center x
        # Center region should contain image content
        center_y = 64 // 2
        center_x = 160 // 2
        assert result[center_y, center_x, 0] != 0

    def test_pad_value_fills_border(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = letterbox(img, target_h=64, target_w=160, pad_value=114)
        # Corner should be pad value (far from center where image is)
        assert result[0, 0, 0] == 114
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestLetterbox -v
```

Expected: FAIL — `cannot import name 'letterbox'`

- [ ] **Step 3: letterbox 구현**

`src/yolo_ocr_reader.py`:

```python
import re
import threading

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: int = 114,
) -> np.ndarray:
    """Resize image with letterbox padding, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestLetterbox -v
```

Expected: 3 passed

### 2-2: IoU 계산

- [ ] **Step 5: IoU 테스트 작성**

`tests/test_yolo_ocr_reader.py`에 추가:

```python
from src.yolo_ocr_reader import compute_iou


class TestComputeIou:
    def test_identical_boxes(self):
        a = [0, 0, 10, 10]
        assert compute_iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = [0, 0, 10, 10]
        b = [20, 20, 30, 30]
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        # intersection: 5x5=25, union: 100+100-25=175
        assert compute_iou(a, b) == pytest.approx(25 / 175)
```

- [ ] **Step 6: 테스트 실패 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestComputeIou -v
```

Expected: FAIL — `cannot import name 'compute_iou'`

- [ ] **Step 7: compute_iou 구현**

`src/yolo_ocr_reader.py`에 추가:

```python
def compute_iou(a: list, b: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = max((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter, 1e-12)
    return inter / union
```

- [ ] **Step 8: 테스트 통과 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestComputeIou -v
```

Expected: 3 passed

### 2-3: NMS

- [ ] **Step 9: NMS 테스트 작성**

`tests/test_yolo_ocr_reader.py`에 추가:

```python
from src.yolo_ocr_reader import nms


class TestNms:
    def test_removes_overlapping_lower_confidence(self):
        # Two nearly identical boxes, different confidence
        rows = [
            [0, 0, 10, 10, 0.9, 2],  # high conf
            [1, 1, 11, 11, 0.5, 2],  # low conf, overlaps heavily
        ]
        kept = nms(rows, iou_threshold=0.5)
        assert len(kept) == 1
        assert kept[0][4] == 0.9

    def test_keeps_non_overlapping(self):
        rows = [
            [0, 0, 10, 10, 0.9, 1],
            [50, 50, 60, 60, 0.8, 2],
        ]
        kept = nms(rows, iou_threshold=0.5)
        assert len(kept) == 2

    def test_empty_input(self):
        assert nms([], iou_threshold=0.5) == []
```

- [ ] **Step 10: 테스트 실패 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestNms -v
```

Expected: FAIL

- [ ] **Step 11: NMS 구현**

`src/yolo_ocr_reader.py`에 추가:

```python
def nms(rows: list, iou_threshold: float = 0.70) -> list:
    """Non-Maximum Suppression: keep highest-confidence, remove overlaps."""
    ordered = sorted(rows, key=lambda r: float(r[4]), reverse=True)
    kept = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [o for o in ordered if compute_iou(current, o) < iou_threshold]
    return kept
```

- [ ] **Step 12: 테스트 통과 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestNms -v
```

Expected: 3 passed

### 2-4: 숫자 조합 (assemble_digits)

- [ ] **Step 13: assemble_digits 테스트 작성**

`tests/test_yolo_ocr_reader.py`에 추가:

```python
from src.yolo_ocr_reader import assemble_digits

# CLASS_MAP: {0: "x", 1: "0", 2: "1", 3: "2", 4: "3", 5: "4",
#             6: "5", 7: "6", 8: "7", 9: "8", 10: "9"}


class TestAssembleDigits:
    def test_normal_quantity(self):
        # "x123" — detections sorted by x-center
        detections = [
            [10, 5, 20, 15, 0.9, 0],   # x_center=15, class 0 → "x"
            [25, 5, 35, 15, 0.85, 2],   # x_center=30, class 2 → "1"
            [40, 5, 50, 15, 0.80, 3],   # x_center=45, class 3 → "2"
            [55, 5, 65, 15, 0.88, 4],   # x_center=60, class 4 → "3"
        ]
        value, text, conf = assemble_digits(detections)
        assert value == 123
        assert text == "x123"
        assert conf == pytest.approx((0.9 + 0.85 + 0.80 + 0.88) / 4)

    def test_single_digit(self):
        detections = [
            [10, 5, 20, 15, 0.9, 0],   # "x"
            [25, 5, 35, 15, 0.8, 6],   # class 6 → "5"
        ]
        value, text, conf = assemble_digits(detections)
        assert value == 5
        assert text == "x5"

    def test_no_x_prefix_returns_none(self):
        detections = [
            [10, 5, 20, 15, 0.9, 2],   # "1" (no "x")
            [25, 5, 35, 15, 0.8, 3],   # "2"
        ]
        value, text, conf = assemble_digits(detections)
        assert value is None
        assert text == "12"

    def test_empty_detections(self):
        value, text, conf = assemble_digits([])
        assert value is None
        assert text == ""
        assert conf == 0.0

    def test_too_many_digits_returns_none(self):
        # "x12345" — 5자리, 패턴은 1~4자리만 허용
        detections = [
            [10, 5, 20, 15, 0.9, 0],   # "x"
            [25, 5, 35, 15, 0.8, 2],   # "1"
            [40, 5, 50, 15, 0.8, 3],   # "2"
            [55, 5, 65, 15, 0.8, 4],   # "3"
            [70, 5, 80, 15, 0.8, 5],   # "4"
            [85, 5, 95, 15, 0.8, 6],   # "5"
        ]
        value, text, conf = assemble_digits(detections)
        assert value is None
```

- [ ] **Step 14: 테스트 실패 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestAssembleDigits -v
```

Expected: FAIL

- [ ] **Step 15: assemble_digits 구현**

`src/yolo_ocr_reader.py`에 추가:

```python
CLASS_MAP = {0: "x", 1: "0", 2: "1", 3: "2", 4: "3", 5: "4",
             6: "5", 7: "6", 8: "7", 9: "8", 10: "9"}

COUNT_PATTERN = re.compile(r"^x([0-9]{1,4})$")


def assemble_digits(detections: list) -> tuple[int | None, str, float]:
    """Assemble detected digit boxes into a quantity value.

    Args:
        detections: List of [x1, y1, x2, y2, confidence, class_id] rows.

    Returns:
        (value, raw_text, avg_confidence) — value is None if pattern doesn't match.
    """
    if not detections:
        return None, "", 0.0

    # Sort by x-center, then y-center
    ordered = sorted(detections, key=lambda d: ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2))
    raw_text = "".join(CLASS_MAP.get(int(round(d[5])), "?") for d in ordered)
    avg_conf = sum(float(d[4]) for d in ordered) / len(ordered)

    match = COUNT_PATTERN.fullmatch(raw_text)
    if not match:
        return None, raw_text, avg_conf

    return int(match.group(1)), raw_text, avg_conf
```

- [ ] **Step 16: 테스트 통과 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestAssembleDigits -v
```

Expected: 5 passed

- [ ] **Step 17: 커밋**

```bash
git add src/yolo_ocr_reader.py tests/test_yolo_ocr_reader.py
git commit -m "feat: add YOLO OCR utility functions (letterbox, IoU, NMS, assemble_digits)"
```

---

## Task 3: YoloOcrReader — ONNX 추론 및 read_quantity

**Files:**
- Modify: `src/yolo_ocr_reader.py`
- Modify: `tests/test_yolo_ocr_reader.py`

### 3-1: YoloOcrReader 클래스 (ONNX 세션 + 전처리 + 후처리)

- [ ] **Step 1: YoloOcrReader 테스트 작성**

`tests/test_yolo_ocr_reader.py`에 추가:

```python
from pathlib import Path
from src.yolo_ocr_reader import YoloOcrReader

MODEL_PATH = Path("models/YOLO26m_BA_AUTO_CAL_digt_v1.onnx")


class TestYoloOcrReader:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not MODEL_PATH.exists():
            pytest.skip("ONNX model not found")
        self.reader = YoloOcrReader(str(MODEL_PATH))

    def test_read_quantity_returns_tuple_or_none(self):
        """Return type is (int, float) or None."""
        img = np.full((20, 60, 3), 200, dtype=np.uint8)
        cv2.putText(img, "x500", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (45, 70, 99), 1)
        result = self.reader.read_quantity(img)
        # Synthetic image may not work with YOLO; just check type contract
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 2
            qty, conf = result
            assert isinstance(qty, int)
            assert isinstance(conf, float)
            assert 0.0 <= conf <= 1.0

    def test_empty_image_returns_none(self):
        """Blank white image should return None (no digits detected)."""
        img = np.full((20, 60, 3), 255, dtype=np.uint8)
        result = self.reader.read_quantity(img)
        assert result is None

    def test_read_quantity_with_real_image(self, docs_dir):
        """Smoke test with a real game screenshot cell crop."""
        img_path = docs_dir / "input-image" / "01.png"
        if not img_path.exists():
            pytest.skip("Test image not found")
        from src.grid_detector import detect_cells, crop_text_region
        image = cv2.imread(str(img_path))
        cells = detect_cells(image)
        if not cells:
            pytest.skip("No cells detected")
        text_img = crop_text_region(image, cells[0])
        result = self.reader.read_quantity(text_img)
        # At least verify it doesn't crash; result can be None or (int, float)
        if result is not None:
            qty, conf = result
            assert qty > 0
            assert 0.0 <= conf <= 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestYoloOcrReader -v
```

Expected: FAIL — `cannot import name 'YoloOcrReader'`

- [ ] **Step 3: YoloOcrReader 구현**

`src/yolo_ocr_reader.py`에 추가:

```python
import onnxruntime as ort


CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.70
DUPLICATE_IOU_THRESHOLD = 0.80
VALID_CLASS_IDS = set(range(11))
EXPORT_COUNT_SIZE = (64, 160)  # (height, width) — first letterbox target


class YoloOcrReader:
    """YOLO-based digit detection OCR reader.

    Uses a YOLO ONNX model trained to detect individual digits (0-9) and 'x' prefix
    as separate objects, then assembles them into quantity values.
    """

    _lock = threading.Lock()
    _session_cache: dict[str, ort.InferenceSession] = {}

    def __init__(self, model_path: str = "models/YOLO26m_BA_AUTO_CAL_digt_v1.onnx"):
        self._model_path = str(model_path)
        self._session: ort.InferenceSession | None = None
        self._input_h: int = 160
        self._input_w: int = 160

    def _get_session(self) -> ort.InferenceSession:
        if self._session is not None:
            return self._session
        with self._lock:
            if self._model_path in self._session_cache:
                self._session = self._session_cache[self._model_path]
            else:
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                session = ort.InferenceSession(
                    self._model_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._session_cache[self._model_path] = session
                self._session = session
            # Read input size from model metadata
            inp = self._session.get_inputs()[0]
            self._input_h = inp.shape[2] if isinstance(inp.shape[2], int) else 160
            self._input_w = inp.shape[3] if isinstance(inp.shape[3], int) else 160
        return self._session

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Two-stage letterbox + tensor conversion."""
        # Stage 1: letterbox to 160x64 with white padding
        stage1 = letterbox(image_bgr, EXPORT_COUNT_SIZE[0], EXPORT_COUNT_SIZE[1], pad_value=255)
        # Stage 2: letterbox to model input size with gray(114) padding
        stage2 = letterbox(stage1, self._input_h, self._input_w, pad_value=114)
        # BGR → RGB → float32 /255 → CHW → NCHW
        rgb = cv2.cvtColor(stage2, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _decode_output(self, output) -> list:
        """Filter raw ONNX output by confidence and class validity."""
        rows = np.asarray(output, dtype=np.float32)
        if rows.ndim == 3:
            rows = rows[0]
        candidates = []
        for row in rows:
            if not np.isfinite(row).all():
                continue
            conf = float(row[4])
            class_id = int(round(float(row[5])))
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if conf < CONF_THRESHOLD:
                continue
            if class_id not in VALID_CLASS_IDS:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append([x1, y1, x2, y2, conf, class_id])
        return candidates

    def read_quantity(self, image: np.ndarray) -> tuple[int, float] | None:
        """Read quantity from a cell's text region image.

        Args:
            image: BGR image from crop_text_region().

        Returns:
            (quantity, avg_confidence) tuple, or None if detection fails.
        """
        if image.size == 0:
            return None

        session = self._get_session()
        tensor = self._preprocess(image)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})

        detections = self._decode_output(outputs[0])
        detections = nms(detections, NMS_IOU_THRESHOLD)

        # Additional deduplication pass
        kept = []
        for det in sorted(detections, key=lambda d: d[4], reverse=True):
            if not any(compute_iou(existing, det) >= DUPLICATE_IOU_THRESHOLD for existing in kept):
                kept.append(det)

        value, raw_text, avg_conf = assemble_digits(kept)
        if value is None:
            return None
        return (value, avg_conf)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_yolo_ocr_reader.py::TestYoloOcrReader -v
```

Expected: 3 passed (또는 모델 없으면 skipped)

- [ ] **Step 5: 커밋**

```bash
git add src/yolo_ocr_reader.py tests/test_yolo_ocr_reader.py
git commit -m "feat: implement YoloOcrReader with ONNX inference pipeline"
```

---

## Task 4: compare_backends.py — A/B 비교 스크립트

**Files:**
- Create: `src/compare_backends.py`
- Create: `tests/test_compare_backends.py`

### 4-1: 비교 로직 함수

- [ ] **Step 1: 비교 로직 테스트 작성**

`tests/test_compare_backends.py`:

```python
import pytest
from src.compare_backends import build_comparison_report


class TestBuildComparisonReport:
    def test_both_correct(self):
        easyocr_results = {"100": 2375, "101": 1475}
        yolo_results = {"100": 2375, "101": 1475}
        answer = {"owned_materials": {"100": "2375", "101": "1475"}}
        report = build_comparison_report(easyocr_results, yolo_results, answer)
        assert report["easyocr"]["correct"] == 2
        assert report["yolo"]["correct"] == 2
        assert report["easyocr"]["accuracy"] == 100.0
        assert report["yolo"]["accuracy"] == 100.0
        assert report["disagreements"] == []

    def test_yolo_better(self):
        easyocr_results = {"100": 2375, "101": 9999}
        yolo_results = {"100": 2375, "101": 1475}
        answer = {"owned_materials": {"100": "2375", "101": "1475"}}
        report = build_comparison_report(easyocr_results, yolo_results, answer)
        assert report["easyocr"]["correct"] == 1
        assert report["yolo"]["correct"] == 2
        assert len(report["disagreements"]) == 1
        d = report["disagreements"][0]
        assert d["id"] == "101"
        assert d["answer"] == 1475
        assert d["easyocr"] == 9999
        assert d["yolo"] == 1475

    def test_both_wrong_differently(self):
        easyocr_results = {"100": 111}
        yolo_results = {"100": 222}
        answer = {"owned_materials": {"100": "2375"}}
        report = build_comparison_report(easyocr_results, yolo_results, answer)
        assert report["easyocr"]["correct"] == 0
        assert report["yolo"]["correct"] == 0
        assert len(report["disagreements"]) == 1

    def test_missing_item_in_one_backend(self):
        """If an item is only in one backend's results, it still appears in disagreements."""
        easyocr_results = {"100": 2375}
        yolo_results = {"100": 2375, "101": 1475}
        answer = {"owned_materials": {"100": "2375", "101": "1475"}}
        report = build_comparison_report(easyocr_results, yolo_results, answer)
        # "101" is only in yolo — disagreement
        assert len(report["disagreements"]) == 1
        d = report["disagreements"][0]
        assert d["id"] == "101"
        assert d["easyocr"] is None
        assert d["yolo"] == 1475
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_compare_backends.py -v
```

Expected: FAIL — `cannot import name 'build_comparison_report'`

- [ ] **Step 3: build_comparison_report 구현**

`src/compare_backends.py`:

```python
"""A/B comparison of EasyOCR vs YOLO digit detection backends."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.accuracy_checker import compare_results


def build_comparison_report(
    easyocr_results: dict[str, int],
    yolo_results: dict[str, int],
    answer_data: dict,
) -> dict:
    """Compare two backends' results against ground truth.

    Returns:
        {
            "easyocr": {correct, total, errors, accuracy},
            "yolo": {correct, total, errors, accuracy},
            "disagreements": [{id, answer, easyocr, yolo}, ...],
        }
    """
    easyocr_report = compare_results(easyocr_results, answer_data)
    yolo_report = compare_results(yolo_results, answer_data)

    answer = answer_data.get("owned_materials", {})
    all_ids = sorted(set(easyocr_results.keys()) | set(yolo_results.keys()))

    disagreements = []
    for mid in all_ids:
        e_val = easyocr_results.get(mid)
        y_val = yolo_results.get(mid)
        if e_val != y_val and mid in answer:
            disagreements.append({
                "id": mid,
                "answer": int(answer[mid]),
                "easyocr": e_val,
                "yolo": y_val,
            })

    return {
        "easyocr": easyocr_report,
        "yolo": yolo_report,
        "disagreements": disagreements,
    }
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_compare_backends.py -v
```

Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/compare_backends.py tests/test_compare_backends.py
git commit -m "feat: add build_comparison_report for A/B result comparison"
```

### 4-2: 리포트 출력 및 CLI

- [ ] **Step 6: print_comparison_report 및 main 구현**

`src/compare_backends.py`에 추가:

```python
from src.accuracy_checker import load_answer
from src.core.pipeline import load_item_order, process_all_images
from src.grid_detector import crop_text_region, detect_cells
from src.item_matcher import ItemMatcher
from src.ocr_reader import OcrReader
from src.yolo_ocr_reader import YoloOcrReader


def print_comparison_report(report: dict) -> None:
    """Print formatted comparison report to stdout."""
    e = report["easyocr"]
    y = report["yolo"]

    print("\n=== Backend Comparison Report ===\n")
    print(f"{'Backend':<12} {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"{'-'*12} {'-'*7}  {'-'*5}  {'-'*8}")
    print(f"{'EasyOCR':<12} {e['correct']:>7}  {e['total']:>5}  {e['accuracy']:>7.1f}%")
    print(f"{'YOLO':<12} {y['correct']:>7}  {y['total']:>5}  {y['accuracy']:>7.1f}%")

    disag = report["disagreements"]
    if disag:
        print(f"\n=== Disagreements ({len(disag)} items) ===\n")
        print(f"{'Item':<14} {'Answer':>6}  {'EasyOCR':>8}  {'YOLO':>8}")
        print(f"{'-'*14} {'-'*6}  {'-'*8}  {'-'*8}")
        for d in disag:
            e_str = str(d['easyocr']) if d['easyocr'] is not None else "N/A"
            y_str = str(d['yolo']) if d['yolo'] is not None else "N/A"
            e_flag = " [OK]" if d['easyocr'] == d['answer'] else " [X]" if d['easyocr'] is not None else ""
            y_flag = " [OK]" if d['yolo'] == d['answer'] else " [X]" if d['yolo'] is not None else ""
            print(f"{d['id']:<14} {d['answer']:>6}  {e_str + e_flag:>8}  {y_str + y_flag:>8}")
    else:
        print("\nNo disagreements — both backends produced identical results.")


def run_comparison(
    image_paths: list[str],
    order_path: str,
    answer_path: str,
    refs_path: str = "references",
    model_path: str = "models/YOLO26m_BA_AUTO_CAL_digt_v1.onnx",
) -> dict:
    """Run both backends on the same images and return comparison report."""
    item_order = load_item_order(Path(order_path))
    answer_data = load_answer(Path(answer_path))
    matcher = ItemMatcher(Path(refs_path))

    easyocr_reader = OcrReader()
    yolo_reader = YoloOcrReader(model_path)

    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)

    print(f"Loaded {len(images)} images, {len(matcher.references)} references")

    # EasyOCR pass
    print("\nRunning EasyOCR...")
    easyocr_full, _ = process_all_images(images, item_order, matcher, easyocr_reader)
    easyocr_results = {mid: qty for mid, (qty, _) in easyocr_full.items()}

    # YOLO pass
    print("Running YOLO...")
    yolo_full, _ = process_all_images(images, item_order, matcher, yolo_reader)
    yolo_results = {mid: qty for mid, (qty, _) in yolo_full.items()}

    print(f"\nEasyOCR recognized: {len(easyocr_results)} items")
    print(f"YOLO recognized: {len(yolo_results)} items")

    return build_comparison_report(easyocr_results, yolo_results, answer_data)


def main():
    parser = argparse.ArgumentParser(
        description="Compare EasyOCR vs YOLO digit detection on inventory screenshots"
    )
    parser.add_argument("--images", nargs="+", required=True, help="Screenshot image paths")
    parser.add_argument("--order", default="item_order.json", help="Item order JSON")
    parser.add_argument("--answer", required=True, help="Answer JSON for ground truth")
    parser.add_argument("--refs", default="references", help="Reference images directory")
    parser.add_argument("--model", default="models/YOLO26m_BA_AUTO_CAL_digt_v1.onnx",
                        help="YOLO ONNX model path")
    args = parser.parse_args()

    report = run_comparison(args.images, args.order, args.answer, args.refs, args.model)
    print_comparison_report(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 커밋**

```bash
git add src/compare_backends.py
git commit -m "feat: add CLI and formatted report output for compare_backends"
```

---

## Task 5: 전체 테스트 실행 및 A/B 테스트 실행

**Files:** 없음 (실행만)

- [ ] **Step 1: 전체 테스트 스위트 실행**

```bash
pytest tests/ -v
```

Expected: 전체 통과 (모델 없는 환경에서는 YOLO 통합 테스트만 skipped)

- [ ] **Step 2: A/B 비교 실행**

```bash
python -m src.compare_backends --images docs/input-image/01.png docs/input-image/02.png docs/input-image/03.png docs/input-image/04.png docs/input-image/05.png docs/input-image/06.png docs/input-image/07.png docs/input-image/10.png docs/input-image/11.png docs/input-image/12.png --answer answer.json --order item_order.json --refs references --model models/YOLO26m_BA_AUTO_CAL_digt_v1.onnx
```

Expected: 두 백엔드의 정확도 비교 리포트 출력

- [ ] **Step 3: 결과 확인 후 커밋**

결과를 확인하고, 문제 없으면 최종 커밋:

```bash
git add -A
git commit -m "feat: complete YOLO A/B test implementation"
```

---

## Task 6: 결과 판단 및 정리

- [ ] **Step 1: 결과 분석**

A/B 테스트 리포트를 바탕으로 판단:
- YOLO가 전체적으로 우수하면 → 정식 통합 진행 (별도 계획 참고)
- EasyOCR이 동등하거나 우수하면 → 워크트리 삭제로 롤백

- [ ] **Step 2: 워크트리 정리 (YOLO 채택 시)**

```bash
cd C:\Users\admin\Documents\workspace\test\ba-InventoryAnalyzer-master
git merge feat/yolo-ab-test
git worktree remove ../ba-InventoryAnalyzer-yolo-ab-test
```

- [ ] **Step 3: 워크트리 정리 (롤백 시)**

```bash
cd C:\Users\admin\Documents\workspace\test\ba-InventoryAnalyzer-master
git worktree remove ../ba-InventoryAnalyzer-yolo-ab-test
git branch -D feat/yolo-ab-test
```
