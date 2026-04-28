"""Tests for YoloOcrReader — utility functions and ONNX inference."""

import numpy as np
import pytest
from pathlib import Path

from src.yolo_ocr_reader import YoloOcrReader
from src.count_ocr_backend import CountOcrBackend


# ---------------------------------------------------------------------------
# Test: letterbox
# ---------------------------------------------------------------------------
class TestLetterbox:
    def test_output_shape(self):
        from src.yolo_ocr_reader import letterbox

        img = np.zeros((50, 100, 3), dtype=np.uint8)
        out = letterbox(img, 64, 160, 114)
        assert out.shape == (64, 160, 3)

    def test_preserves_aspect_ratio(self):
        """A 50x100 image into 64x160 should scale uniformly."""
        from src.yolo_ocr_reader import letterbox

        img = np.full((50, 100, 3), 128, dtype=np.uint8)
        out = letterbox(img, 64, 160, 0)
        # Scale factor = min(64/50, 160/100) = min(1.28, 1.6) = 1.28
        # Scaled size = 64 x 128 → pad left/right by 16 each
        # The centre 128 columns should not be pad (0), at least some pixel is 128
        centre_col = out[:, 80, :]
        assert np.any(centre_col == 128)

    def test_pad_value_fills_border(self):
        from src.yolo_ocr_reader import letterbox

        img = np.full((50, 100, 3), 128, dtype=np.uint8)
        out = letterbox(img, 64, 160, 255)
        # With scale 1.28 → 64x128, padding on left/right = 16 each
        # Left border column should be pad value
        assert np.all(out[:, 0, :] == 255)
        assert np.all(out[:, -1, :] == 255)

    def test_single_channel(self):
        from src.yolo_ocr_reader import letterbox

        img = np.zeros((30, 60), dtype=np.uint8)
        out = letterbox(img, 64, 160, 114)
        assert out.shape[:2] == (64, 160)


# ---------------------------------------------------------------------------
# Test: compute_iou
# ---------------------------------------------------------------------------
class TestComputeIou:
    def test_identical_boxes(self):
        from src.yolo_ocr_reader import compute_iou

        box = [10, 10, 50, 50]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        from src.yolo_ocr_reader import compute_iou

        a = [0, 0, 10, 10]
        b = [20, 20, 30, 30]
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        from src.yolo_ocr_reader import compute_iou

        # Box a: 0,0 → 10,10 (area 100)
        # Box b: 5,5 → 15,15 (area 100)
        # Intersection: 5,5 → 10,10 (area 25)
        # Union: 100+100-25 = 175
        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        assert compute_iou(a, b) == pytest.approx(25 / 175)


# ---------------------------------------------------------------------------
# Test: nms
# ---------------------------------------------------------------------------
class TestNms:
    def test_removes_overlapping_lower_conf(self):
        from src.yolo_ocr_reader import nms

        rows = [
            [0, 0, 10, 10, 0.9, 1],  # high conf
            [1, 1, 11, 11, 0.5, 1],  # overlapping, lower conf → removed
        ]
        result = nms(rows, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0][4] == pytest.approx(0.9)

    def test_keeps_non_overlapping(self):
        from src.yolo_ocr_reader import nms

        rows = [
            [0, 0, 10, 10, 0.9, 1],
            [50, 50, 60, 60, 0.8, 2],
        ]
        result = nms(rows, iou_threshold=0.5)
        assert len(result) == 2

    def test_empty_input(self):
        from src.yolo_ocr_reader import nms

        assert nms([], 0.5) == []


# ---------------------------------------------------------------------------
# Test: assemble_digits
# ---------------------------------------------------------------------------
class TestAssembleDigits:
    def test_normal_x123(self):
        from src.yolo_ocr_reader import assemble_digits

        # Detections: [x1,y1,x2,y2,conf,class_id] sorted by x-center
        detections = [
            [10, 10, 20, 30, 0.9, 0],  # 'x'
            [25, 10, 35, 30, 0.8, 2],  # '1'
            [40, 10, 50, 30, 0.85, 3],  # '2'
            [55, 10, 65, 30, 0.7, 4],  # '3'
        ]
        result = assemble_digits(detections)
        assert result is not None
        value, raw, avg_conf = result
        assert value == 123
        assert raw == "x123"
        assert avg_conf == pytest.approx((0.9 + 0.8 + 0.85 + 0.7) / 4)

    def test_single_digit_x5(self):
        from src.yolo_ocr_reader import assemble_digits

        detections = [
            [10, 10, 20, 30, 0.9, 0],  # 'x'
            [25, 10, 35, 30, 0.8, 6],  # '5'
        ]
        result = assemble_digits(detections)
        assert result is not None
        assert result[0] == 5
        assert result[1] == "x5"

    def test_no_x_prefix_returns_none(self):
        from src.yolo_ocr_reader import assemble_digits

        detections = [
            [10, 10, 20, 30, 0.9, 2],  # '1' — no x prefix
            [25, 10, 35, 30, 0.8, 3],  # '2'
        ]
        assert assemble_digits(detections) is None

    def test_empty_returns_none(self):
        from src.yolo_ocr_reader import assemble_digits

        assert assemble_digits([]) is None

    def test_five_plus_digits_returns_none(self):
        from src.yolo_ocr_reader import assemble_digits

        detections = [
            [10, 10, 20, 30, 0.9, 0],   # 'x'
            [25, 10, 35, 30, 0.8, 2],   # '1'
            [40, 10, 50, 30, 0.8, 3],   # '2'
            [55, 10, 65, 30, 0.8, 4],   # '3'
            [70, 10, 80, 30, 0.8, 5],   # '4'
            [85, 10, 95, 30, 0.8, 6],   # '5'
        ]
        assert assemble_digits(detections) is None


# ---------------------------------------------------------------------------
# Test: YoloOcrReader interface (no model required)
# ---------------------------------------------------------------------------
class TestYoloOcrReaderInterface:
    """Tests that don't require the ONNX model file."""

    def test_is_count_ocr_backend(self):
        assert issubclass(YoloOcrReader, CountOcrBackend)


# ---------------------------------------------------------------------------
# Test: YoloOcrReader (skip if model not found)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent.parent / "models" / "YOLO26m_BA_AUTO_CAL_digt_v1.onnx"
skip_no_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="ONNX model not found"
)


@skip_no_model
class TestYoloOcrReader:
    @pytest.fixture(autouse=True)
    def _reader(self):
        from src.yolo_ocr_reader import YoloOcrReader

        self.reader = YoloOcrReader(str(MODEL_PATH))

    def test_read_quantity_returns_tuple_or_none(self):
        img = np.zeros((64, 160, 3), dtype=np.uint8)
        result = self.reader.read_quantity(img)
        assert result is None or (isinstance(result, tuple) and len(result) == 2)

    def test_blank_white_image_returns_none(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = self.reader.read_quantity(img)
        assert result is None

    def test_smoke_real_image(self, docs_dir):
        import cv2

        img_path = docs_dir / "input-image" / "01.png"
        if not img_path.exists():
            pytest.skip("Real test image not found")
        img = cv2.imread(str(img_path))
        assert img is not None
        # Just run inference — we can't know exact result but it shouldn't crash
        # The full image is a grid; crop a small region for a single cell test
        h, w = img.shape[:2]
        cell = img[0 : h // 5, 0 : w // 7]
        result = self.reader.read_quantity(cell)
        # Result can be None or a tuple — just verify no crash
        assert result is None or (isinstance(result, tuple) and len(result) == 2)
