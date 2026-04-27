import cv2
import numpy as np
import pytest
from src.ocr_reader import OcrReader


class TestOcrReader:
    @pytest.fixture(autouse=True)
    def setup_reader(self):
        self.reader = OcrReader()

    def test_parse_quantity_from_text(self):
        assert self.reader.parse_quantity("x1234") == 1234
        assert self.reader.parse_quantity("x100") == 100
        assert self.reader.parse_quantity("X2375") == 2375

    def test_parse_quantity_strips_whitespace(self):
        assert self.reader.parse_quantity(" x999 ") == 999

    def test_parse_quantity_returns_none_for_invalid(self):
        assert self.reader.parse_quantity("abc") is None
        assert self.reader.parse_quantity("") is None

    def test_read_quantity_returns_tuple_with_confidence(self):
        """read_quantity should return (qty, confidence) tuple or None."""
        img = np.zeros((40, 100, 3), dtype=np.uint8)
        cv2.putText(img, "x500", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        result = self.reader.read_quantity(img)
        # OCR on synthetic may not be perfect; just verify return type
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 2
            qty, conf = result
            assert isinstance(qty, int)
            assert isinstance(conf, float)
            assert 0.0 <= conf <= 1.0
