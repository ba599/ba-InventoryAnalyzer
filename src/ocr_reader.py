import re
import cv2
import numpy as np
import easyocr


class OcrReader:
    def __init__(self):
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

        Converts to grayscale, upscales small images, and applies
        horizontal shear to correct the game's italic font.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if gray.shape[0] < 40:
            scale = 40 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Correct italic font tilt with horizontal shear
        h, w = gray.shape[:2]
        shear = 0.15
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        new_w = int(w + shear * h)
        gray = cv2.warpAffine(gray, M, (new_w, h), borderValue=255)
        return gray

    def read_quantity(self, image: np.ndarray) -> tuple[int, float] | None:
        """Read quantity from a cell's text region image.

        Returns:
            (quantity, confidence) tuple, or None if OCR fails.
        """
        processed = self.preprocess(image)
        results = self._reader.readtext(processed, allowlist="xX0123456789")
        for bbox, text, conf in results:
            qty = self.parse_quantity(text)
            if qty is not None:
                return (qty, float(conf))
        return None
