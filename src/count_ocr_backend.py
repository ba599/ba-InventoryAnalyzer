"""Abstract base class for OCR backends and factory function."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CountOcrBackend(ABC):
    """Common interface for quantity-reading OCR backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for this backend."""

    @abstractmethod
    def read_quantity(self, image: np.ndarray) -> tuple[int, float] | None:
        """Read quantity from a cell's text region image.

        Returns:
            (quantity, confidence) tuple, or None if recognition fails.
        """


def build_backend(**kwargs) -> CountOcrBackend:
    """Create the YOLO OCR backend.

    Args:
        **kwargs: Forwarded to YoloOcrReader constructor (e.g. model_path).

    Returns:
        A CountOcrBackend instance.
    """
    from src.yolo_ocr_reader import YoloOcrReader
    return YoloOcrReader(**kwargs)
