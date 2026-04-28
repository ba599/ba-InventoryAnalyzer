"""YoloOcrReader — ONNX-based digit detection for inventory quantity reading."""

from __future__ import annotations

import ast
import re
import threading
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from src.count_ocr_backend import CountOcrBackend

CLASS_MAP = {
    0: "x", 1: "0", 2: "1", 3: "2", 4: "3",
    5: "4", 6: "5", 7: "6", 8: "7", 9: "8", 10: "9",
}

_QUANTITY_RE = re.compile(r"^x([0-9]{1,4})$")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def letterbox(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: int = 114,
) -> np.ndarray:
    """Resize *img* preserving aspect ratio, pad remaining area with *pad_value*."""
    if img.ndim == 2:
        h, w = img.shape
        channels = 0
    else:
        h, w = img.shape[:2]
        channels = img.shape[2]

    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create output canvas
    if channels:
        out = np.full((target_h, target_w, channels), pad_value, dtype=img.dtype)
    else:
        out = np.full((target_h, target_w), pad_value, dtype=img.dtype)

    # Centre the resized image
    dy = (target_h - new_h) // 2
    dx = (target_w - new_w) // 2
    if channels:
        out[dy : dy + new_h, dx : dx + new_w, :] = resized
    else:
        out[dy : dy + new_h, dx : dx + new_w] = resized

    return out


def compute_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU between two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(
    rows: list[list[float]],
    iou_threshold: float = 0.70,
) -> list[list[float]]:
    """Non-Maximum Suppression on ``[x1, y1, x2, y2, confidence, class_id]`` rows."""
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: r[4], reverse=True)
    keep: list[list[float]] = []
    for row in sorted_rows:
        if all(compute_iou(row, k) < iou_threshold for k in keep):
            keep.append(row)
    return keep


def assemble_digits(
    detections: list[list[float]],
) -> tuple[int, str, float] | None:
    """Sort detections by x-centre, build text, validate, return ``(value, raw, avg_conf)``."""
    if not detections:
        return None
    # Sort by x-centre
    dets = sorted(detections, key=lambda d: (d[0] + d[2]) / 2)
    chars: list[str] = []
    confs: list[float] = []
    for d in dets:
        cls_id = int(d[5])
        ch = CLASS_MAP.get(cls_id)
        if ch is None:
            return None
        chars.append(ch)
        confs.append(d[4])
    raw = "".join(chars)
    m = _QUANTITY_RE.match(raw)
    if m is None:
        return None
    value = int(m.group(1))
    avg_conf = sum(confs) / len(confs)
    return value, raw, avg_conf


# ---------------------------------------------------------------------------
# YoloOcrReader
# ---------------------------------------------------------------------------


class YoloOcrReader(CountOcrBackend):
    """ONNX YOLO-based digit detector for inventory quantity reading."""

    _session_cache: dict[str, ort.InferenceSession] = {}
    _lock = threading.Lock()

    @property
    def backend_name(self) -> str:
        return "yolo"

    def __init__(self, model_path: str = "models/YOLO26m_BA_AUTO_CAL_digt_v1.onnx"):
        self._model_path = model_path
        self._session = self._get_session(model_path)

        # Read model input size from metadata (fallback 160x160)
        meta = self._session.get_modelmeta().custom_metadata_map
        try:
            imgsz = ast.literal_eval(meta.get("imgsz", "[160, 160]"))
            self._input_h, self._input_w = int(imgsz[0]), int(imgsz[1])
        except Exception:
            self._input_h, self._input_w = 160, 160

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name

    @classmethod
    def _get_session(cls, model_path: str) -> ort.InferenceSession:
        with cls._lock:
            if model_path not in cls._session_cache:
                cls._session_cache[model_path] = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
            return cls._session_cache[model_path]

    # ---- preprocessing ----------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Two-stage letterbox → BGR→RGB → float32/255 → CHW → NCHW."""
        img = image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Stage 1: letterbox to 64×160, white padding
        img = letterbox(img, 64, 160, pad_value=255)

        # Stage 2: letterbox to model input size, gray padding
        img = letterbox(img, self._input_h, self._input_w, pad_value=114)

        # BGR → RGB → float32 / 255 → CHW → NCHW
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, 0)  # NCHW
        return img

    # ---- postprocessing ---------------------------------------------------

    @staticmethod
    def _postprocess(output: np.ndarray) -> list[list[float]]:
        """Decode ONNX output, filter, NMS, dedup."""
        # output shape: [1, N, 6] → [x1, y1, x2, y2, conf, cls]
        preds = output[0]  # (N, 6)

        # Filter by confidence and validity
        rows: list[list[float]] = []
        for det in preds:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            if conf < 0.25:
                continue
            cls_id_int = int(round(cls_id))
            if cls_id_int not in CLASS_MAP:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append([x1, y1, x2, y2, conf, float(cls_id_int)])

        # NMS
        rows = nms(rows, iou_threshold=0.70)

        # Additional deduplication
        rows = nms(rows, iou_threshold=0.80)

        return rows

    # ---- public API -------------------------------------------------------

    def read_quantity(self, image: np.ndarray) -> tuple[int, float] | None:
        """Read quantity from a cell image.

        Returns ``(quantity, avg_confidence)`` or ``None``.
        """
        blob = self._preprocess(image)
        output = self._session.run(None, {self._input_name: blob})[0]
        detections = self._postprocess(output)
        result = assemble_digits(detections)
        if result is None:
            return None
        value, _raw, avg_conf = result
        return (value, avg_conf)
