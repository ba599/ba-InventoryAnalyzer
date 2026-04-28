# YOLO 기반 숫자 인식(Digit OCR) 기술 레퍼런스

> **출처**: "블라리 인랭 계산기" v0.1.0 프로젝트의 `YOLO26m_BA_AUTO_CAL_digt_v1.onnx` 모델 및 관련 Python 코드를 역분석한 결과입니다.
>
> **목적**: 기존 PaddleOCR 기반 수량 인식을 YOLO 기반 digit detection 방식으로 전환하기 위한 A/B 테스트 구현 레퍼런스.

---

## 1. 핵심 아이디어: OCR이 아니라 Object Detection

기존 OCR 방식(PaddleOCR, Tesseract 등)은 **텍스트 인식** 파이프라인을 거칩니다:
`Detection → Recognition → Post-processing`

이 프로젝트의 접근법은 근본적으로 다릅니다:
- 각 숫자(0~9)와 접두 문자("x")를 **개별 객체(object)로 탐지**
- 탐지된 객체들을 **x좌표 순으로 정렬**하여 문자열 조합
- 즉, YOLO의 bounding box detection을 글자 단위로 적용한 것

### 왜 이 방식이 게임 UI 숫자 인식에 유리한가

| 항목 | 전통 OCR (PaddleOCR 등) | YOLO Digit Detection |
|------|------------------------|---------------------|
| 글자 분할 | 텍스트 라인 → 글자 분할 필요 | 각 글자가 독립 객체, 분할 불필요 |
| 학습 데이터 | 범용 텍스트 학습 → 게임 폰트에 약할 수 있음 | 대상 게임 폰트로 직접 학습 가능 |
| 배경 간섭 | 복잡한 게임 배경에 취약 | confidence threshold로 필터링 |
| 겹치는 글자 | 분할 오류 발생 가능 | NMS + IoU로 자연스럽게 처리 |
| 추론 속도 | Detection + Recognition 2단계 | Detection 1단계로 끝 |
| confidence | 인식 결과 전체에 대한 confidence만 제공 | 글자별 개별 confidence 제공 |

---

## 2. 모델 스펙

### 2.1 모델 파일

```
YOLO26m_BA_AUTO_CAL_digt_v1.onnx
```

- **형식**: ONNX (Open Neural Network Exchange)
- **추정 아키텍처**: Ultralytics YOLO 계열 (YOLOv8/v11의 커스텀 학습 버전)
  - 근거: Ultralytics 표준 letterbox padding 값(114) 사용, 출력 텐서 형식 일치
- **모델 크기**: "m" (medium) 변형 — 속도와 정확도의 균형

### 2.2 클래스 정의 (11개)

```python
# class_id → 문자 매핑
CLASS_MAP = {
    0: "x",   # 곱하기 기호 (수량 접두사: "x123")
    1: "0",
    2: "1",
    3: "2",
    4: "3",
    5: "4",
    6: "5",
    7: "6",
    8: "7",
    9: "8",
    10: "9",
}
# 규칙: class_id == 0 → "x", 그 외 → str(class_id - 1)
```

### 2.3 입출력 형식

**입력**:
- Shape: `[1, 3, H, W]` (batch=1, RGB, height, width)
- 기본 입력 크기: `160x160` (ONNX 메타데이터에서 동적으로 읽음)
- 값 범위: `0.0 ~ 1.0` (float32, /255 정규화)

**출력**:
- Shape: `[1, N, 6]` 또는 `[N, 6]`
- 각 row: `[x1, y1, x2, y2, confidence, class_id]`
- 좌표: 입력 이미지 기준 픽셀 좌표 (letterbox 적용 후)

---

## 3. 전처리 파이프라인

### 3.1 수량 영역(count region) 크롭

게임 인벤토리 슬롯에서 수량이 표시되는 영역만 잘라냅니다:

```python
# 슬롯 전체 영역 대비 수량 표시 위치 (비율)
count_box_ratio = (0.34, 0.65, 0.98, 0.98)
# → 슬롯의 우하단 34~98%(x), 65~98%(y) 영역

# 크롭된 이미지를 160x64 크기로 letterbox 리사이즈
export_count_size = (64, 160)  # (height, width)
```

### 3.2 Letterbox 리사이즈 (외부, 첫 번째)

모델에 입력하기 전, 크롭된 이미지를 고정 크기로 리사이즈합니다. 비율을 유지하면서 빈 공간은 **흰색(255)**으로 채웁니다:

```python
def resize_with_letterbox(img, target_size, bg_value=255):
    """
    target_size: (height, width) = (64, 160)
    bg_value: 255 (흰색 배경)
    """
    th, tw = target_size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((th, tw, 3), bg_value, dtype=np.uint8)
    x = (tw - nw) // 2
    y = (th - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas
```

### 3.3 YOLO 내부 Letterbox (두 번째)

YOLO 모델 입력 크기에 맞추는 두 번째 letterbox. 이때 패딩 색상은 **Ultralytics 표준값 114**입니다:

```python
ULTRALYTICS_LETTERBOX_PAD = 114

def yolo_letterbox(image_bgr, input_width, input_height):
    """모델 입력 크기에 맞추는 letterbox (회색 패딩)"""
    src_h, src_w = image_bgr.shape[:2]
    scale = min(input_width / src_w, input_height / src_h)
    resized_w = max(1, round(src_w * scale))
    resized_h = max(1, round(src_h * scale))
    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    offset_x = (input_width - resized_w) // 2
    offset_y = (input_height - resized_h) // 2
    canvas[offset_y:offset_y+resized_h, offset_x:offset_x+resized_w] = resized
    return canvas
```

### 3.4 텐서 변환

```python
def preprocess(image_bgr):
    canvas = yolo_letterbox(image_bgr, input_width, input_height)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))       # HWC → CHW
    return np.expand_dims(chw, axis=0)                # → NCHW [1,3,H,W]
```

---

## 4. 후처리 파이프라인

### 4.1 출력 디코딩

```python
# ONNX 출력: shape [1, N, 6] 또는 [N, 6]
# 각 row: [x1, y1, x2, y2, confidence, class_id]

CONF_THRESHOLD = 0.25
VALID_CLASS_IDS = set(range(11))  # 0~10

def decode_output(output):
    rows = np.asarray(output, dtype=np.float32)
    if rows.ndim == 3:
        rows = rows[0]  # batch 차원 제거

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
        if x2 <= x1 or y2 <= y1:  # 유효하지 않은 박스 제거
            continue

        candidates.append(row)
    return candidates
```

### 4.2 NMS (Non-Maximum Suppression)

```python
NMS_IOU_THRESHOLD = 0.70

def nms(rows, iou_threshold=0.70):
    """confidence 높은 순으로 정렬, IoU가 threshold 이상인 중복 제거"""
    ordered = sorted(rows, key=lambda r: float(r[4]), reverse=True)
    kept = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [other for other in ordered
                   if compute_iou(current, other) < iou_threshold]
    return kept
```

### 4.3 추가 중복 제거

NMS 이후에도 남을 수 있는 중복을 한 번 더 제거합니다:

```python
DUPLICATE_IOU_THRESHOLD = 0.80

def deduplicate(rows, iou_threshold=0.80):
    """더 높은 IoU threshold로 한 번 더 중복 제거"""
    kept = []
    for row in sorted(rows, key=lambda r: float(r[4]), reverse=True):
        if not any(compute_iou(existing, row) >= iou_threshold for existing in kept):
            kept.append(row)
    return kept
```

### 4.4 IoU 계산

```python
def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = max(area_a + area_b - inter_area, 1e-12)
    return inter_area / union
```

### 4.5 숫자 조합 및 검증

```python
import re

COUNT_PATTERN = re.compile(r"^x([0-9]{1,4})$")

def assemble_digits(detections):
    """
    탐지된 digit들을 x좌표 순으로 정렬하여 문자열 조합.
    예: [x, 1, 2, 3] → "x123" → 123
    """
    # x좌표 중심값 기준 정렬
    ordered = sorted(detections, key=lambda d: (d.x_center, d.y_center))
    raw_text = "".join(d.digit for d in ordered)  # "x123"

    # 정규식으로 strict 검증: "x" + 1~4자리 숫자만 허용
    match = COUNT_PATTERN.fullmatch(raw_text)
    if not match:
        return None, raw_text, 0.0

    value = int(match.group(1))  # 123
    avg_confidence = sum(d.confidence for d in ordered) / len(ordered)
    return value, raw_text, avg_confidence
```

---

## 5. 아키텍처 설계 패턴

### 5.1 추상 Backend 인터페이스

기존 OCR과 YOLO를 A/B 테스트하려면, 공통 인터페이스를 정의하세요:

```python
# count_ocr_backend.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class CountOcrResult:
    attempted: bool         # OCR 시도 여부
    backend: str            # 사용된 백엔드 이름
    text: str               # 원본 인식 텍스트 ("x123")
    normalized_text: str    # 정규화된 숫자 ("123")
    value: Optional[int]    # 파싱된 정수값 (123) 또는 None
    confidence: Optional[float]  # 신뢰도 (0.0~1.0)
    error: Optional[str]    # 에러 메시지

class CountOcrBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...

    @abstractmethod
    def warmup(self) -> None:
        """첫 추론 전 모델 로딩 및 워밍업"""
        ...

    @abstractmethod
    def recognize_count(self, image_bgr) -> CountOcrResult:
        """BGR 이미지를 받아 수량 인식 결과 반환"""
        ...
```

### 5.2 PaddleOCR Backend (기존)

```python
class PaddleOcrBackend(CountOcrBackend):
    @property
    def backend_name(self) -> str:
        return "paddleocr_v1"

    def warmup(self) -> None:
        # PaddleOCR 초기화
        ...

    def recognize_count(self, image_bgr) -> CountOcrResult:
        # 기존 PaddleOCR 로직
        ...
```

### 5.3 YOLO Backend (신규)

```python
class YoloCountOcrBackend(CountOcrBackend):
    _backend_prefix = "yolo_digit_v1"

    def __init__(self, model_path: str):
        self._model_path = Path(model_path)
        self._session = None  # lazy init

    @property
    def backend_name(self) -> str:
        return self._backend_prefix

    def warmup(self) -> None:
        self._get_session()
        # 더미 이미지로 첫 추론 실행
        dummy = np.full((64, 160, 3), 255, dtype=np.uint8)
        self.recognize_count(dummy)

    def recognize_count(self, image_bgr) -> CountOcrResult:
        # 위 섹션 3~4의 파이프라인 구현
        ...
```

### 5.4 A/B 테스트 팩토리

```python
def build_count_ocr_backend(backend_type: str = "yolo") -> CountOcrBackend:
    if backend_type == "yolo":
        return YoloCountOcrBackend(model_path="path/to/model.onnx")
    elif backend_type == "paddle":
        return PaddleOcrBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_type}")

# A/B 비교 실행
def compare_backends(image_bgr):
    yolo_result = build_count_ocr_backend("yolo").recognize_count(image_bgr)
    paddle_result = build_count_ocr_backend("paddle").recognize_count(image_bgr)
    return {
        "yolo": {"value": yolo_result.value, "confidence": yolo_result.confidence},
        "paddle": {"value": paddle_result.value, "confidence": paddle_result.confidence},
    }
```

---

## 6. ONNX Runtime 설정

### 6.1 의존성

```
onnxruntime>=1.16.0        # CPU 전용
# 또는
onnxruntime-directml>=1.16.0  # Windows GPU (DirectML)
# 또는
onnxruntime-gpu>=1.16.0    # CUDA GPU
```

### 6.2 세션 생성

```python
import onnxruntime as ort

def create_onnx_session(model_path: str, device: str = "auto"):
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # WARNING만 출력

    available = set(ort.get_available_providers())

    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device in ("directml", "gpu"):
        if "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    else:  # auto
        if "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )
    return session
```

### 6.3 싱글톤 캐싱

원본 프로젝트에서는 모델 세션을 클래스 변수로 캐싱하여 재사용합니다:

```python
import threading

class YoloCountOcrBackend(CountOcrBackend):
    _instance_lock = threading.Lock()
    _session = None
    _warmup_done = False

    def _get_session(self):
        if type(self)._session is not None:
            return type(self)._session

        with type(self)._instance_lock:
            if type(self)._session is None:
                type(self)._session = create_onnx_session(self._model_path)
        return type(self)._session
```

---

## 7. YOLO 모델 학습 가이드 (직접 학습 시)

이 프로젝트의 모델은 이미 학습된 ONNX 파일로 제공되지만, 자체 데이터로 학습하려면:

### 7.1 데이터셋 준비

```yaml
# dataset.yaml
path: ./digit_dataset
train: images/train
val: images/val

names:
  0: x        # 곱하기 기호
  1: digit_0  # 숫자 0
  2: digit_1  # 숫자 1
  3: digit_2
  4: digit_3
  5: digit_4
  6: digit_5
  7: digit_6
  8: digit_7
  9: digit_8
  10: digit_9
```

YOLO 형식 라벨 (각 이미지당 `.txt` 파일):
```
# class_id  x_center  y_center  width  height  (모두 0~1 정규화)
0 0.12 0.50 0.08 0.60
2 0.25 0.50 0.07 0.58
3 0.38 0.50 0.07 0.58
```

### 7.2 학습 (Ultralytics)

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # 또는 yolov8m.pt
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=160,     # 입력 크기 (작은 영역이므로 크게 할 필요 없음)
    batch=32,
    name="digit_ocr_v1",
)
```

### 7.3 ONNX Export

```python
model = YOLO("runs/detect/digit_ocr_v1/weights/best.pt")
model.export(format="onnx", imgsz=160, simplify=True)
```

---

## 8. 주의사항 및 팁

### 8.1 Letterbox가 2중으로 적용됨

원본 코드에서는 letterbox가 **두 번** 적용됩니다:
1. `resize_with_letterbox()`: 크롭 이미지 → 160x64, 흰색(255) 패딩
2. `_YoloCountOcrRunner._letterbox()`: 160x64 → 모델 입력 크기, 회색(114) 패딩

구현 시 이 점을 주의하세요. 모델이 특정 letterbox 방식으로 학습되었다면 동일한 전처리를 적용해야 합니다.

### 8.2 정렬 기준

숫자를 조합할 때 **x좌표 중심값 → y좌표 중심값 → class_id** 순으로 정렬합니다.
게임 UI에서 수량은 항상 한 줄로 표시되므로, x좌표만으로도 충분하지만 y좌표를 보조 키로 사용합니다.

```python
ordered = sorted(detections, key=lambda d: (d.x_center, d.y_center, d.class_id))
```

### 8.3 수량 텍스트 검증 규칙

```python
# 유효한 패턴: "x" + 1~4자리 숫자
# 예: "x1", "x42", "x999", "x1234"
# 무효: "123" (x 없음), "x" (숫자 없음), "x12345" (5자리 이상)
COUNT_PATTERN = re.compile(r"^x([0-9]{1,4})$")
```

이 패턴은 블루아카이브 게임의 인벤토리 수량 표시 형식에 맞춘 것입니다.
다른 게임/앱에 적용할 때는 패턴을 조정하세요.

### 8.4 Confidence Threshold 튜닝

```python
CONF_THRESHOLD = 0.25     # 낮을수록 recall 증가, precision 감소
NMS_IOU_THRESHOLD = 0.70  # 겹치는 박스 제거 기준
DUPLICATE_IOU_THRESHOLD = 0.80  # 추가 중복 제거 (더 엄격)
```

### 8.5 에러 메시지 구조

원본 프로젝트의 에러 처리 패턴:
- 입력 이미지 비어있음 → `attempted=False`
- 모델 추론 실패 → `attempted=True, error="..."` 
- 숫자 미감지 → `attempted=True, confidence=0.0`
- 패턴 불일치 → `attempted=True, value=None, error="패턴 불일치"`

---

## 9. 파일 구조 요약

구현 시 참고할 원본 프로젝트의 파일 구조:

```
ba_inventory_studio_robust/
  count_ocr_backend.py          # 추상 인터페이스 + CountOcrResult
  yolo_count_ocr_backend.py     # YOLO 구현체 (핵심)
  onnx_backend_common.py        # ONNX 세션 생성, 경로 해석
  ba_inventory_studio_core.py   # 이미지 전처리, letterbox, 슬롯 탐지
  session_bridge.py             # 상위 파이프라인 (OCR 호출 통합)
  icon_classifier_backend.py    # 아이콘 분류 (별도 모델, OCR과 무관)

classifier/
  YOLO26m_BA_AUTO_CAL_digt_v1.onnx   # YOLO digit detection 모델
```

---

## 10. 빠른 구현 체크리스트

A/B 테스트를 위한 최소 구현 순서:

1. [ ] `onnxruntime` (또는 `onnxruntime-gpu`) 패키지 설치
2. [ ] `CountOcrResult` 데이터 클래스 정의
3. [ ] `CountOcrBackend` 추상 클래스 정의
4. [ ] 기존 PaddleOCR 로직을 `PaddleOcrBackend`로 래핑
5. [ ] `YoloCountOcrBackend` 구현:
   - ONNX 세션 로딩 (싱글톤)
   - letterbox 전처리 (114 패딩)
   - BGR→RGB→float32 /255→CHW→NCHW 변환
   - 출력 디코딩 (conf threshold 0.25)
   - NMS (IoU 0.70) + 중복 제거 (IoU 0.80)
   - x좌표 순 정렬 → 문자열 조합 → 정규식 검증
6. [ ] 팩토리 함수로 backend 전환 가능하게 구성
7. [ ] 동일 이미지셋으로 양쪽 결과 비교 로깅
