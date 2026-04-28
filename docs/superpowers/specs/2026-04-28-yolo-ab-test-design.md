# YOLO Digit Detection A/B Test Design

> **Date**: 2026-04-28
> **Goal**: EasyOCR과 YOLO digit detection 모델의 정확도를 동일 이미지셋으로 비교하여 전환 여부를 판단한다.

---

## 1. 배경

- 현재 `src/ocr_reader.py`에서 EasyOCR로 수량 텍스트를 인식
- Group 1 (일반 아이템): 99% 정확도, Group 2 (장비 아이템): 81% 정확도
- "블라리 인랭 계산기" 프로젝트의 `YOLO26m_BA_AUTO_CAL_digt_v1.onnx` 모델이 블루아카이브 게임 폰트에 특화되어 있어 성능 향상 가능성 있음
- 레퍼런스: `docs/YOLO_DIGIT_OCR_REFERENCE.md`

## 2. 접근법

**접근법 2 (최소 변경, 비교 전용)** 채택:
- 기존 코드를 변경하지 않음
- YOLO 리더를 별도 파일로 구현
- 비교 전용 스크립트에서 두 백엔드를 동시에 실행하고 결과를 비교 리포트로 출력
- YOLO가 우수하다고 판명되면 추상 Backend 인터페이스로 정식 통합 (별도 계획)

## 3. 파일 구조

```
src/
  ocr_reader.py              # 기존 EasyOCR (변경 없음)
  yolo_ocr_reader.py         # 신규 - YOLO digit detection
  compare_backends.py        # 신규 - A/B 비교 스크립트
models/
  YOLO26m_BA_AUTO_CAL_digt_v1.onnx   # docs/에서 이동
```

변경 파일: `requirements.txt` (onnxruntime 추가)
신규 파일: `src/yolo_ocr_reader.py`, `src/compare_backends.py`
이동 파일: ONNX 모델 → `models/`

## 4. YoloOcrReader 클래스 설계

### 4.1 인터페이스

```python
class YoloOcrReader:
    def __init__(self, model_path: str = "models/YOLO26m_BA_AUTO_CAL_digt_v1.onnx"):
        ...

    def read_quantity(self, image: np.ndarray) -> tuple[int, float] | None:
        """셀의 텍스트 영역 이미지에서 수량을 읽는다.
        Returns: (수량, confidence) 또는 None
        """
```

기존 `OcrReader.read_quantity()`와 동일한 입력(crop_text_region 출력)을 받되, confidence를 함께 반환한다.

### 4.2 전처리 파이프라인

1. 입력 이미지(BGR, crop_text_region 출력)를 160x64로 letterbox 리사이즈 (흰색 255 패딩)
2. 모델 입력 크기로 두 번째 letterbox (Ultralytics 표준 114 패딩)
3. BGR → RGB → float32 /255 → HWC→CHW → NCHW 변환

### 4.3 후처리 파이프라인

1. 출력 디코딩: `[x1, y1, x2, y2, confidence, class_id]`
2. Confidence 필터: threshold 0.25
3. NMS: IoU threshold 0.70
4. 추가 중복 제거: IoU threshold 0.80
5. x좌표 중심값 기준 정렬 → 문자열 조합
6. 정규식 검증: `r"^x([0-9]{1,4})$"`
7. 평균 confidence 계산

### 4.4 클래스 매핑

```python
CLASS_MAP = {0: "x", 1: "0", 2: "1", 3: "2", 4: "3", 5: "4",
             6: "5", 7: "6", 8: "7", 9: "8", 10: "9"}
```

### 4.5 ONNX 세션

- `onnxruntime` CPU 전용 (`CPUExecutionProvider`)
- 세션 싱글톤 (thread-safe, threading.Lock)
- 모델 입력 크기는 ONNX 메타데이터에서 동적으로 읽음

## 5. compare_backends.py 비교 스크립트

### 5.1 실행 방법

```bash
python -m src.compare_backends \
  --images docs/input-image/01.png docs/input-image/02.png ... \
  --order item_order.json \
  --answer answer.json
```

### 5.2 처리 흐름

1. 각 이미지에 대해 `detect_cells()` → `crop_text_region()` 실행 (공통)
2. 동일한 크롭 이미지를 EasyOCR과 YOLO 양쪽에 전달
3. `answer.json` 기준으로 정답 비교
4. 결과를 테이블로 출력

### 5.3 출력 형식

```
=== Backend Comparison Report ===
Images: 10 files, 287 items

Backend      Correct  Total  Accuracy
-----------  -------  -----  --------
EasyOCR      270      287    94.1%
YOLO         282      287    98.3%

=== Group Breakdown ===
Group          EasyOCR    YOLO
-------------- --------   --------
General(01-07) 204/206    205/206
Equip(10-12)   66/81      77/81

=== Disagreements ===
Item     Answer  EasyOCR       YOLO
T3_Hat   454     445 [WRONG]   454 [OK]
...
```

### 5.4 의존 모듈 (기존, 변경 없음)

- `src.grid_detector`: detect_cells, crop_text_region
- `src.ocr_reader`: OcrReader
- `src.item_matcher`: ItemMatcher
- `src.accuracy_checker`: 정확도 계산 유틸

## 6. 의존성

`requirements.txt`에 추가:
```
onnxruntime>=1.16.0
```

## 7. 입력 이미지 호환성

현재 프로젝트는 그리드 영역만 캡쳐한 스크린샷을 사용하고, 레퍼런스 프로젝트는 전체 게임 화면을 사용했으나 문제없음:
- 두 프로젝트 모두 개별 셀에서 수량 텍스트 영역만 크롭한 이미지를 OCR에 전달
- 현재 프로젝트: `crop_text_region()` → 하단 22%, 우측 65%
- 레퍼런스: `count_box_ratio` → 하단 33%, 우측 64%
- 비율 차이는 YOLO의 letterbox 전처리가 흡수

## 8. 성공 기준

- YOLO가 전체 이미지셋에서 EasyOCR 대비 동등 이상의 정확도를 보이면 정식 통합 검토
- 특히 Group 2 (장비 아이템, 현재 81%)에서 유의미한 개선이 있으면 전환 추천

## 9. 후속 작업 (본 스펙 범위 밖)

YOLO가 우수하다고 판명될 경우 추상 Backend 인터페이스 도입으로 정식 통합.
상세 계획은 별도 기록됨.
