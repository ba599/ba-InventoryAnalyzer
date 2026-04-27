# OCR 정확도 개선 설계

## 배경

Blue Archive 인벤토리 스크린샷 분석기의 OCR 정확도가 장비 아이템(그룹2: 10-12.png, T2_Hat~T10_Necklace)에서 불충분하다.

- **그룹1 (일반 아이템)**: EasyOCR 99% 정확 (206개 중 204개)
- **그룹2 (장비 아이템)**: EasyOCR 81% (81개 중 66개), PaddleOCR 76% (81개 중 ~62개)

### 오류 원인 분석

장비 아이템의 특성이 OCR을 어렵게 만든다:
1. **파란/그래디언트 배경**: 텍스트와 배경 간 대비 부족
2. **작은 폰트**: 수량 텍스트가 셀 하단 22% × 우측 65% 영역에 있어 매우 작음
3. **현재 전처리 부족**: 높이 40px 미만일 때 upscale만 수행, 색상/대비 처리 없음

### 오류 패턴

- 첫 자리 누락: 1326→326, 1293→293
- 숫자 오인식: 1↔7 혼동 (131→731)
- 작은 텍스트 인식 실패: OCR 결과 None

## 목표

1. 장비 아이템 OCR 정확도를 81%→90%+ 로 개선
2. 불확실한 결과를 사용자에게 플래그하여 수동 검수 가능하게 함
3. `answer.json` ground truth를 활용한 자동 정확도 검증 스크립트 제공

## 설계

### 1. 전처리 강화 (`ocr_reader.py`)

`preprocess()` 메서드를 다음과 같이 강화한다:

- **upscale 최소 크기 증가**: 40px → 80px. 장비 텍스트가 작아서 40px로는 부족.
- **색상 기반 텍스트 마스크 (primary)**: 폰트 색상 `#2D4663` (BGR: 99,70,45) ±20 tolerance로 텍스트 픽셀만 추출하여 이진 마스크 생성. 색상 분석 결과 텍스트 색이 좁은 범위에 집중되어 있고 배경(흰색 R:240+)과의 거리가 매우 커서 오탐 위험 거의 없음.
  - `cv2.inRange(image, lower_bound, upper_bound)` → 마스크 → 반전 (검은 배경에 흰 텍스트 → 흰 배경에 검은 텍스트)
  - 안티앨리어싱 픽셀도 포함하기 위해 tolerance ±30까지 확장 가능 (구현 시 튜닝)
- **Otsu 이진화 (fallback)**: 색상 마스크가 텍스트를 충분히 캡처하지 못할 경우 (마스크 내 흰 픽셀 비율이 너무 낮을 때) grayscale + Otsu로 fallback.

처리 순서: 원본 → 색상 마스크 (또는 Otsu fallback) → upscale (80px 미만 시)

### 2. Confidence 반환 (`ocr_reader.py`)

`read_quantity()` 반환 타입을 변경한다:

- **기존**: `int | None`
- **변경**: `tuple[int, float] | None` — `(수량, confidence)`

Confidence 소스:
- **EasyOCR**: `readtext(detail=1)` → `(bbox, text, confidence)`. 현재 `detail=0`을 `detail=1`로 변경.
- **PaddleOCR**: `line[1][1]`이 confidence score.

### 3. 출력 형식 변경 (`main.py`)

`process_screenshots_sequential()`과 `process_screenshot_matched()`에서:

- 모든 항목에 confidence 표시: `T3_Hat: 454 (0.92)`
- confidence < 0.7 인 항목에 `[CHECK]` 마킹: `[CHECK] T2_Charm: 1421 (0.45)`
- confidence threshold는 CLI 인자 `--confidence-threshold`로 조정 가능 (기본값 0.7)
- 최종 요약에 검수 필요 항목 수 표시

반환 타입도 변경: `dict[str, int]` → `dict[str, tuple[int, float]]`

### 4. 정확도 검증 스크립트 (`src/accuracy_checker.py`)

`answer.json`과 OCR 결과를 비교하는 독립 스크립트:

- 입력: OCR 결과 dict, answer.json 경로
- 비교: 각 material_id의 OCR 결과 vs 정답
- 출력: 정확/오류 수, 오류 목록 (material_id, OCR값, 정답값), 정확도 퍼센트
- main.py에 `--check` 플래그로 통합: 실행 후 자동으로 answer.json과 비교

## 변경 대상 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/ocr_reader.py` | 전처리 강화 (색상 마스크 #2D4663 ±20, Otsu fallback, upscale 80px), confidence 반환 |
| `src/main.py` | confidence 표시, [CHECK] 플래그, --confidence-threshold 인자, --check 인자 |
| `src/accuracy_checker.py` | 신규 — answer.json 기반 정확도 비교 |
| `tests/test_ocr_reader.py` | 새 반환 타입에 맞게 테스트 수정 |
| `tests/test_accuracy_checker.py` | 신규 — accuracy_checker 테스트 |

## 변경하지 않는 파일

- `src/grid_detector.py` — 크롭 비율은 이미 최적화됨 (78/22, 우측 65%)
- `src/item_matcher.py` — 아이콘 매칭과 무관
- `src/json_updater.py` — JSON 업데이트 로직 변경 없음 (confidence는 저장하지 않음, 수량만 저장)
- `item_order.json` — 변경 없음

## 성공 기준

1. 장비 아이템 정확도 90%+ (81개 중 73개 이상 정확)
2. 일반 아이템 정확도 유지 (99%)
3. 오류 항목이 [CHECK]로 플래그됨 (false negative 최소화)
4. `--check` 플래그로 answer.json 대비 자동 검증 가능
