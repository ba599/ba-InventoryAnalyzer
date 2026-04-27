# GUI Design Spec — Blue Archive Inventory Analyzer

## Overview

기존 CLI 앱에 웹(Flask)과 데스크톱(PySide6) GUI를 추가. 핵심 로직을 `core/`로 분리하여 CLI/웹/데스크톱이 공유.

## Architecture

```
src/
  core/
    pipeline.py      # process_image(img, item_order, refs) → results + review_items
    review.py         # 검수 판정 로직 (confidence, deviation)
  web/
    app.py            # Flask 앱
    static/           # CSS, JS
    templates/        # HTML (Jinja2)
  desktop/
    app.py            # PySide6 메인 윈도우
  grid_detector.py    # 기존 유지
  ocr_reader.py       # 기존 유지
  item_matcher.py     # 기존 유지
  json_updater.py     # 기존 유지
  main.py             # CLI (core.pipeline 호출로 변경)
```

### Core 분리

- `core/pipeline.py` — `main.py`의 `process_screenshots_sequential()` 에서 추출. 이미지 한 장을 받아 `{material_id: (qty, confidence)}` + `review_items` 반환.
- `core/review.py` — REVIEW 판정 로직. confidence < 0.7 또는 기존값 대비 ±100 이상 차이 시 플래그. GUI에서는 REVIEW 항목을 반영 보류하고 사용자 승인 후 반영.
- 기존 모듈은 변경 없이 그대로 사용.

## UI Flow (웹/데스크톱 공통)

### Step 1: 입력 화면

- **JSON 입력**: 텍스트 영역에 justin163.json 내용 붙여넣기
- **이미지 입력**: Ctrl+V로 스크린샷 붙여넣기 (여러 장)
  - 붙여넣을 때마다 썸네일 미리보기 추가
  - 삭제 가능
- **"분석 시작" 버튼** → 전체 이미지 처리 후 Step 2로

### Step 2: 검수 화면

REVIEW 항목이 없으면 스킵 → 바로 Step 3.

한 건씩 순차 검수:

```
┌─────────────────────────┐
│        3 / 18           │
├─────────────────────────┤
│                         │
│   [셀 크롭 이미지 확대]   │
│                         │
├─────────────────────────┤
│  T8_Charm               │
│  [  520  ]  ← 오토포커스, 전체선택 │
│  low confidence (0.53)  │
│                         │
│  Enter: 확정  /  Skip   │
└─────────────────────────┘
```

- **상단**: 진행률 표시 (예: `3 / 18`)
- **이미지**: 해당 셀을 크롭하여 확대 표시
- **키 이름**: material_id 표시
- **입력 필드**: OCR 값이 기본값, 오토포커스 + 전체선택 상태
- **REVIEW 사유**: "low confidence (0.53)" 또는 "was 1520" 등
- **Enter**: 현재 값 확정 → 다음 항목
- **Skip**: OCR 값 그대로 수용 → 다음 항목
- **마지막 항목 처리 후** → Step 3

### Step 3: 결과 화면

- 업데이트된 justin163.json 전체를 텍스트 박스에 표시
- **"복사" 버튼**: 원클릭 클립보드 복사
- 복사 완료 피드백 (버튼 텍스트 변경 등)

## 이미지 처리

- 이미지 순서 무관: 각 이미지마다 `item_matcher`로 시작점 독립 판별
- `--start-id` 직접 지정 모드 미지원 (자동 탐색만)
- 셀 크롭: 검수 화면에서 해당 셀의 bounding box 기준 크롭

## 기술 세부사항

### 웹 버전 (Flask)

- 클립보드 붙여넣기: JS `paste` 이벤트로 이미지 캡처 → FormData로 서버 전송
- 분석: 서버에서 `core/pipeline.py` 호출, 결과를 JSON으로 반환
- 검수 화면: SPA 스타일, JS에서 항목 순회, Enter/Skip 키보드 이벤트 처리
- 셀 크롭 이미지: 서버에서 크롭 후 base64로 전달
- 결과 복사: `navigator.clipboard.writeText()` API

### 데스크톱 버전 (PySide6)

- 클립보드 붙여넣기: `QApplication.clipboard().image()`
- 분석: `core/pipeline.py` 직접 호출
- 검수 화면: `QStackedWidget`, `QLineEdit` 오토포커스+전체선택
- 셀 크롭 이미지: `QLabel` + `QPixmap`
- 결과 복사: `QApplication.clipboard().setText()`

## 의존성

- 웹: Flask 추가
- 데스크톱: PySide6 추가
- 기존: opencv-python, easyocr, numpy, pytest 유지

## 기존 코드 변경

- `main.py` — 파이프라인 로직을 `core/pipeline.py`로 추출. CLI는 이를 호출하도록 변경. 외부 동작 변경 없음.

## 범위 외

- `.exe` 패키징 (PyInstaller)
- 설정 UI (threshold 등은 코드단에서 조정)
- `--start-id`, `--check` 등 고급 CLI 옵션의 GUI 지원
- `--confidence-threshold` UI 노출 (기본값 0.7 고정)
