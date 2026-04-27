# Blue Archive Inventory Analyzer - Design Spec

## Overview

Blue Archive 게임 인벤토리 스크린샷에서 아이템 수량을 자동 인식하여 Justin Planner JSON의 `owned_materials`를 업데이트하는 CLI 도구.

## Workflow

```
Justin Planner에서 JSON export
  → 게임 인벤토리 스크린샷 캡쳐 (여러 장)
  → 앱이 아이템 수량 인식
  → JSON의 owned_materials 업데이트
  → Justin Planner에 JSON import
```

## 전제 조건

- 사용자는 모든 아이템을 소유한 상태 (미소유 아이템으로 인한 순서 건너뜀 없음)
- 게임 인벤토리의 "기본" 정렬 기준 사용
- 아이템 정렬 순서는 별도 설정 파일(`item_order.json`)로 사용자가 제공

## 입력

- **스크린샷**: 그리드 영역만 수동 캡쳐한 PNG 이미지들 (크기/위치 매번 다름)
- **JSON**: Justin Planner에서 export한 JSON 파일
- **아이템 순서**: material_id의 게임 내 정렬 순서 목록

## 핵심 전략: 첫 아이템 매칭 + 순서 기반 OCR

각 스크린샷에서:
1. 셀 패턴을 검출하여 개별 아이템 셀 위치를 찾음
2. 좌상단 첫 번째 유효 셀의 아이콘을 레퍼런스 DB와 템플릿 매칭
3. 매칭된 아이템 위치부터 정해진 순서대로 나머지 셀의 수량만 OCR
4. 여러 스크린샷 간 동일 material_id 중복 시 스킵

## 기술 스택

- **Python 3.10+**
- **OpenCV**: 셀 검출, 템플릿 매칭, 이미지 전처리
- **EasyOCR**: 수량 텍스트("x2375") 인식
- **레퍼런스 이미지**: SchaleDB에서 아이템 아이콘 다운로드

## 모듈 설계

### 프로젝트 구조

```
image-analyzer/
├── src/
│   ├── grid_detector.py    # 그리드 영역 + 셀 분할
│   ├── item_matcher.py     # 첫 아이템 템플릿 매칭
│   ├── ocr_reader.py       # 수량 OCR
│   ├── json_updater.py     # JSON 읽기/쓰기
│   ├── ref_builder.py      # 레퍼런스 이미지 생성기
│   └── main.py             # CLI 진입점
├── references/             # 레퍼런스 아이콘 이미지
├── item_order.json         # 아이템 정렬 순서 정의
├── docs/
└── requirements.txt
```

### 1. grid_detector.py — 셀 검출

**입력**: 그리드 영역 스크린샷 (크기 가변)
**출력**: 개별 셀의 좌표 목록 (행/열 정보 포함)

- 고정 좌표가 아닌 셀 패턴 기반 검출 (contour detection)
- 같은 행 판단: 상단 Y좌표 유사도 기준 그룹핑
- 반쪽 셀(가장자리 잘림) 판단: 정상 셀 대비 너비/높이 부족 시 스킵
- 6~7열 가변 대응
- 각 셀에서 두 영역 분리:
  - 상단 ~70%: 아이콘 영역 (매칭용)
  - 하단 ~30%: 수량 텍스트 영역 (OCR용)

### 2. item_matcher.py — 첫 아이템 매칭

**입력**: 셀 아이콘 크롭 이미지
**출력**: material_id

- 레퍼런스 이미지 DB (references/ 폴더)에서 260+개 아이콘 로드
- OpenCV 템플릿 매칭으로 가장 유사한 아이템 식별
- 크기 정규화 후 비교 (캡쳐 크기가 매번 다르므로)
- 추후 성능 필요 시 phash 1차 필터링 → 템플릿 매칭 2차 확인으로 최적화 가능

### 3. ocr_reader.py — 수량 OCR

**입력**: 수량 텍스트 영역 크롭 이미지
**출력**: 정수 (수량)

- EasyOCR로 텍스트 인식
- "x" 접두사 제거 후 숫자 파싱
- 이미지 전처리: 이진화, 대비 강화로 인식률 향상
- 인식 실패 시 해당 셀 스킵 및 경고 로그

### 4. json_updater.py — JSON 업데이트

**입력**: {material_id: quantity} 딕셔너리, 원본 JSON 경로
**출력**: 업데이트된 JSON 파일

- Justin Planner JSON 구조 유지하며 `owned_materials` 섹션만 업데이트
- 기존 키 구조 보존 (숫자 ID + 이름 기반 키)

### 5. main.py — CLI 진입점

**사용법** (초기):
```bash
python src/main.py --images screenshot1.png screenshot2.png --json justin163.json
```

**처리 흐름:**
1. 이미지 목록과 JSON 로드
2. 각 이미지에 대해:
   a. 그리드 검출 → 셀 분할
   b. 좌상단 셀 아이콘 → 레퍼런스 매칭 → 시작 material_id 결정
   c. 각 셀 수량 OCR
   d. 순서 목록 기반으로 material_id 매핑
3. 전체 결과 병합 (중복 제거)
4. JSON 업데이트 및 저장

## 중복 제거 전략

- 스크린샷 순서가 보장되지 않으므로, 각 스크린샷을 독립적으로 처리
- 같은 material_id가 여러 스크린샷에서 인식되면, 첫 번째 값 사용 (또는 경고)
- 순서 목록 기반이므로 한 스크린샷 내에서는 중복 발생하지 않음

## 레퍼런스 이미지 생성

SchaleDB 등 외부 아이콘은 투명 배경의 클린 이미지라 실제 인게임 캡쳐(셀 배경, 레어리티 프레임, 그라데이션 포함)와 차이가 커서 매칭 정확도가 낮음.

**대안: 인게임 캡쳐 기반 레퍼런스 생성기**

사용자가 인게임 인벤토리를 캡쳐하면, grid_detector를 재활용하여 각 셀의 아이콘 영역만 크롭하여 레퍼런스 이미지로 저장.

### ref_builder.py — 레퍼런스 이미지 생성기

**사용법:**
```bash
python src/ref_builder.py --images ref_capture1.png ref_capture2.png --start-id 100
```

**처리 흐름:**
1. 스크린샷에서 셀 검출 (grid_detector 재활용)
2. 각 셀에서 아이콘 영역 크롭 (수량 텍스트 부분 제외)
3. item_order.json 순서에 따라 `--start-id`부터 순차적으로 material_id 부여
4. `references/{material_id}.png`로 저장

**장점:**
- 레퍼런스와 실제 캡쳐가 동일한 시각적 조건 (배경, 프레임, 크기 비율)
- grid_detector 모듈을 공유하므로 추가 개발 최소화
- 게임 업데이트로 아이콘이 바뀌어도 재캡쳐로 즉시 대응 가능

**파일명 규칙:** `{material_id}.png` (예: `100.png`, `T8_Necklace.png`)

## 초기 개발 범위 (MVP)

1. 단일 스크린샷에서 셀 검출 + OCR 동작 확인
2. 레퍼런스 이미지 생성기(ref_builder) 구현 및 레퍼런스 이미지 생성
3. 첫 아이템 템플릿 매칭 확인
4. 여러 스크린샷 처리 + JSON 업데이트

## 향후 확장 (현재 범위 밖)

- GUI (클립보드 붙여넣기 지원)
- 미소유 아이템 대응 (모든 셀 매칭 방식 B로 전환)
- phash 기반 매칭 최적화
