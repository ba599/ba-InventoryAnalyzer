# ba-InventoryAnalyzer

블루 아카이브(Blue Archive) 인벤토리 스크린샷에서 아이템 수량을 자동으로 읽어 [Justin Planner](https://justin163.com/)의 JSON 데이터를 업데이트하는 도구입니다.

## 다운로드

[Releases](../../releases) 페이지에서 최신 `ba-InventoryAnalyzer-win64.zip`을 받으세요.

## 사용법

### 1. 준비

1. **zip 파일 압축 해제** — 원하는 위치에 풀어주세요
2. **Justin Planner JSON 내보내기** — [justin163.com](https://justin163.com/)에서 계정 데이터를 JSON으로 내보냅니다
3. **인벤토리 스크린샷 준비** — 블루 아카이브 인벤토리 화면을 캡처합니다

### 2. 실행 (GUI)

`ba-InventoryAnalyzer.exe`를 실행합니다.

> 첫 실행 시 EasyOCR 모델을 다운로드하므로 시간이 걸릴 수 있습니다.

#### 스크린샷 붙여넣기

- 인벤토리 스크린샷을 클립보드에 복사한 뒤 앱에서 `Ctrl+V`로 붙여넣기
- 여러 장을 연속으로 붙여넣을 수 있습니다

#### JSON 입력

- Justin Planner에서 내보낸 JSON 텍스트를 하단 텍스트 박스에 붙여넣기

#### 분석 및 결과

- **Analyze** 버튼을 누르면 OCR 분석이 시작됩니다
- 신뢰도가 낮은 항목은 `[REVIEW]` 표시와 함께 수동 검수를 요청합니다
- 검수 완료 후 업데이트된 JSON을 복사하여 Justin Planner에 다시 가져오면 됩니다

### 3. 실행 (CLI)

Python 환경이 있다면 CLI로도 사용할 수 있습니다.

```bash
pip install -r requirements.txt

python -m src.main \
  --images screenshots/*.png \
  --json justin163.json \
  --output result.json
```

#### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--images` | (필수) | 스크린샷 경로 (복수 가능) |
| `--json` | (필수) | Justin Planner JSON 경로 |
| `--order` | `item_order.json` | 아이템 순서 JSON |
| `--start-id` | (없음) | 첫 아이템 ID 직접 지정 |
| `--refs` | `references` | 참조 이미지 디렉토리 |
| `--output` | (입력 JSON 덮어쓰기) | 출력 JSON 경로 |
| `--check` | (없음) | 정답 JSON 경로 (정확도 검증) |
| `--confidence-threshold` | `0.7` | 수동 검수 플래그 임계값 |

## 수동 검수 (Manual Review)

OCR 결과 중 검수가 필요한 항목을 `[REVIEW]`로 표시합니다:

- **Low confidence** — OCR 신뢰도가 threshold 미만
- **Large deviation** — 기존 JSON 값 대비 100 이상 차이

```
=== Manual Review Required: 3 items ===
  [REVIEW] T2_Bag: 1693 - low confidence (0.53)
  [REVIEW] T8_Charm: 520 - deviation -1000 (was 1520)
  [REVIEW] T6_Watch: 112 - low confidence (0.61), deviation -959 (was 1071)
```

## 알려진 한계

- OCR 정확도는 약 90~92% 수준입니다 (italic deskew 적용 후)
- 첫 자리 누락, 끝자리 ±2 오차 등의 패턴이 있을 수 있습니다
- 스크린샷 상단이 잘린 아이템은 잘못 인식될 수 있습니다

## 삭제 방법

1. 압축 해제한 `ba-InventoryAnalyzer` 폴더를 통째로 삭제하면 됩니다
2. EasyOCR이 다운로드한 모델 파일도 삭제하려면 `C:\Users\<사용자명>\.EasyOCR` 폴더를 삭제하세요

레지스트리나 AppData에 별도로 저장하는 데이터는 없습니다.

## 기술 스택

- Python 3.13, OpenCV, EasyOCR, PySide6
- PyInstaller (배포용 빌드)

## 라이선스

이 프로젝트는 AI(Claude)의 도움으로 생성되었으며, [MIT License](LICENSE)로 배포됩니다.
