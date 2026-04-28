# Pipeline Optimization & Progressive UI Design

## Goal

Reduce image analysis time by minimizing icon matching, and show results progressively so users can start reviewing immediately.

## Problem

Current pipeline matches every cell twice (once in `find_start_cell`, once in `process_single_image`), each match comparing against all reference images via template matching + histogram. This is the primary bottleneck.

## Design

### 1. Single-Pass Pipeline

Replace the two-phase approach (find_start → process) with a single pass:

1. Iterate cells top-left to bottom-right
2. Match each cell's icon against references
3. Skip untrackable items (not in `item_order`)
4. Once **7 trackable items** are matched, run consensus voting to determine start index + direction (forward/reverse)
5. For cells already matched in step 2-4, perform OCR to get quantities
6. For remaining cells, **skip matching entirely** — assign material_id by walking `item_order` from the determined position, OCR only
7. When walking `item_order`, **stop at null (dump) slots** — these mark the boundary of trackable items

### 2. Progressive UI

Replace the blocking "분석 중..." flow with a streaming approach:

**Analysis Page (new state within existing flow):**
- Top: progress bar showing `분석 중 1/n`
- Bottom: placeholder text "분석 중입니다"

**As items are processed:**
- Worker emits each item result as it's ready
- UI checks if item needs review (low confidence / large deviation)
- If review needed: replace placeholder with review UI immediately
- User can review items while analysis continues in background

**On completion:**
- Show result page with JSON output
- `QTextEdit` already supports text selection (Ctrl+A, drag) — ensure `setReadOnly(True)` preserves this (it does in Qt)
- Keep existing copy button

### 3. Worker Signals

Current `AnalyzeWorker` emits only `finished(results, cell_images)`.

New signals:
- `progress(current_image: int, total_images: int)` — progress bar update
- `item_ready(material_id: str, qty: int, confidence: float, cell_image: ndarray)` — single item result
- `finished()` — all images done (no payload; results accumulated via `item_ready`)

### 4. item_order Dump Boundary

`item_order` structure: `[null, null, ..., trackable1, trackable2, ..., trackableN, null(dump), null, ...]`

- Leading nulls: cells before first trackable item (skip during matching phase)
- Trailing nulls (dump): once the walker hits null after trackable items, stop processing that image
- This prevents assigning material_ids to non-trackable items at the end of the inventory
