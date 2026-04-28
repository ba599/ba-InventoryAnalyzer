import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from src.core.pipeline import process_image_streaming, process_all_images_streaming, CellResult, ImageProgress


def _make_cell_info(x, y, w, h, row, col):
    from src.grid_detector import CellInfo
    return CellInfo(x=x, y=y, w=w, h=h, row=row, col=col)


def test_process_image_streaming_returns_results():
    """process_image_streaming yields CellResult for each matched cell."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = [str(i) for i in range(100, 110)]  # 10 items

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    # 6 cells so odd indices (1,3,5) get matched
    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(6)]

    # Odd cells (1,3,5) match items 101,103,105 -> fwd offsets: 100,100,100
    mock_matcher.match_with_score.side_effect = [
        ("101", 0.9), ("103", 0.9), ("105", 0.9),
    ]

    # All 6 cells get OCR in phase 2
    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(6)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    assert len(results) == 6
    mids = {r.material_id for r in results}
    assert "100" in mids
    assert "101" in mids
    assert "105" in mids
    r100 = next(r for r in results if r.material_id == "100")
    assert r100.quantity == 0
    assert r100.confidence == 0.95


def test_process_image_streaming_no_match_returns_empty():
    """When no cell matches a reference, yield nothing."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    # Need at least 2 cells so odd index 1 exists
    cells = [_make_cell_info(0, 0, 80, 80, 0, 0), _make_cell_info(80, 0, 80, 80, 0, 1)]
    mock_matcher.match_with_score.return_value = (None, -1.0)

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    assert results == []


def test_process_image_streaming_stops_at_null_dump():
    """Null entries in item_order act as BREAK_POINT — processing stops."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = ["100", "101", None, "103", "104", "105"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    # 6 cells, odd indices (1,3,5) get matched
    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(6)]

    # Odd cells match: cell1->101(idx1), cell3->103(idx3), cell5->105(idx5)
    # fwd offsets: 1-1=0, 3-3=0, 5-5=0 -> consensus offset=0
    mock_matcher.match_with_score.side_effect = [
        ("101", 0.9), ("103", 0.9), ("105", 0.9),
    ]

    # Phase 2 walks from cell 0: item_order[0]="100", item_order[1]="101",
    # item_order[2]=None -> BREAK_POINT, stop
    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (200, 0.80),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    mids = [r.material_id for r in results]
    assert "100" in mids
    assert "101" in mids
    # Stops at BREAK_POINT — items after null are NOT processed
    assert "103" not in mids
    assert None not in mids


def test_process_image_streaming_only_matches_odd_cells():
    """Matcher is only called for odd-indexed cells, not all cells."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = [str(i) for i in range(100, 120)]  # 20 items

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(10)]

    # 5 odd cells (1,3,5,7,9) get matched
    mock_matcher.match_with_score.side_effect = [
        (str(100 + i), 0.9) for i in range(1, 10, 2)
    ]

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(10)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    assert len(results) == 10
    # Matcher called exactly 5 times (odd cells only)
    assert mock_matcher.match_with_score.call_count == 5
    # Reader called 10 times (all cells get OCR)
    assert mock_reader.read_quantity.call_count == 10


def test_process_image_streaming_stops_at_dump():
    """When item_order hits null (BREAK_POINT), stop yielding results."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    # 5 trackable, then BREAK_POINT (null), then more items
    item_order = ["100", "101", "102", "103", "104", None, "200", "201"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(8)]

    # Odd cells (1,3,5,7): cell1->101, cell3->103, cell5->??, cell7->??
    # For offset=100: cell1 -> idx 101 (101), cell3 -> idx 103 (103)
    # cell5 -> idx 105 = None (untrackable), cell7 -> idx 107 = "201"
    # fwd offsets: 1-1=100, 3-3=100, 7-7=100 -> consensus=100
    mock_matcher.match_with_score.side_effect = [
        ("101", 0.9), ("103", 0.9), ("200", 0.1), ("201", 0.9),
    ]

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(5)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    # Only 5 results — stopped at BREAK_POINT
    assert len(results) == 5
    result_mids = [r.material_id for r in results]
    assert "200" not in result_mids
    assert "201" not in result_mids


def test_process_image_streaming_skips_leading_untrackable():
    """Leading null slots in item_order are skipped, not treated as BREAK_POINT."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = [None, None, None, "100", "101", "102", "103", "104", "105", "106"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(10)]

    # Odd cells (1,3,5,7,9) matched. With offset=3 (leading nulls):
    # cell1->idx4="101", cell3->idx6="103", cell5->idx8="105", cell7->idx10=OOB
    # fwd: 4-1=3, 6-3=3, 8-5=3 -> consensus offset=3
    mock_matcher.match_with_score.side_effect = [
        ("101", 0.9), ("103", 0.9), ("105", 0.9),
        (None, -1.0), (None, -1.0),
    ]

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(7)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    # 7 trackable results (items 100-106), leading nulls skipped
    assert len(results) == 7
    result_mids = [r.material_id for r in results]
    assert "100" in result_mids
    assert "106" in result_mids


def test_process_all_images_streaming_yields_progress_and_results():
    """Iterating multiple images yields progress markers and cell results."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = [str(i) for i in range(100, 106)]  # 6 items

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(6)]

    # Each image: odd cells (1,3,5) matched -> offsets 100,100,100
    mock_matcher.match_with_score.side_effect = [
        ("101", 0.9), ("103", 0.9), ("105", 0.9),  # image 1
        ("101", 0.9), ("103", 0.9), ("105", 0.9),  # image 2
    ]

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(12)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        events = list(process_all_images_streaming(
            [fake_image, fake_image], item_order, mock_matcher, mock_reader
        ))

    # Filter progress events
    progress_events = [e for e in events if isinstance(e, ImageProgress)]
    cell_events = [e for e in events if isinstance(e, CellResult)]

    assert len(progress_events) == 2
    assert progress_events[0].current == 1
    assert progress_events[0].total == 2
    assert progress_events[1].current == 2

    # Second image should not re-yield already-seen material_ids
    assert len(cell_events) == 6  # only from first image (second has same items)
