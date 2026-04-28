import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from src.core.pipeline import process_image_streaming, process_all_images_streaming, CellResult, ImageProgress


def _make_cell_info(x, y, w, h, row, col):
    from src.grid_detector import CellInfo
    return CellInfo(x=x, y=y, w=w, h=h, row=row, col=col)


def test_process_image_streaming_returns_results():
    """process_image_streaming yields CellResult for each matched cell."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101", "102"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
    ]

    match_returns = [("100", 0.9), ("101", 0.9), ("102", 0.9)]
    mock_matcher.match_with_score.side_effect = match_returns

    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (200, 0.80),
        (100, 0.50),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    mids = {r.material_id for r in results}
    assert "100" in mids
    assert "101" in mids
    assert "102" in mids
    r100 = next(r for r in results if r.material_id == "100")
    assert r100.quantity == 500
    assert r100.confidence == 0.95


def test_process_image_streaming_no_match_returns_empty():
    """When no cell matches a reference, yield nothing."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(0, 0, 80, 80, 0, 0)]
    mock_matcher.match_with_score.return_value = (None, -1.0)

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    assert results == []


def test_process_image_streaming_skips_null_items():
    """Null entries in item_order are skipped."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", None, "102", "103", "104"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
        _make_cell_info(240, 0, 80, 80, 0, 3),
    ]

    match_returns = [("100", 0.9), ("100", 0.9), ("102", 0.9), ("103", 0.9)]
    mock_matcher.match_with_score.side_effect = match_returns

    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (200, 0.80),
        (300, 0.90),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    mids = [r.material_id for r in results]
    assert "100" in mids
    assert "102" in mids
    assert None not in mids


def test_process_image_streaming_matches_only_first_7_trackable():
    """After 7 trackable matches, remaining cells use index walking (no matcher calls)."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = [str(i) for i in range(100, 115)]  # 15 trackable items

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(10)]

    # First 7 cells match items 100-106 (consensus: forward, offset=100 in item_order)
    match_returns = [(str(100 + i), 0.9) for i in range(7)]
    mock_matcher.match_with_score.side_effect = match_returns

    # All 10 cells get OCR
    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(10)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    assert len(results) == 10
    # Matcher called exactly 7 times (not 10)
    assert mock_matcher.match_with_score.call_count == 7
    # Reader called 10 times (all cells get OCR)
    assert mock_reader.read_quantity.call_count == 10


def test_process_image_streaming_stops_at_dump():
    """When item_order hits null (dump), stop yielding results."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    # 5 trackable, then dump (null), then more items
    item_order = ["100", "101", "102", "103", "104", None, "200", "201"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(8)]

    # First 5 cells match trackable items 100-104; remaining 3 cells match
    # unknown items (not in item_order). Since we can't reach consensus_count=7
    # with only 5 trackable matches, the loop exhausts all cells.
    # Consensus from 5 matches (>=3) should still work.
    match_returns = [(str(100 + i), 0.9) for i in range(5)]
    match_returns += [("UNKNOWN_X", 0.9), ("UNKNOWN_Y", 0.9), ("UNKNOWN_Z", 0.9)]
    mock_matcher.match_with_score.side_effect = match_returns

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(5)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    # Only 5 results — stopped at dump
    assert len(results) == 5
    result_mids = [r.material_id for r in results]
    assert "200" not in result_mids
    assert "201" not in result_mids


def test_process_image_streaming_skips_leading_untrackable():
    """Cells with icons not in item_order are skipped during matching phase."""
    fake_image = np.zeros((200, 800, 3), dtype=np.uint8)
    item_order = [None, None, None, "100", "101", "102", "103", "104", "105", "106"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    # 10 cells: first 3 are untrackable, then 7 trackable
    cells = [_make_cell_info(i * 80, 0, 80, 80, 0, i) for i in range(10)]

    # First 3 cells match unknown IDs, next 7 match trackable items
    match_returns = [
        ("UNKNOWN_1", 0.9), ("UNKNOWN_2", 0.9), ("UNKNOWN_3", 0.9),
        ("100", 0.9), ("101", 0.9), ("102", 0.9), ("103", 0.9),
        ("104", 0.9), ("105", 0.9), ("106", 0.9),
    ]
    mock_matcher.match_with_score.side_effect = match_returns

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(7)]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results = list(process_image_streaming(
            fake_image, item_order, mock_matcher, mock_reader
        ))

    # 7 trackable results, not 10
    assert len(results) == 7
    result_mids = [r.material_id for r in results]
    assert "UNKNOWN_1" not in result_mids
    assert "100" in result_mids


def test_process_all_images_streaming_yields_progress_and_results():
    """Iterating multiple images yields progress markers and cell results."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101", "102"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
    ]

    match_returns = [(str(100 + i), 0.9) for i in range(3)]
    mock_matcher.match_with_score.side_effect = match_returns * 2  # 2 images

    mock_reader.read_quantity.side_effect = [(i * 100, 0.95) for i in range(6)]

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
    assert len(cell_events) == 3  # only from first image (second has same items)
