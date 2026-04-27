import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.core.pipeline import process_single_image


def _make_cell_info(x, y, w, h, row, col):
    from src.grid_detector import CellInfo
    return CellInfo(x=x, y=y, w=w, h=h, row=row, col=col)


def test_process_single_image_returns_results_and_cell_images():
    """process_single_image returns {mid: (qty, conf)} and {mid: cell_crop}."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101", "102"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
    ]

    mock_matcher.match_with_score.return_value = ("100", 0.9)

    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (200, 0.80),
        (100, 0.50),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results, cell_images = process_single_image(
            fake_image, item_order, mock_matcher, mock_reader
        )

    assert "100" in results
    assert "101" in results
    assert "102" in results
    assert results["100"] == (500, 0.95)
    assert results["101"] == (200, 0.80)
    assert results["102"] == (100, 0.50)
    assert "100" in cell_images
    assert "101" in cell_images
    assert "102" in cell_images


def test_process_single_image_no_match_returns_empty():
    """When no cell matches a reference, return empty results."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", "101"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [_make_cell_info(0, 0, 80, 80, 0, 0)]
    mock_matcher.match_with_score.return_value = (None, -1.0)

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]):
        results, cell_images = process_single_image(
            fake_image, item_order, mock_matcher, mock_reader
        )

    assert results == {}
    assert cell_images == {}


def test_process_single_image_skips_null_items():
    """Null entries in item_order are skipped."""
    fake_image = np.zeros((200, 400, 3), dtype=np.uint8)
    item_order = ["100", None, "102"]

    mock_matcher = MagicMock()
    mock_reader = MagicMock()

    cells = [
        _make_cell_info(0, 0, 80, 80, 0, 0),
        _make_cell_info(80, 0, 80, 80, 0, 1),
        _make_cell_info(160, 0, 80, 80, 0, 2),
    ]
    mock_matcher.match_with_score.return_value = ("100", 0.9)
    mock_reader.read_quantity.side_effect = [
        (500, 0.95),
        (999, 0.99),
        (100, 0.50),
    ]

    with patch("src.core.pipeline.detect_cells", return_value=cells), \
         patch("src.core.pipeline.crop_icon_region", return_value=fake_image[:80, :80]), \
         patch("src.core.pipeline.crop_text_region", return_value=fake_image[:20, :20]):
        results, cell_images = process_single_image(
            fake_image, item_order, mock_matcher, mock_reader
        )

    assert "100" in results
    assert "102" in results
    assert None not in results
    assert len(results) == 2
