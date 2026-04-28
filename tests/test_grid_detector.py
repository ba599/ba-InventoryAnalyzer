import cv2
import numpy as np
import pytest
from src.grid_detector import detect_cells, CellInfo


class TestDetectCells:
    def test_detects_cells_in_synthetic_grid(self, sample_grid_image):
        cells = detect_cells(sample_grid_image)
        # 7x5 grid = 35 cells
        assert len(cells) >= 30  # allow some tolerance for edge detection

    def test_cells_have_position_info(self, sample_grid_image):
        cells = detect_cells(sample_grid_image)
        assert len(cells) > 0
        cell = cells[0]
        assert isinstance(cell, CellInfo)
        assert cell.x >= 0
        assert cell.y >= 0
        assert cell.w > 0
        assert cell.h > 0
        assert cell.row >= 0
        assert cell.col >= 0

    def test_cells_sorted_top_left_to_bottom_right(self, sample_grid_image):
        cells = detect_cells(sample_grid_image)
        for i in range(1, len(cells)):
            prev, curr = cells[i - 1], cells[i]
            # Either on a later row, or same row but later column
            assert (curr.row > prev.row) or (curr.row == prev.row and curr.col >= prev.col)

    def test_skips_partial_cells(self):
        """Cells cut off at edges (< 60% normal size) should be skipped."""
        cell_w, cell_h = 90, 90
        gap = 10
        # Create grid where first column is half-cut
        img = np.ones((200, 500, 3), dtype=np.uint8) * 240
        # One partial cell (half width)
        cv2.rectangle(img, (0, gap), (40, gap + cell_h), (200, 200, 200), -1)
        cv2.rectangle(img, (0, gap), (40, gap + cell_h), (150, 100, 50), 2)
        # Three full cells
        for c in range(3):
            x = 50 + c * (cell_w + gap)
            cv2.rectangle(img, (x, gap), (x + cell_w, gap + cell_h), (200, 200, 200), -1)
            cv2.rectangle(img, (x, gap), (x + cell_w, gap + cell_h), (150, 100, 50), 2)

        cells = detect_cells(img)
        # Partial cell should be skipped, only 3 full cells detected
        assert len(cells) == 3


from src.grid_detector import crop_icon_region, crop_text_region


class TestCropRegions:
    def test_crop_icon_region(self, sample_grid_image):
        cells = detect_cells(sample_grid_image)
        assert len(cells) > 0
        icon = crop_icon_region(sample_grid_image, cells[0])
        assert icon.shape[0] > 0
        assert icon.shape[1] > 0
        # Icon height should be ~78% of cell height
        assert icon.shape[0] == int(cells[0].h * 0.78)

    def test_crop_text_region(self, sample_grid_image):
        cells = detect_cells(sample_grid_image)
        assert len(cells) > 0
        text = crop_text_region(sample_grid_image, cells[0])
        assert text.shape[0] > 0
        assert text.shape[1] > 0
        # Text height should be ~22% of cell height
        expected_h = cells[0].h - int(cells[0].h * 0.78)
        assert text.shape[0] == expected_h
