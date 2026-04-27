import cv2
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_cell_image():
    """A synthetic 100x100 cell image with a colored rectangle border and 'x1234' text."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 220  # light gray background
    # Draw a border (blue-ish, like in-game rarity frame)
    cv2.rectangle(img, (2, 2), (97, 97), (180, 120, 60), 2)
    # Draw "x1234" text at bottom-right
    cv2.putText(img, "x1234", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


@pytest.fixture
def sample_grid_image():
    """A synthetic 700x500 grid image with 7x5 cells."""
    cell_w, cell_h = 90, 90
    gap = 10
    cols, rows = 7, 5
    width = cols * cell_w + (cols + 1) * gap
    height = rows * cell_h + (rows + 1) * gap
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # light background

    for r in range(rows):
        for c in range(cols):
            x = gap + c * (cell_w + gap)
            y = gap + r * (cell_h + gap)
            # Cell background
            cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), (200, 200, 200), -1)
            # Cell border
            cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), (150, 100, 50), 2)
            # Quantity text
            num = r * cols + c + 1
            cv2.putText(img, f"x{num * 100}", (x + 25, y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


@pytest.fixture
def docs_dir():
    """Path to docs directory with real game screenshots."""
    return Path(__file__).parent.parent / "docs"


@pytest.fixture
def references_dir(tmp_path):
    """Temporary directory for reference images during tests."""
    ref_dir = tmp_path / "references"
    ref_dir.mkdir()
    return ref_dir
