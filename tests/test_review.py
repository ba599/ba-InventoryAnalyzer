import numpy as np
from src.core.review import find_review_items, ReviewItem


def test_low_confidence_flagged():
    results = {"100": (500, 0.50)}
    existing = {"100": "500"}
    cell_images = {"100": np.zeros((80, 80, 3), dtype=np.uint8)}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert len(items) == 1
    assert items[0].material_id == "100"
    assert items[0].ocr_qty == 500
    assert "low confidence" in items[0].reasons[0]


def test_large_deviation_flagged():
    results = {"100": (500, 0.95)}
    existing = {"100": "1500"}
    cell_images = {"100": np.zeros((80, 80, 3), dtype=np.uint8)}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert len(items) == 1
    assert "deviation" in items[0].reasons[0]


def test_both_reasons_flagged():
    results = {"100": (500, 0.50)}
    existing = {"100": "1500"}
    cell_images = {"100": np.zeros((80, 80, 3), dtype=np.uint8)}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert len(items) == 1
    assert len(items[0].reasons) == 2


def test_no_review_needed():
    results = {"100": (500, 0.95)}
    existing = {"100": "510"}
    cell_images = {"100": np.zeros((80, 80, 3), dtype=np.uint8)}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert len(items) == 0


def test_review_item_has_cell_image():
    results = {"100": (500, 0.50)}
    existing = {"100": "500"}
    cell_img = np.ones((80, 80, 3), dtype=np.uint8) * 128
    cell_images = {"100": cell_img}

    items = find_review_items(results, existing, cell_images, threshold=0.7)

    assert items[0].cell_image is cell_img
