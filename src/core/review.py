from dataclasses import dataclass

import numpy as np


@dataclass
class ReviewItem:
    """An OCR result that needs manual review."""
    material_id: str
    ocr_qty: int
    confidence: float
    reasons: list[str]
    cell_image: np.ndarray


def find_review_items(
    results: dict[str, tuple[int, float]],
    existing_materials: dict[str, str],
    cell_images: dict[str, np.ndarray],
    threshold: float = 0.7,
    deviation_limit: int = 100,
) -> list[ReviewItem]:
    """Identify items needing manual review.

    Flags items with:
    - OCR confidence below threshold
    - Quantity change >= deviation_limit from existing value

    Returns:
        List of ReviewItem, one per flagged material.
    """
    items: list[ReviewItem] = []

    for mid, (qty, conf) in results.items():
        reasons: list[str] = []

        if conf < threshold:
            reasons.append(f"low confidence ({conf:.2f})")

        if mid in existing_materials:
            old_val = int(existing_materials[mid]) if existing_materials[mid] else 0
            if abs(qty - old_val) >= deviation_limit:
                reasons.append(f"deviation {qty - old_val:+d} (was {old_val})")

        if reasons:
            items.append(ReviewItem(
                material_id=mid,
                ocr_qty=qty,
                confidence=conf,
                reasons=reasons,
                cell_image=cell_images[mid],
            ))

    return items
