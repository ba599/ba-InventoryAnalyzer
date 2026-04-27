import cv2
import numpy as np
import pytest
from src.item_matcher import ItemMatcher
from src.ref_builder import build_references


class TestItemMatcher:
    def _make_icon(self, color: tuple[int, int, int], size: int = 60) -> np.ndarray:
        """Create a simple colored icon for testing."""
        img = np.ones((size, size, 3), dtype=np.uint8) * 200
        cv2.circle(img, (size // 2, size // 2), size // 3, color, -1)
        return img

    def test_match_returns_best_material_id(self, references_dir):
        # Create distinct reference images
        red_icon = self._make_icon((0, 0, 200))
        blue_icon = self._make_icon((200, 0, 0))
        cv2.imwrite(str(references_dir / "100.png"), red_icon)
        cv2.imwrite(str(references_dir / "101.png"), blue_icon)

        matcher = ItemMatcher(references_dir)

        # Query with red icon (slightly different size to test resize)
        query = self._make_icon((0, 0, 200), size=80)
        result = matcher.match(query)
        assert result == "100"

    def test_match_returns_none_when_no_references(self, tmp_path):
        empty_dir = tmp_path / "empty_refs"
        empty_dir.mkdir()
        matcher = ItemMatcher(empty_dir)
        icon = self._make_icon((100, 100, 100))
        result = matcher.match(icon)
        assert result is None

    def test_loads_all_reference_images(self, references_dir):
        for mid in ["100", "101", "102"]:
            icon = self._make_icon((int(mid) % 256, 50, 50))
            cv2.imwrite(str(references_dir / f"{mid}.png"), icon)

        matcher = ItemMatcher(references_dir)
        assert len(matcher.references) == 3


class TestRefBuilder:
    def test_builds_references_from_grid(self, sample_grid_image, tmp_path):
        order = [f"{i}" for i in range(100, 200)]  # 100 IDs
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()

        count = build_references(
            images=[sample_grid_image],
            item_order=order,
            start_id="100",
            output_dir=ref_dir,
        )

        # Should have created reference images for detected cells
        assert count > 0
        # Check files exist
        created = list(ref_dir.glob("*.png"))
        assert len(created) == count
        # First file should be named after start_id
        assert (ref_dir / "100.png").exists()
