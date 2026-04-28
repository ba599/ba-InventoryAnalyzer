from pathlib import Path
import cv2
import numpy as np


class ItemMatcher:
    """Matches icon crops against a reference image database using template matching."""

    MATCH_SIZE = (64, 64)  # Normalize all images to this size for comparison

    def __init__(self, references_dir: Path):
        self.references: dict[str, np.ndarray] = {}
        self._load_references(references_dir)

    def _load_references(self, ref_dir: Path) -> None:
        for path in sorted(ref_dir.glob("*.png")):
            material_id = path.stem
            buf = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, self.MATCH_SIZE)
                self.references[material_id] = img

    def match(self, icon: np.ndarray, min_score: float = 0.0) -> str | None:
        """Find the best matching material_id for an icon crop.

        Args:
            icon: BGR image of the icon region.
            min_score: Minimum match score to accept (0.0-1.0).

        Returns:
            material_id string of the best match, or None if no match above min_score.
        """
        best_id, best_score = self.match_with_score(icon)
        if best_id is not None and best_score >= min_score:
            return best_id
        return None

    def match_with_score(self, icon: np.ndarray) -> tuple[str | None, float]:
        """Find the best matching material_id with its score.

        Returns:
            (material_id, score) tuple, or (None, -1.0) if no references loaded.
        """
        if not self.references:
            return None, -1.0

        query = cv2.resize(icon, self.MATCH_SIZE)
        best_id = None
        best_score = -1.0

        for material_id, ref_img in self.references.items():
            result = cv2.matchTemplate(query, ref_img, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]
            if score > best_score:
                best_score = score
                best_id = material_id

        return best_id, best_score
