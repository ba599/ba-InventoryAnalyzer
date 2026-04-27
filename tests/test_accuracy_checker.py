import pytest
from src.accuracy_checker import compare_results


class TestAccuracyChecker:
    def test_all_correct(self):
        ocr = {"100": 2375, "101": 1475}
        answer = {"owned_materials": {"100": "2375", "101": "1475"}}
        result = compare_results(ocr, answer)
        assert result["total"] == 2
        assert result["correct"] == 2
        assert result["errors"] == []

    def test_some_errors(self):
        ocr = {"100": 2375, "101": 9999}
        answer = {"owned_materials": {"100": "2375", "101": "1475"}}
        result = compare_results(ocr, answer)
        assert result["total"] == 2
        assert result["correct"] == 1
        assert len(result["errors"]) == 1
        err = result["errors"][0]
        assert err["id"] == "101"
        assert err["ocr"] == 9999
        assert err["expected"] == 1475

    def test_missing_in_answer(self):
        """OCR found an item not in answer — it is skipped (not counted as error)."""
        ocr = {"100": 2375, "UNKNOWN": 999}
        answer = {"owned_materials": {"100": "2375"}}
        result = compare_results(ocr, answer)
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["skipped"] == 1

    def test_accuracy_percentage(self):
        ocr = {"100": 2375, "101": 9999, "102": 701}
        answer = {"owned_materials": {"100": "2375", "101": "1475", "102": "701"}}
        result = compare_results(ocr, answer)
        assert result["accuracy"] == pytest.approx(66.67, abs=0.01)
