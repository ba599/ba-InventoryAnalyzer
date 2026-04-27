import json
from pathlib import Path


def load_answer(path: Path) -> dict:
    """Load answer.json ground truth file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_results(ocr_results: dict[str, int], answer_data: dict) -> dict:
    """Compare OCR results against answer ground truth.

    Args:
        ocr_results: {material_id: quantity} from OCR.
        answer_data: Full answer.json data with "owned_materials" key.

    Returns:
        Dict with keys: total, correct, errors (list of dicts), skipped, accuracy.
    """
    answer = answer_data.get("owned_materials", {})
    correct = 0
    errors = []
    skipped = 0

    for material_id, ocr_qty in ocr_results.items():
        if material_id not in answer:
            skipped += 1
            continue
        expected = int(answer[material_id])
        if ocr_qty == expected:
            correct += 1
        else:
            errors.append({
                "id": material_id,
                "ocr": ocr_qty,
                "expected": expected,
            })

    total = correct + len(errors)
    accuracy = (correct / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "errors": errors,
        "skipped": skipped,
        "accuracy": round(accuracy, 2),
    }


def print_report(result: dict) -> None:
    """Print a human-readable accuracy report."""
    print(f"\n=== Accuracy Report ===")
    print(f"Total: {result['total']}, Correct: {result['correct']}, "
          f"Errors: {len(result['errors'])}, Skipped: {result['skipped']}")
    print(f"Accuracy: {result['accuracy']}%")

    if result["errors"]:
        print(f"\nErrors:")
        for err in result["errors"]:
            print(f"  {err['id']}: OCR={err['ocr']} Expected={err['expected']}")
