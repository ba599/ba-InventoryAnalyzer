import json
from pathlib import Path


def load_json(path: Path) -> dict:
    """Load Justin Planner JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_owned_materials(data: dict, updates: dict[str, int]) -> dict:
    """Update owned_materials in-place. Values are stored as strings to match Justin format.

    Args:
        data: Full Justin Planner JSON dict.
        updates: {material_id: quantity} with integer quantities.

    Returns:
        The modified data dict (same reference).
    """
    owned = data.get("owned_materials", {})
    for material_id, quantity in updates.items():
        if material_id in owned:
            owned[material_id] = str(quantity)
    data["owned_materials"] = owned
    return data


def save_json(data: dict, path: Path) -> None:
    """Save Justin Planner JSON file, preserving structure."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
