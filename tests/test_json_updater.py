import json
import pytest
from pathlib import Path
from src.json_updater import load_json, update_owned_materials, save_json


class TestJsonUpdater:
    def _make_sample_json(self, tmp_path: Path) -> Path:
        data = {
            "exportVersion": 2,
            "characters": [],
            "owned_materials": {
                "100": "2285",
                "101": "1352",
                "T8_Necklace": "470",
                "Credit": "1880970876"
            },
            "other_field": "keep_this"
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def test_load_json(self, tmp_path):
        path = self._make_sample_json(tmp_path)
        data = load_json(path)
        assert data["exportVersion"] == 2
        assert "owned_materials" in data

    def test_update_owned_materials(self, tmp_path):
        path = self._make_sample_json(tmp_path)
        data = load_json(path)
        updates = {"100": 9999, "101": 500}
        updated = update_owned_materials(data, updates)
        assert updated["owned_materials"]["100"] == "9999"
        assert updated["owned_materials"]["101"] == "500"
        # Untouched keys remain
        assert updated["owned_materials"]["T8_Necklace"] == "470"
        assert updated["owned_materials"]["Credit"] == "1880970876"

    def test_update_preserves_other_fields(self, tmp_path):
        path = self._make_sample_json(tmp_path)
        data = load_json(path)
        updated = update_owned_materials(data, {"100": 1})
        assert updated["other_field"] == "keep_this"
        assert updated["exportVersion"] == 2

    def test_save_json_roundtrip(self, tmp_path):
        path = self._make_sample_json(tmp_path)
        data = load_json(path)
        update_owned_materials(data, {"100": 42})
        out_path = tmp_path / "output.json"
        save_json(data, out_path)
        reloaded = load_json(out_path)
        assert reloaded["owned_materials"]["100"] == "42"
