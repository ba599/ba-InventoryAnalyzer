import numpy as np
import pytest

from src.count_ocr_backend import CountOcrBackend, build_backend


class TestCountOcrBackendABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            CountOcrBackend()

    def test_subclass_must_implement_read_quantity(self):
        class Incomplete(CountOcrBackend):
            @property
            def backend_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_backend_name(self):
        class Incomplete(CountOcrBackend):
            def read_quantity(self, image):
                return None

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_works(self):
        class Dummy(CountOcrBackend):
            @property
            def backend_name(self) -> str:
                return "dummy"

            def read_quantity(self, image: np.ndarray):
                return (42, 0.99)

        d = Dummy()
        assert d.backend_name == "dummy"
        assert d.read_quantity(np.zeros((10, 10, 3), dtype=np.uint8)) == (42, 0.99)


class TestBuildBackend:
    def test_build_backend_returns_yolo(self):
        backend = build_backend()
        assert backend.backend_name == "yolo"
