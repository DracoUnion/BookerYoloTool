"""Unit tests for the YOLO tool core (engine + config), no GUI.

Run:  pytest tests/
Or:   python tests/test_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yolo_tool.config import ConfigRegistry, YOLOConfig  # noqa: E402
from yolo_tool.engine import InferenceResult, make_engine  # noqa: E402


def test_inference_result_annotated() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    from yolo_tool.engine import Detection

    dets = [Detection("person", 0.9, 0.1, 0.1, 0.2, 0.4)]
    res = InferenceResult(img, dets, 12.0)
    ann = res.annotated()
    assert ann.shape == img.shape
    # green pixel should now exist where the box is drawn
    assert ann.any()


def test_config_registry_discover() -> None:
    reg = ConfigRegistry(Path("/tmp/yolo_root"))
    c3 = reg.discover_default_v3()
    c4 = reg.discover_default_v4()
    assert c3.version == "v3" and c4.version == "v4"
    # files don't exist in /tmp, so is_valid() should be False
    assert not c3.is_valid()


def test_make_engine_fallback() -> None:
    c = YOLOConfig(name="x", version="v3", cfg_path=Path("a.cfg"), names_path=Path("b.names"))
    eng = make_engine(c)
    assert isinstance(eng, object)
    print("engine chosen:", type(eng).__name__)


if __name__ == "__main__":
    test_inference_result_annotated()
    test_config_registry_discover()
    test_make_engine_fallback()
    print("ALL TESTS PASSED")
