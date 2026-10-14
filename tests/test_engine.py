"""Unit tests for the YOLO tool core (config + engine), no GUI.

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
from yolo_tool.engine import (
    Detection,
    InferenceResult,
    YOLOEngine,
    make_engine,
)  # noqa: E402


def test_inference_result_annotated() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [Detection("person", 0.9, 0.1, 0.1, 0.2, 0.4)]
    res = InferenceResult(img, dets, 12.0)
    ann = res.annotated()
    assert ann.shape == img.shape
    assert ann.any()  # a green box should now be drawn


def test_detection_to_pix() -> None:
    d = Detection("x", 0.8, 0.25, 0.5, 0.5, 0.5)
    x, y, w, h = d.to_pix(200, 100)
    assert (x, y, w, h) == (50, 50, 100, 50)


def test_config_builtins() -> None:
    reg = ConfigRegistry(Path("."))
    presets = reg.presets()
    assert any(p.name == "yolov8n" for p in presets)
    # yolov8n is a built-in name, not a local file
    assert presets[0].is_builtin()
    assert presets[0].is_valid()


def test_config_from_local_file() -> None:
    cfg = YOLOConfig(name="m", model_source="C:/models/m.pt")
    assert not cfg.is_builtin()
    assert not cfg.is_valid()  # file does not exist
    assert "m" in cfg.model_source


def test_make_engine_type() -> None:
    cfg = YOLOConfig(name="yolov8n", model_source="yolov8n")
    eng = make_engine(cfg)
    assert isinstance(eng, YOLOEngine)


def test_results_to_detections_no_boxes() -> None:
    eng = make_engine(YOLOConfig(name="x", model_source="x"))
    assert eng._results_to_detections([], 640, 480) == []


if __name__ == "__main__":
    test_inference_result_annotated()
    test_detection_to_pix()
    test_config_builtins()
    test_config_from_local_file()
    test_make_engine_type()
    test_results_to_detections_no_boxes()
    print("ALL TESTS PASSED")