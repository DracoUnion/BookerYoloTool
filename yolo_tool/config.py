"""
Configuration management for the YOLO tool (ultralytics).

Model sources are ultralytics-native: local `.pt` / `.onnx` / `.engine` /
`.yaml` files or built-in checkpoint names (e.g. "yolov8n") that
ultralytics auto-downloads on first use.  All model loading + inference +
training go through ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Supported image formats (from the original Open/Save dialog filter)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".ico", ".cur")

# Ultralytics-native model extensions we can load
MODEL_EXTENSIONS = (".pt", ".onnx", ".engine", ".tflite", ".yaml", ".yml")

# Built-in ultralytics checkpoints (auto-downloaded on first use)
BUILTIN_MODELS = (
    ("yolov8n", "yolov8n"),
    ("yolov8s", "yolov8s"),
    ("yolov8m", "yolov8m"),
    ("yolov8l", "yolov8l"),
    ("yolov8x", "yolov8x"),
    ("yolo11n", "yolo11n"),
    ("yolo11s", "yolo11s"),
)


@dataclass
class YOLOConfig:
    """Describes an ultralytics model to load / infer / train."""

    name: str
    model_source: str   # local path or ultralytics built-in name
    task: str = "detect"
    imgsz: int = 640
    data_yaml: Optional[Path] = None
    extra: dict = field(default_factory=dict)

    def is_builtin(self) -> bool:
        """True if model_source is a checkpoint name (not a file path).

        A built-in is a bare name: no path separators and either no
        extension or a supported ultralytics extension.  Anything that
        looks like a path (even a non-existent one) is treated as a file.
        """
        s = str(self.model_source)
        if (
            ":" in s  # drive letter (C:...)
            or "/" in s
            or "\\" in s
            or Path(s).suffix.lower() in MODEL_EXTENSIONS
        ):
            return False
        # bare name like "yolov8n" (optionally with .pt)
        return True

    def is_valid(self) -> bool:
        """Built-ins are always valid (auto-download); files must exist."""
        return self.is_builtin() or Path(self.model_source).expanduser().exists()

    def __post_init__(self) -> None:
        # normalize built-in names: "yolov8n.pt" -> "yolov8n" (bare checkpoint name)
        if self.is_builtin():
            self.model_source = Path(self.model_source).stem


class ConfigRegistry:
    """Provides ultralytics model configs."""

    def __init__(self, install_root: Optional[Path] = None):
        self.install_root = Path(install_root) if install_root else Path.cwd()

    def presets(self) -> list[YOLOConfig]:
        """Return built-in ultralytics checkpoints as configs."""
        return [
            YOLOConfig(name=label, model_source=name, imgsz=640)
            for label, name in BUILTIN_MODELS
        ]