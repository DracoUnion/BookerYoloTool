"""
YOLO inference / training engine (ultralytics).

All YOLO calls now go through the `ultralytics` library (YOLOv8 / YOLO11
family), replacing the original MFC tool's darknet runtime and the earlier
pure-PyTorch fallback.

  * UltralyticsEngine - loads `.pt` / `.onnx` / `.yaml` models (or auto-
    downloads pretrained checkpoints) via `ultralytics.YOLO`, runs
    inference with `model.predict()`, and trains with `model.train()`.

Detection boxes are exposed normalized 0..1 via :class:`Detection`, and
annotated images ("yolo绘制") via :class:`InferenceResult.annotated`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import YOLOConfig


@dataclass
class Detection:
    """A single detection box, normalised 0..1."""

    label: str
    confidence: float
    x: float
    y: float
    w: float
    h: float

    def to_pix(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        return (
            int(self.x * img_w),
            int(self.y * img_h),
            int(self.w * img_w),
            int(self.h * img_h),
        )


class InferenceResult:
    """Holds an image + its detections + elapsed time (ms)."""

    def __init__(self, image: np.ndarray, detections: list[Detection], elapsed_ms: float):
        self.image = image
        self.detections = detections
        self.elapsed_ms = elapsed_ms

    def annotated(self) -> np.ndarray:
        """Return a copy of the image with boxes + labels drawn (yolo绘制)."""
        canvas = self.image.copy()
        h, w = canvas.shape[:2]
        for det in self.detections:
            x, y, bw, bh = det.to_pix(w, h)
            color = (0, 255, 0) if det.confidence > 0.5 else (0, 200, 255)
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
            label = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                canvas, label, (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return canvas


class YOLOEngine:
    """Common interface for ultralytics-backed inference / training."""

    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model = None
        self.last_train_dir: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # model
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Instantiate the ultralytics model from config.model_source.

        `model_source` may be:
          * a path to a local `.pt` / `.onnx` / `.engine` / `.yaml` file
          * an ultralytics built-in name like "yolov8n" / "yolo11n"
            (first use auto-downloads the checkpoint)
        """
        from ultralytics import YOLO  # imported lazily

        source = self.config.model_source
        if not source:
            raise RuntimeError("No model source configured (model_source is empty).")
        self.model = YOLO(str(source))

    def infer(self, image: np.ndarray, conf_thresh: float = 0.5) -> InferenceResult:
        """Run detection on a BGR uint8 image (HxWxC) via model.predict()."""
        if self.model is None:
            raise RuntimeError(
                "Ultralytics model not loaded. Call load() (or set model_source) first."
            )
        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_thresh,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        dets = self._results_to_detections(results, image.shape[1], image.shape[0])
        return InferenceResult(image, dets, elapsed_ms)

    def _results_to_detections(self, results, img_w: int, img_h: int) -> list[Detection]:
        """Convert ultralytics Results (boxes are pixel coords) to normalized."""
        dets: list[Detection] = []
        if not results:
            return dets
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return dets
        names = r.names  # {id: label}
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) pixel
        conf = boxes.conf.cpu().numpy()  # (N,)
        cls = boxes.cls.cpu().numpy().astype(int)  # (N,)
        for (x1, y1, x2, y2), c, cid in zip(xyxy, conf, cls):
            dets.append(
                Detection(
                    label=names.get(cid, str(cid)),
                    confidence=float(c),
                    x=float(x1) / img_w,
                    y=float(y1) / img_h,
                    w=float(x2 - x1) / img_w,
                    h=float(y2 - y1) / img_h,
                )
            )
        return dets

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self, iterations: int, log_path: Path, data_yaml: Optional[Path] = None) -> dict:
        """Run ultralytics training.

        Args:
            iterations: number of epochs.
            log_path: where run outputs (weights, results.csv, logs) land.
            data_yaml: dataset description (YAML). Defaults to config.data_yaml.

        Returns:
            results dict from model.train() (e.g. {'metrics': ..., 'fitness': ...}).
        """
        if self.model is None:
            raise RuntimeError("No ultralytics model loaded. Call load() first.")

        data = data_yaml or self.config.data_yaml
        project = log_path.parent if log_path.parent != Path(".") else None
        name = log_path.stem if log_path.stem else "train"
        res = self.model.train(
            data=str(data),
            epochs=int(iterations),
            imgsz=self.config.imgsz,
            project=str(project) if project else None,
            name=name,
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        # ultralytics saves under {project or runs_dir}/{task}/{name}
        task = getattr(self.model, "task", "detect")
        save_parent = project or (Path.cwd() / "runs")
        self.last_train_dir = save_parent / task / name
        return res


def make_engine(config: YOLOConfig) -> YOLOEngine:
    """Create an ultralytics-backed engine for the given config."""
    return YOLOEngine(config)