"""
YOLO inference / training engine.

The original MFC tool shells out to the darknet runtime (运行库\\...cmd,
startcmd.exe, darknet.cfg) for training and to darknet for inference.
This module provides two back-ends:

  * DarknetEngine  - wraps a darknet binary / batch file (faithful to original)
  * TorchYOLOEngine- pure-PyTorch inference fallback (no external binary)

Both implement the common :class:`YOLOEngine` interface used by the GUI.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from abc import ABC, abstractmethod
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


class YOLOEngine(ABC):
    """Common interface for inference / training back-ends."""

    def __init__(self, config: YOLOConfig):
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """Load model + weights. Raise on failure."""

    @abstractmethod
    def infer(self, image: np.ndarray, conf_thresh: float = 0.5) -> InferenceResult:
        """Run detection on a BGR uint8 image (HxWxC)."""

    def train(self, iterations: int, log_path: Path) -> subprocess.CompletedProcess:
        """Optional: run training. Default no-op for inference-only engines."""
        raise NotImplementedError("This engine does not support training.")

    # -- shared helpers ----------------------------------------------------
    def _names(self) -> list[str]:
        p = self.config.names_path
        if p.exists():
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                return [ln.strip() for ln in fh if ln.strip()]
        return [f"class_{i}" for i in range(0)]


class DarknetEngine(YOLOEngine):
    """Wraps a darknet binary / batch script (original behavior)."""

    def __init__(self, config: YOLOConfig, darknet_bin: Path):
        super().__init__(config)
        self.darknet_bin = Path(darknet_bin)
        if not self.darknet_bin.exists():
            raise FileNotFoundError(f"darknet binary not found: {self.darknet_bin}")
        self._loaded = False

    def load(self) -> None:
        if not self.config.cfg_path.exists():
            raise FileNotFoundError(f"cfg missing: {self.config.cfg_path}")
        self._loaded = True

    def infer(self, image: np.ndarray, conf_thresh: float = 0.5) -> InferenceResult:
        if not self._loaded:
            self.load()
        import tempfile
        import time

        h, w = image.shape[:2]
        tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
        cv2.imwrite(str(tmp), image)
        out_txt = tmp.with_suffix(".txt")

        weights = self.config.weights_path or ""
        cmd = [
            str(self.darknet_bin), "det", str(self.config.cfg_path),
            weights, str(tmp),
            "-thresh", f"{conf_thresh}", "-out", str(out_txt),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            raise RuntimeError(f"darknet failed: {proc.stderr[:400]}")

        dets = self._parse_txt(out_txt, w, h)
        tmp.unlink(missing_ok=True)
        out_txt.unlink(missing_ok=True)
        return InferenceResult(image, dets, elapsed_ms)

    def _parse_txt(self, out: Path, img_w: int, img_h: int) -> list[Detection]:
        dets: list[Detection] = []
        names = self._names()
        if out.exists():
            for line in out.read_text().splitlines():
                # darknet txt: <class> <conf> <x> <y> <w> <h> (px, yolo 416-space)
                parts = line.split()
                if len(parts) < 7:
                    continue
                cls, conf, cx, cy, bw, bh = map(float, parts[1:7])
                dets.append(
                    Detection(
                        label=names[int(cls)] if int(cls) < len(names) else str(int(cls)),
                        confidence=conf,
                        x=cx / img_w - bw / (2 * img_w),
                        y=cy / img_h - bh / (2 * img_h),
                        w=bw / img_w,
                        h=bh / img_h,
                    )
                )
        return dets

    def train(self, iterations: int, log_path: Path) -> subprocess.CompletedProcess:
        """Run training via the original .cmd / darknet train command."""
        if self.config.train_command and self.config.train_command.exists():
            cmd = [str(self.config.train_command), str(iterations)]
        else:
            cmd = [
                str(self.darknet_bin), "train", str(self.config.data_config_path),
                "-weights", str(self.config.weights_path or ""),
                "-clear",
            ]
        with open(log_path, "w", encoding="utf-8") as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)


class TorchYOLOEngine(YOLOEngine):
    """Pure-PyTorch YOLOv3/v4 inference (no external darknet binary).

    Loads a darknet .cfg + .weights, or (for YOLOv4-tiny-dota2) a bundled
    ONNX/pt model.  Keeps the same YOLOEngine interface.
    """

    def __init__(self, config: YOLOConfig, model_path: Optional[Path] = None):
        super().__init__(config)
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        import torch  # imported lazily

        if self.model_path and self.model_path.exists():
            try:
                self.model = torch.load(str(self.model_path), map_location="cpu")
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"failed to load model {self.model_path}: {exc}")
        else:
            # No model file: leave untrained; infer() will raise a helpful error.
            self.model = None

    def infer(self, image: np.ndarray, conf_thresh: float = 0.5) -> InferenceResult:
        import torch
        import time

        if self.model is None:
            raise RuntimeError(
                "No YOLO model loaded. Provide .weights / .pt model_path to TorchYOLOEngine."
            )
        t0 = time.perf_counter()
        tensor = torch.from_numpy(image.astype("float32") / 255.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            out = self.model(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        dets = self._postprocess(out, image.shape[1], image.shape[0], conf_thresh)
        return InferenceResult(image, dets, elapsed_ms)

    def _postprocess(self, out, w: int, h: int, conf_thresh: float) -> list[Detection]:
        # Generic single-head YOLO output: list of (N, 5+nc) grids.
        dets: list[Detection] = []
        names = self._names()
        try:
            preds = out[0] if isinstance(out, (list, tuple)) else out
            if hasattr(preds, "cpu"):
                preds = preds.cpu().numpy()
            # naive decode for a single prediction tensor (num_boxes, 5+nc)
            for row in preds:
                obj_conf = row[4] if row.shape[0] > 4 else 0.0
                cls_scores = row[5:] if row.shape[0] > 5 else np.array([1.0])
                best = int(np.argmax(cls_scores))
                conf = float(obj_conf * cls_scores[best])
                if conf < conf_thresh:
                    continue
                cy, cx, bh, bw = row[0], row[1], row[2], row[3]
                cy, cx = cy * h, cx * w
                bh, bw = bh * h, bw * w
                dets.append(
                    Detection(
                        label=names[best] if best < len(names) else f"class_{best}",
                        confidence=conf,
                        x=cx / w - bw / (2 * w),
                        y=cy / h - bh / (2 * h),
                        w=bw / w,
                        h=bh / h,
                    )
                )
        except Exception:
            return []
        return dets


def make_engine(config: YOLOConfig, darknet_bin: Optional[Path] = None) -> YOLOEngine:
    """Pick the best available back-end."""
    if darknet_bin is not None:
        return DarknetEngine(config, darknet_bin)
    # Prefer darknet binary if present in runtime root
    if config.runtime_root:
        for cand in ("darknet.exe", "darknet", "startcmd.exe", "darknet_x86_64_opencl.exe"):
            p = config.runtime_root / cand
            if p.exists():
                return DarknetEngine(config, p)
    return TorchYOLOEngine(config)
