"""
Service layer: a zero-Qt ``ServiceFacade`` as the single entry for the UI.

Wraps :class:`YOLOEngine` and the :class:`TaskManager` so model load,
inference and training all run off the UI thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from yolo_tool.common.events import EventHub
from yolo_tool.engine import InferenceResult, YOLOEngine, make_engine
from yolo_tool.tasks import Task, TaskCancelled, TaskManager

if TYPE_CHECKING:
    import numpy as np

    from yolo_tool.config import YOLOConfig


class ServiceFacade:
    """UI-facing facade; no Qt imports."""

    def __init__(
        self,
        *,
        config_provider,
        events: EventHub,
        tasks: TaskManager,
    ) -> None:
        self._config_provider = config_provider
        self.events = events
        self.tasks = tasks
        self.engine: Optional[YOLOEngine] = None

    # -- model ----------------------------------------------------------- #

    def load_model(self, cfg: "YOLOConfig", label: str = "load-model") -> Task:
        """Load/instantiate an ultralytics model in the background.

        ``model_source`` may need a first-use download, so this must not run
        on the UI thread.
        """

        def _run(*, cancel_event, **_: Any) -> YOLOEngine:
            if cancel_event.is_set():
                raise TaskCancelled()
            engine = make_engine(cfg)
            engine.load()
            return engine

        return self.tasks.submit(_run, label=label)

    def load_model_now(self, cfg: "YOLOConfig") -> YOLOEngine:
        """Synchronous model load (used when a cached model is required)."""
        if self.engine is not None and self.engine.config == cfg and self.engine.model is not None:
            return self.engine
        engine = make_engine(cfg)
        engine.load()
        self.engine = engine
        return engine

    # -- inference ------------------------------------------------------- #

    def infer(
        self,
        image,
        conf_thresh: float = 0.5,
        label: str = "inference",
    ) -> Task:
        """Run detection on an image off the UI thread."""

        def _run(*, cancel_event, **_: Any) -> InferenceResult:
            if cancel_event.is_set():
                raise TaskCancelled()
            if self.engine is None or self.engine.model is None:
                raise RuntimeError("No model loaded. Call load_model() first.")
            return self.engine.infer(image, conf_thresh)

        return self.tasks.submit(_run, label=label)

    # -- training -------------------------------------------------------- #

    def train(
        self,
        cfg: "YOLOConfig",
        iterations: int,
        log_path: Path,
        data_yaml: Optional[Path] = None,
        label: str = "training",
    ) -> Task:
        """Run model training off the UI thread."""

        def _run(*, cancel_event, **_: Any) -> Any:
            if cancel_event.is_set():
                raise TaskCancelled()
            engine = self.load_model_now(cfg)
            return engine.train(iterations, log_path, data_yaml=data_yaml)

        return self.tasks.submit(_run, label=label)