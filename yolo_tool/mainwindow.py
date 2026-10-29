"""
Main application window.

Recreates the MFC single-window YOLO tool as a PySide6 tabbed layout.
All YOLO work now goes through the ultralytics library:

  * 模型选择 (Model) - pick an ultralytics model (.pt/.onnx/.yaml or built-in)
  * 推理/绘制 (Inference / yolo绘制) - run detection, show annotated image
  * 训练 (Training) - run ultralytics model.train()
  * 日志 (Log) - console output from ultralytics
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import ConfigRegistry, YOLOConfig
from .engine import InferenceResult
from .tasks import Task, TaskManager


class TaskBridge(QObject):
    """Adapts zero-Qt TaskManager events into Qt signals (UI thread)."""

    task_done = Signal(str, object)        # label, task
    task_failed = Signal(str, str)         # label, error
    task_progress = Signal(str, str, int)  # label, message, pct


def qimage_from_bgr(img):
    """Convert an OpenCV BGR ndarray to a Qt QImage (deep copy)."""
    import cv2
    from PySide6.QtGui import QImage

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    rgb = img[:, :, ::-1].copy()
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class YoloMainWindow(QMainWindow):
    image_loaded = Signal(Path)

    def __init__(self, install_root=None, *, ctx=None):
        super().__init__()
        self.setWindowTitle("YOLO 综合工具 (PySide6 + ultralytics)")
        self.resize(1080, 720)

        # Bootstrapped services (zero-Qt), injected by YoloTool.
        if ctx is None:
            from .bootstrap import build_context

            ctx = build_context(install_root=install_root or str(Path.cwd()))
        self.ctx = ctx
        self.facade = ctx.services
        self.tasks: TaskManager = ctx.tasks
        self.bridge = TaskBridge()
        self.tasks.subscribe(self._on_task_event)

        self.registry = ctx.extra("registry") or ConfigRegistry(install_root or Path.cwd())
        self.engine = None
        self.current_config: Optional[YOLOConfig] = None
        self.result: Optional[InferenceResult] = None
        self._last_image = None
        self._active_task: Optional[Task] = None

        self._build_ui()
        self._populate_models()
        self._status("就绪 / ready")

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._model_tab(), "模型选择")
        tabs.addTab(self._inference_tab(), "推理 / 绘制")
        tabs.addTab(self._train_tab(), "训练")
        tabs.addTab(self._log_tab(), "日志")
        self.setCentralWidget(tabs)

        self._status_label = QLabel(" ")
        self._model_label = QLabel("未选择模型 / no model")
        sbar = QStatusBar()
        sbar.addWidget(self._model_label, 1)
        sbar.addWidget(self._status_label)
        self.setStatusBar(sbar)

    def _model_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        gb = QGroupBox("模型 (ultralytics)")
        form = QFormLayout(gb)
        self.cmb_model = QComboBox()
        self.cmb_model.setMinimumWidth(380)
        self.cmb_model.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("模型预设:", self.cmb_model)
        self._line_model_file = QLineEdit()
        self._line_model_file.setProperty("role", "modelfile")
        self._line_model_file.setPlaceholderText("本地 .pt / .onnx / .yaml 模型文件")
        form.addRow("模型文件:", self._line_model_file)

        btns = QHBoxLayout()
        self.btn_refresh_models = QPushButton("刷新预设")
        self.btn_refresh_models.clicked.connect(self._populate_models)
        self.btn_select_model = QPushButton("选择模型文件…")
        self.btn_select_model.clicked.connect(self._pick_model_file)
        btns.addWidget(self.btn_refresh_models)
        btns.addWidget(self.btn_select_model)
        btns.addStretch(1)

        lay.addWidget(gb)
        lay.addLayout(btns)
        lay.addStretch(1)
        return w

    def _inference_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        self.btn_open_image = QPushButton("打开图片…")
        self.btn_open_image.clicked.connect(self._open_image)
        self.btn_run = QPushButton("运行检测 (yolo绘制)")
        self.btn_run.clicked.connect(self._run_inference)
        top.addWidget(self.btn_open_image)
        top.addWidget(self.btn_run)
        top.addStretch(1)
        lay.addLayout(top)

        self.lbl_image = QLabel("未加载图片 / no image")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(420, 300)
        self.lbl_image.setStyleSheet("border: 1px solid #888;")
        lay.addWidget(self.lbl_image, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("置信度:"))
        self._prog_slider = QSlider(Qt.Orientation.Horizontal)
        self._prog_slider.setRange(1, 99)
        self._prog_slider.setValue(50)
        self._prog_slider.valueChanged.connect(self._on_conf_changed)
        self._conf_label = QLabel("0.50")
        bottom.addWidget(self._prog_slider, 1)
        bottom.addWidget(self._conf_label)
        bottom.addWidget(QLabel("耗时:"))
        self._time_label = QLabel("-")
        bottom.addWidget(self._time_label)
        lay.addLayout(bottom)
        return w

    def _train_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        gb = QGroupBox("训练 (ultralytics model.train)")
        form = QFormLayout(gb)
        self.spn_iters = QSpinBox()
        self.spn_iters.setRange(1, 1_000_000)
        self.spn_iters.setValue(100)
        self.spn_iters.setSuffix(" 轮/epochs")
        form.addRow("训练轮数:", self.spn_iters)
        self._line_data_yaml = QLineEdit()
        self._line_data_yaml.setPlaceholderText("数据集 YAML (可选, 默认 coco8.yaml)")
        form.addRow("数据集 YAML:", self._line_data_yaml)
        self._line_train_out = QLineEdit("runs/train")
        self._line_train_out.setProperty("role", "trainlog")
        form.addRow("输出目录:", self._line_train_out)

        self.btn_train = QPushButton("开始训练")
        self.btn_train.clicked.connect(self._start_training)
        lay.addWidget(gb)
        lay.addWidget(self.btn_train)
        lay.addStretch(1)
        return w

    def _log_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFontFamily("Consolas")
        lay.addWidget(self._log_edit)
        return w

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _populate_models(self) -> None:
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        self._model_configs: dict[str, YOLOConfig] = {}
        for cfg in self.registry.presets():
            self._model_configs[cfg.name] = cfg
            self.cmb_model.addItem(f"[内置] {cfg.name}")
        self.cmb_model.blockSignals(False)
        if self.cmb_model.count():
            self.cmb_model.setCurrentIndex(0)
            self._on_preset_changed(0)

    def _current_config(self) -> Optional[YOLOConfig]:
        cfg_name = self.cmb_model.currentText()
        if not cfg_name:
            return self.current_config
        # prefer keyed lookup; fall back to matching against display text
        for key, cfg in self._model_configs.items():
            if cfg_name.endswith(key) or cfg_name.startswith(key):
                return cfg
        for cfg in self._model_configs.values():
            if cfg.model_source in cfg_name:
                return cfg
        return self.current_config

    def _on_preset_changed(self, _idx: int) -> None:
        cfg = self._current_config()
        if cfg is None:
            return
        self.current_config = cfg
        self._line_model_file.setText(cfg.model_source)
        self._model_label.setText(f"模型: {cfg.name}")
        self._status(f"已选择模型: {cfg.name}")

    def _status(self, text: str) -> None:
        self._status_label.setText(text)

    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_edit.append(f"[{ts}] {text}")

    def _pick_model_file(self) -> None:
        filt = ("Ultralytics 模型 (*.pt *.onnx *.engine *.tflite *.yaml *.yml)|"
                "*.pt;*.onnx;*.engine;*.tflite;*.yaml;*.yml")
        p = QFileDialog.getOpenFileName(self, "选择模型文件", str(Path.cwd()), filt)[0]
        if p:
            self._line_model_file.setText(p)
            self.current_config = YOLOConfig(
                name=Path(p).stem,
                model_source=p,
                imgsz=640,
            )
            self._model_label.setText(f"模型: {self.current_config.name}")
            self._status(f"已选择本地模型: {p}")

    def _line_text(self, role: str) -> str:
        for it in self.findChildren(QLineEdit):
            if it.property("role") == role:
                return it.text().strip()
        return ""

    def _on_conf_changed(self, v: int) -> None:
        self._conf_label.setText(f"{v / 100:.2f}")

    # ------------------------------------------------------------------ #
    # background task bridge
    # ------------------------------------------------------------------ #
    def _on_task_event(self, ev: dict) -> None:
        """Called from a worker thread: forward to Qt signals (thread-safe)."""
        ev_type = ev["type"]
        task: Optional[Task] = ev.get("task")
        label = task.label if task is not None else ""
        if ev_type == "done":
            self.bridge.task_done.emit(label, task)
        elif ev_type == "failed":
            self.bridge.task_failed.emit(label, ev.get("error", "unknown error"))
        elif ev_type == "progress":
            pct = ev.get("pct", 0)
            msg = ev.get("msg", "")
            self.bridge.task_progress.emit(label, msg, pct)

    def _submit(self, label: str, task: Task) -> Task:
        """Track a background task and wire its completion to the UI."""
        self._active_task = task
        self._status(f"{label} 进行中…")
        self.bridge.task_done.connect(self._on_task_done, Qt.ConnectionType.QueuedConnection)
        self.bridge.task_failed.connect(self._on_task_failed, Qt.ConnectionType.QueuedConnection)
        return task

    def _on_task_done(self, label: str, task: Task) -> None:
        if label == "load-model":
            self.engine = task.result
            self._status(f"模型加载完成: {getattr(self.engine, 'config', None).name if self.engine else '?'}")
        elif label == "inference":
            self.result = task.result
            self._show_result(self.result)
        elif label == "training":
            self._log(f"训练完成: 输出目录 = {getattr(self.engine, 'last_train_dir', None)}")
            self._status("训练完成!")
        self._active_task = None

    def _on_task_failed(self, label: str, error: str) -> None:
        self._log(f"{label} 失败: {error}")
        self._status(f"{label} 失败!")
        QMessageBox.critical(self, f"{label} 失败", error)
        self._active_task = None

    def _show_result(self, result: InferenceResult) -> None:
        ann = qimage_from_bgr(result.annotated())
        self.lbl_image.setPixmap(QPixmap.fromImage(ann))
        conf = self._prog_slider.value() / 100.0
        self._time_label.setText(f"{result.elapsed_ms:.0f} ms")
        self._status(f"检测到 {len(result.detections)} 个目标 @ {conf:.2f}")
        self._log(
            f"推理完成: {len(result.detections)} 目标, 置信度 {conf:.2f}, "
            f"耗时 {result.elapsed_ms:.1f} ms"
        )

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #
    def _open_image(self) -> None:
        filt = "所有图片文件 (*.JPG;*.PNG;*.BMP;*.GIF;*.ICO;*.CUR)|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.ico;*.cur"
        p = QFileDialog.getOpenFileName(self, "打开图片", str(Path.cwd()), filt)[0]
        if p:
            self._load_image(Path(p))

    def _load_image(self, path: Path) -> None:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            self._log(f"无法读取图片: {path}")
            return
        self._last_image = img
        self.lbl_image.setPixmap(QPixmap.fromImage(qimage_from_bgr(img)))
        self.lbl_image.scaledContents = True
        self._status(f"已加载图片: {path.name}")
        self.image_loaded.emit(path)

    def _run_inference(self) -> None:
        cfg = self._current_config() or self.current_config
        # if the user browsed a model file line directly, use it
        file_line = self._line_text("modelfile")
        if file_line and (cfg is None or cfg.model_source != Path(file_line).name):
            cfg = YOLOConfig(name=Path(file_line).stem, model_source=file_line, imgsz=640)
        if cfg is None:
            QMessageBox.information(self, "提示", "请先选择模型。")
            return
        self.current_config = cfg
        if self._last_image is None:
            QMessageBox.information(self, "提示", "请先打开一张图片。")
            return

        # Ensure a model is loaded in the background first, then run inference.
        if self.engine is None or self.engine.model is None:
            task = self.facade.load_model(cfg, label="load-model")
        else:
            task = None

        if task is not None:
            self._submit("模型加载", task)
            # Chain inference after model load completes.
            self.bridge.task_done.connect(
                lambda label, t: self._chain_inference(label, t, cfg),
                Qt.ConnectionType.QueuedConnection,
            )
        else:
            self._run_inference_now(cfg)

    def _chain_inference(self, label: str, task: Task, cfg: YOLOConfig) -> None:
        if label != "load-model" or task.state.name != "DONE":
            return
        self.engine = task.result
        self._run_inference_now(cfg)

    def _run_inference_now(self, cfg: YOLOConfig) -> None:
        if self._last_image is None:
            return
        conf = self._prog_slider.value() / 100.0
        self._submit("推理", self.facade.infer(self._last_image, conf, label="inference"))

    def _start_training(self) -> None:
        cfg = self._current_config() or self.current_config
        if cfg is None:
            QMessageBox.information(self, "提示", "请先选择模型。")
            return

        iters = self.spn_iters.value()
        data_yaml = self._line_data_yaml.text().strip() or None
        out_dir = self._line_train_out.text().strip() or "runs/train"
        log_path = Path(out_dir)
        data = Path(data_yaml) if data_yaml else None
        self._log(f"开始训练 {cfg.name}, epochs={iters}, 输出={out_dir}")

        # Training runs off the UI thread; TaskManager emits done/failed
        # events that the bridge forwards as Qt signals.
        task = self.facade.train(cfg, iters, log_path, data_yaml=data, label="training")
        self._submit("训练", task)


class YoloTool:
    """Application wrapper: builds the service graph once and hands it to the UI."""

    def __init__(self, install_root: Optional[Path] = None, *, ctx=None):
        from PySide6.QtWidgets import QApplication

        from .bootstrap import build_context

        self.install_root = install_root or Path.cwd()
        self.ctx = ctx or build_context(install_root=str(self.install_root))
        self.app = QApplication.instance() or QApplication([])
        self.window = YoloMainWindow(self.install_root, ctx=self.ctx)

    def run(self) -> int:
        self.window.show()
        return self.app.exec()


def main() -> int:
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    tool = YoloTool(root)
    return tool.run()


if __name__ == "__main__":
    raise SystemExit(main())