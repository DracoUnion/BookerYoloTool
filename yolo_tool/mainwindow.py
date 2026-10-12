"""
Main application window.

Recreates the MFC single-window YOLO tool as a PyQt6 tabbed layout:

  * 模型选择 (Model) - pick .cfg/.names/.weights, browse images
  * 推理/绘制 (Inference / yolo绘制) - run detection, show annotated image
  * 训练 (Training) - run darknet train via the bundled .cmd
  * 日志 (Log) - console output from darknet / subprocesses
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
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
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import ConfigRegistry, IMAGE_EXTENSIONS
from .engine import (
    Detection,
    InferenceResult,
    make_engine,
    YOLOEngine,
)


def qimage_from_bgr(img) -> QImage:
    """Convert an OpenCV BGR ndarray to a Qt QImage (deep copy)."""
    import numpy as np

    if img.ndim == 2:
        h, w = img.shape
        rgb = cv2_grayscale_to_rgb(img)
        return qimage_from_bgr(rgb)
    h, w, c = img.shape
    rgb = img[:, :, ::-1].copy()
    return QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888).copy()


def cv2_grayscale_to_rgb(g):
    import cv2

    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class YoloMainWindow(QMainWindow):
    image_loaded = pyqtSignal(Path)

    def __init__(self, install_root: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("YOLO 综合工具 (PyQt6)")
        self.resize(1080, 720)

        self.registry = ConfigRegistry(install_root or Path.cwd())
        self.engine: Optional[YOLOEngine] = None
        self.current_config = None
        self.result: Optional[InferenceResult] = None
        self._last_image = None

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
        self._prog = self._prog_slider
        self._time_label = QLabel("耗时: -")
        self._model_label = QLabel("未选择模型 / no model")
        sbar = QStatusBar()
        sbar.addWidget(self._model_label, 1)
        sbar.addWidget(self._status_label)
        self.setStatusBar(sbar)

    def _model_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        gb = QGroupBox("模型 / 运行库 (cfg + names + weights)")
        form = QFormLayout(gb)
        self.cmb_model = QComboBox()
        self.cmb_model.setMinimumWidth(260)
        form.addRow("模型预设:", self.cmb_model)
        form.addRow("CFG 文件:", self._browse_line("cfg"))
        form.addRow("NAMES 文件:", self._browse_line("names"))
        form.addRow("WEIGHTS 文件:", self._browse_line("weights"))

        btns = QHBoxLayout()
        self.btn_refresh_models = QPushButton("刷新预设")
        self.btn_refresh_models.clicked.connect(self._populate_models)
        self.btn_select_cfg = QPushButton("选择 .cfg…")
        self.btn_select_cfg.clicked.connect(lambda: self._pick("cfg", "模型文件(*.cfg)", "模型文件 (*.cfg)|*.cfg"))
        self.btn_select_weights = QPushButton("选择 .weights / .pt…")
        self.btn_select_weights.clicked.connect(
            lambda: self._pick("weights", "权重文件", "权重文件 (*.weights;*.pt;*.onnx)|*.weights;*.pt;*.onnx")
        )
        btns.addWidget(self.btn_refresh_models)
        btns.addWidget(self.btn_select_cfg)
        btns.addWidget(self.btn_select_weights)
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
        gb = QGroupBox("训练 (darknet train)")
        form = QFormLayout(gb)
        self.spn_iters = QSpinBox()
        self.spn_iters.setRange(1, 1_000_000)
        self.spn_iters.setValue(500)
        self.spn_iters.setSuffix(" 轮")
        form.addRow("训练轮数:", self.spn_iters)
        self.chk_clear = QPushButton("启用 -clear (从零训练)")
        self.chk_clear.setCheckable(True)
        form.addRow("", self.chk_clear)
        self._line_train_log = QLineEdit()
        self._line_train_log.setProperty("role", "trainlog")
        form.addRow("输出日志:", self._line_train_log)

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
    def _browse_line(self, role: str) -> QLineEdit:
        e = QLineEdit()
        e.setProperty("role", role)
        return e

    def _populate_models(self) -> None:
        self.cmb_model.clear()
        for cfg in (self.registry.discover_default_v3(), self.registry.discover_default_v4()):
            mark = "[OK]" if cfg.is_valid() else "[..]"
            self.cmb_model.addItem(f"{mark} {cfg.name} ({cfg.version})  {cfg.cfg_path}")
        if self.cmb_model.count():
            self.cmb_model.setCurrentIndex(0)

    def _status(self, text: str) -> None:
        self._status_label.setText(text)

    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_edit.append(f"[{ts}] {text}")

    def _pick(self, role: str, caption: str, filter: str) -> None:
        p = QFileDialog.getOpenFileName(self, caption, str(Path.cwd()), filter)[0]
        if p:
            self._set_line(role, p)
            if role == "cfg":
                self._on_cfg_changed(p)

    def _set_line(self, role: str, text: str) -> None:
        for it in self.findChildren(QLineEdit):
            if it.property("role") == role:
                it.setText(text)
                break

    def _line_text(self, role: str) -> str:
        for it in self.findChildren(QLineEdit):
            if it.property("role") == role:
                return it.text().strip()
        return ""

    def _on_cfg_changed(self, cfg_path: str) -> None:
        try:
            self.current_config = self.registry.load_cfg_by_path(cfg_path)
            self._model_label.setText(f"模型: {self.current_config.name}  ({self.current_config.version})")
            self._status(f"已加载模型: {self.current_config.name}")
        except Exception as exc:
            self._log(f"配置加载失败: {exc}")

    def _on_conf_changed(self, v: int) -> None:
        self._conf_label.setText(f"{v / 100:.2f}")

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #
    def _open_image(self) -> None:
        filt = "所有图片文件 (*.JPG;*.PNG;*.BMP;*.GIF;*.ICO;*.CUR)|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.ico;*.cur"
        p = QFileDialog.getOpenFileName(self, "打开图片", str(Path.cwd()), filt)[0]
        if not p:
            return
        self._set_line("cfg", self._line_text("cfg"))  # no-op keep lines
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
        if self._last_image is None:
            QMessageBox.information(self, "提示", "请先打开一张图片。")
            return
        if self.current_config is None:
            cfg_txt = self._line_text("cfg")
            if cfg_txt:
                self._on_cfg_changed(cfg_txt)
            else:
                QMessageBox.information(self, "提示", "请先选择模型 .cfg。")
                return
        try:
            self.engine = make_engine(self.current_config)
            self.engine.load()
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", str(exc))
            self._log(f"模型加载失败: {exc}")
            return

        t0 = time.perf_counter()
        conf = self._prog_slider.value() / 100.0
        try:
            self.result = self.engine.infer(self._last_image, conf)
        except Exception as exc:
            self._log(f"推理失败: {exc}")
            QMessageBox.critical(self, "推理失败", str(exc))
            return
        self._time_label.setText(f"{self.result.elapsed_ms:.0f} ms")
        ann = qimage_from_bgr(self.result.annotated())
        self.lbl_image.setPixmap(QPixmap.fromImage(ann))
        self._status(f"检测到 {len(self.result.detections)} 个目标 @ {conf:.2f}")
        self._log(
            f"推理完成: {len(self.result.detections)} 目标, 置信度 {conf:.2f}, "
            f"耗时 {self.result.elapsed_ms:.1f} ms"
        )

    def _start_training(self) -> None:
        if self.current_config is None:
            QMessageBox.information(self, "提示", "请先选择模型。")
            return
        try:
            self.engine = make_engine(self.current_config)
        except Exception as exc:
            self._log(f"引擎初始化失败: {exc}")
            return
        iters = self.spn_iters.value()
        log_path = Path(self._line_text("trainlog") or f"train_{self.current_config.name}.log")
        self._log(f"开始训练 {self.current_config.name}, 轮数={iters}")
        try:
            proc = self.engine.train(iters, log_path)
            tail = ""
            if log_path.exists():
                tail = log_path.read_text(encoding="utf-8", errors="ignore")[-300:]
            self._log(f"训练进程退出码 {proc.returncode}")
            self._log(tail or "(无日志输出)")
        except NotImplementedError:
            self._log("当前引擎不支持训练 (纯推理后端)。请配置 darknet 运行库。")
        except Exception as exc:
            self._log(f"训练失败: {exc}")


class YoloTool:
    """Application bootstrap (thin wrapper around the main window)."""

    def __init__(self, install_root: Optional[Path] = None):
        self.install_root = install_root or Path.cwd()
        self.window = YoloMainWindow(self.install_root)

    def run(self) -> int:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QApplication as _CA  # noqa: F401

        app = QApplication.instance() or QApplication([])
        self.window.show()
        return app.exec()


def main() -> int:
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    tool = YoloTool(root)
    return tool.run()


if __name__ == "__main__":
    raise SystemExit(main())
