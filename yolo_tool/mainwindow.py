"""
Main application window.

Recreates the MFC single-window YOLO tool as a PyQt6 tabbed layout.
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

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
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
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import ConfigRegistry, YOLOConfig
from .engine import InferenceResult, make_engine


def qimage_from_bgr(img):
    """Convert an OpenCV BGR ndarray to a Qt QImage (deep copy)."""
    import cv2
    from PyQt6.QtGui import QImage

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    rgb = img[:, :, ::-1].copy()
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class YoloMainWindow(QMainWindow):
    image_loaded = pyqtSignal(Path)

    def __init__(self, install_root: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("YOLO 综合工具 (PyQt6 + ultralytics)")
        self.resize(1080, 720)

        self.registry = ConfigRegistry(install_root or Path.cwd())
        self.engine = None
        self.current_config: Optional[YOLOConfig] = None
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
        self._model_label = QLabel("未选择模型 / no model")
        sbar = QStatusBar()
        sbar.addWidget(self._model_label, 1)
        sbar.addWidget(self._status_label)
        self.setStatusBar(sbar)

    def _model_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        gb = QGroupBox("模型 (ultralytics: .pt / .onnx / .yaml)")
        form = QFormLayout(gb)
        self.cmb_model = QComboBox()
        self.cmb_model.setMinimumWidth(380)
        self.cmb_model.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("模型预设:", self.cmb_model)
        self._line_model_file = QLineEdit()
        self._line_model_file.setProperty("role", "modelfile")
        self._line_model_file.setPlaceholderText("本地 .pt / .onnx / .yaml 模型文件")
        form.addRow("模型文件:", self._line_model_file)
        self._line_names = QLineEdit()
        self._line_names.setProperty("role", "names")
        form.addRow("NAMES 文件:", self._line_names)  # display-only best-effort

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
            mark = "[内置]" if cfg.is_builtin() else "[OK]"
            self._model_configs[cfg.name] = cfg
            self.cmb_model.addItem(f"{mark} {cfg.name}")
        for cfg in self.registry.scan_runtime_models():
            mark = "[本地]" if cfg.is_valid() else "[..]"
            self._model_configs[f"local:{cfg.name}"] = cfg
            self.cmb_model.addItem(f"{mark} {cfg.name}: {cfg.model_source}")
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
        names = self.registry.discover_names("v4" if "v4" in cfg.name else "v3")
        self._line_names.setText(str(names) if names else "")
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

        try:
            self.engine = make_engine(cfg)
            self._status(f"正在加载模型 {cfg.name} (首次使用会自动下载)...")
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
        cfg = self._current_config() or self.current_config
        if cfg is None:
            QMessageBox.information(self, "提示", "请先选择模型。")
            return
        try:
            self.engine = make_engine(cfg)
            self.engine.load()
        except Exception as exc:
            self._log(f"引擎初始化失败: {exc}")
            QMessageBox.critical(self, "引擎初始化失败", str(exc))
            return

        iters = self.spn_iters.value()
        data_yaml = self._line_data_yaml.text().strip() or None
        out_dir = self._line_train_out.text().strip() or "runs/train"
        log_path = Path(out_dir)
        data = Path(data_yaml) if data_yaml else None
        self._log(f"开始训练 {cfg.name}, epochs={iters}, 输出={out_dir}")
        try:
            self.engine.train(iters, log_path, data_yaml=data)
            save_dir = getattr(self.engine, "last_train_dir", None)
            self._log(f"训练完成: 输出目录 = {save_dir}")
            out = (save_dir / "weights" / "best.pt") if save_dir else None
            self._status("训练完成!" if out and out.exists() else "训练完成 (输出目录未含 best.pt)")
        except Exception as exc:
            self._log(f"训练失败: {exc}")
            QMessageBox.critical(self, "训练失败", str(exc))


class YoloTool:
    """Application bootstrap (thin wrapper around the main window)."""

    def __init__(self, install_root: Optional[Path] = None):
        from PyQt6.QtWidgets import QApplication

        self.install_root = install_root or Path.cwd()
        self.app = QApplication.instance() or QApplication([])
        self.window = YoloMainWindow(self.install_root)

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