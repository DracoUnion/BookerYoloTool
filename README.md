# YOLO 综合工具 (PyQt6 + ultralytics 重写)

A Python + PyQt6 rewrite of `yolo综合工具X0215.exe`, a 32-bit MFC tool
for YOLO training and inference. **All YOLO calls now use the
[ultralytics](https://github.com/ultralytics/ultralytics) library**
(YOLOv8 / YOLO11), replacing the original darknet runtime.

## 功能 / Features

* **模型选择** — ultralytics 模型 (`yolov8n/s/m/l/x`, `yolo11n/s`) 自动下载,
  或选择本地 `.pt` / `.onnx` / `.engine` / `.yaml` 文件
* **推理 / 绘制 (yolo绘制)** — 加载图片 → 检测 → 标注边框+标签 → 耗时统计
* **训练** — `model.train()`，支持数据集 YAML（默认 coco8），输出到 `runs/`
* **日志** — ultralytics 训练/推理输出

## 环境 / Setup

The rewrite needs a Python ≥3.10 environment with ultralytics. The original
MFC binary + `运行库`/`运行库V4` are kept in `runtime/` for dataset/class-name
reference, but the YOLO engine itself is pure ultralytics.

```bash
# create venv + install deps
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .

# run GUI (optionally pass install root that contains runtime/)
.venv/Scripts/python.exe yolo_tool_gui.py [install_root]
```

## 目录结构 / Layout

```
yolo_tool/
  config.py      # YOLOConfig + ConfigRegistry (ultralytics presets + runtime scan)
  engine.py      # YOLOEngine (ultralytics): load / infer / train
  mainwindow.py  # PyQt6 tabbed UI: 模型选择 / 推理·绘制 / 训练 / 日志
runtime/         # original MFC exe + 运行库/运行库V4 (gitignored binaries)
yolo_tool_gui.py # console entry point
tests/test_engine.py
```

## 测试 / Tests

```bash
.venv/Scripts/python.exe tests/test_engine.py
# or
pytest tests/
```

## 依赖 / Dependencies

`PyQt6`, `ultralytics` (pulls torch, opencv). Installed in the venv to
avoid touching the system Python that IDA's MCP server uses.

## 原始程序逆向信息 / Original RE notes

* 32-bit MFC app; entry `start`(0x51f8ce) → `WinMain`(0x52ee54) →
  `sub_5382BC`(0x5382bc) → `CWinApp::Run`(0x536991)
* Runtime dirs `运行库`/`运行库V4` hold darknet cfg/names/data + DLLs
* Original image filter: `*.JPG;*.PNG;*.BMP;*.GIF;*.ICO;*.CUR`
