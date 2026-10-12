# YOLO 综合工具 (PyQt6 重写)

A Python + PyQt6 rewrite of `yolo综合工具X0215.exe`, a 32-bit MFC tool
for YOLO v3 / YOLO v4 training and inference that wraps the darknet
runtime (`运行库` / `运行库V4`).

## Original (MFC) structure — from reverse engineering

| MFC symbol | Address | Role |
|---|---|---|
| `start` | `0x51f8ce` | CRT bootstrap |
| `_WinMain@16` | `0x52ee54` | entry → `sub_5382BC` |
| `sub_5382BC` | `0x5382bc` | `AfxWinInit` + app `Run` |
| `sub_53CB59` | `0x53cb59` | `AfxWinInit` |
| `sub_53703B` | `0x53703b` | `AfxInitThread` (installs msg filter) |
| `CWinApp::Run` | `0x536991` | message loop (`sub_5370AC`) |
| app ctor | `0x47e890` | `CWinApp` subclass (vtable `off_427A788`) |

Features observed in the binary:
- `.cfg` / `.names` / `.weights` model selection
- training via `yolo_train_dota2.cmd` / `startcmd.exe` / `darknet.cfg`
- inference + box visualization ("yolo绘制" / "yolo绘制G")
- image filter `*.JPG;*.PNG;*.BMP;*.GIF;*.ICO;*.CUR`
- `CCheckListBox` option panels

## Layout

```
yolo_tool/
  config.py      # YOLOConfig, ConfigRegistry (运行库 / 运行库V4 layout)
  engine.py      # YOLOEngine base, DarknetEngine, TorchYOLOEngine
  mainwindow.py  # PyQt6 tabbed UI: 模型选择 / 推理·绘制 / 训练 / 日志
yolo_tool_gui.py # console entry point
tests/test_engine.py
```

## Run

```bash
# GUI (pass install root that contains 运行库 / 运行库V4)
python yolo_tool_gui.py /path/to/install_root

# or, default cwd
python yolo_tool_gui.py
```

## Tests

```bash
python tests/test_engine.py
# or
pytest tests/
```

## Dependencies

`PyQt6`, `opencv-python`, `numpy`, `torch` (optional, for the
`TorchYOLOEngine` back-end). `DarknetEngine` shells out to a darknet
binary / batch script, matching the original's runtime-library behavior.

## Notes

- Two inference back-ends: `DarknetEngine` (faithful to original, requires a
  `darknet.exe` in the runtime dir) and `TorchYOLOEngine` (pure PyTorch
  fallback, loads a `.pt`/`.onnx` model).
- The concrete MFC `InitInstance` / main-window class was not a named symbol
  in the IDB (reached via the `CWinApp` vtable). The PyQt6 rewrite
  reconstructs the observable behavior (model selection, train, infer/draw,
  log console) rather than 1:1 widget identity.
