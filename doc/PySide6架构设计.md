# BookerYoloTool 架构设计

> YOLO 综合工具（PySide6 + ultralytics）。本文件描述 PySide6 迁移与参考架构对齐后的当前代码结构（2026-10-30）。

## 1. 项目简介

将 32 位 MFC 工具 `yolo综合工具X0215.exe` 以 Python + **PySide6** 重写为桌面应用，
所有 YOLO 调用经 ultralytics（YOLOv8 / YOLO11），提供模型选择、推理/绘制、训练、日志四个页签。

变更历史：
- `e01f2c2` 迁移 PyQt6 → PySide6。
- `e0e238a` 新增零 Qt 架构层（common/tasks/services/bootstrap）并把模型加载/推理/训练任务化，
  修复"长任务阻塞 UI 线程"的头号反例。

## 2. 技术栈与依赖

| 项 | 值 |
|---|---|
| Qt 绑定 | PySide6 ≥ 6.5（`pyproject.toml`） |
| 推理/训练 | ultralytics ≥ 8.0（拉取 torch、opencv） |
| 图像 | opencv-python ≥ 4.5、numpy |
| 配置 | `YOLOConfig` dataclass + `ConfigRegistry`（内置模型清单 + 本地文件） |
| 测试 | pytest ≥ 7 + unittest（引擎单测 + 零 Qt 架构层测试 + offscreen UI smoke） |

## 3. 目录结构

```text
BookerYoloTool/
├─ pyproject.toml               # console script yolo-tool = yolo_tool_gui:main
├─ yolo_tool_gui.py             # 入口：UTF-8 重配置 → YoloTool(root).run()
├─ yolo_tool/
│  ├─ bootstrap.py              # build_context()：EventHub + TaskManager + ServiceFacade → AppContext
│  ├─ common/                   # ★ 零 Qt
│  │  ├─ events.py              # EventHub
│  │  ├─ context.py             # AppContext
│  │  ├─ exceptions.py          # YoloToolError/ModelError/InferenceError/TrainError
│  │  └─ __init__.py
│  ├─ tasks/                    # ★ 零 Qt
│  │  ├─ task.py                # Task/TaskState/TaskManager + TaskCancelled
│  │  └─ __init__.py
│  ├─ services/                 # ★ 零 Qt
│  │  ├─ facade.py              # ServiceFacade：load_model / infer / train（后台 Task）
│  │  └─ __init__.py
│  ├─ config.py                 # YOLOConfig + ConfigRegistry（零 Qt）
│  ├─ engine.py                 # YOLOEngine（ultralytics）+ Detection/InferenceResult（薄引擎）
│  └─ mainwindow.py             # PySide6：TaskBridge + YoloMainWindow + YoloTool 包装
├─ tests/
│  ├─ test_engine.py            # 引擎/检测/配置单测
│  ├─ test_tasks.py             # EventHub/TaskManager 零 Qt 单测
│  └─ test_ui_smoke.py          # offscreen 主窗口构建 + 任务桥接线验证
└─ runtime/                     # 原 MFC exe/DLL/训练集（保留参考，不参与执行，gitignored）
```

## 4. 架构分层与依赖方向

```text
ui/ (mainwindow.py, 唯一 PySide6)
 └─ services/facade.py  (ServiceFacade：load/infer/train → Task)
     ├─ tasks/  TaskManager（线程池执行）
     └─ engine.py  YOLOEngine（纯算法，零 Qt）
         └─ config.py  YOLOConfig/ConfigRegistry
```

## 5. 关键流程（修阻塞反例）

```text
yolo_tool_gui.py → YoloTool(root).run()
```

- `YoloTool` 通过 `bootstrap.build_context()` 装配一次服务图，注入主窗口。
- **模型加载**：`facade.load_model(cfg, label="load-model")` → 后台下载/加载 → `task_done` 信号。
- **推理**：`facade.infer(image, conf, label="inference")` → 后台 `engine.infer()` →
  `bridge.task_done` → `_show_result()` 渲染标注图。
- **训练**：`facade.train(...)` → 后台 `engine.train()` → `bridge.task_done/_task_failed`。

`TaskBridge(QObject)` 把零 Qt `TaskManager` 事件转为 Qt 信号（`task_done/task_failed/`），
UI 线程只收信号，主窗口在模型加载/推理/训练期间保持响应（可继续操作界面）。

## 6. 线程模型

| 线程 | 职责 |
|---|---|
| Qt 主线程 | 窗口/控件、信号槽（桥梁在 UI 线程） |
| TaskManager 池（零 Qt） | load / infer / train 后台执行；协作式 `cancel_event` |

## 7. 配置管理

- 模型配置：`YOLOConfig`（name/model_source/imgsz/task）+ `ConfigRegistry.presets()`（内置 yolov8/yolo11）。
- 本地模型：文件选择对话框（.pt/.onnx/.engine/.yaml），`_pick_model_file` 构建配置。
- 训练输出：`runs/`（ultralytics 默认，gitignored）。
- 当前无 QSettings 持久化窗口偏好（可后续归入 `ui/settings_store.py`）。

## 8. 测试

```bash
.venv/Scripts/python.exe tests/test_engine.py   # 引擎/检测框/配置
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests # 含 tasks + UI smoke
```

## 9. 与参考架构的差距（后续 TODO）

- 无 SQLAlchemy / 持久化（无任务历史需求，不强引 ORM）。
- 无 QSettings / `.env` 配置分层；模型源硬编码在 `BUILTIN_MODELS`。
- `mainwindow.py` 仍承担部分 UI 装配逻辑，可进一步拆 widgets/pages。
- 训练进度（epoch）尚未经 TaskManager progress 事件上抛（只有完成/失败）。