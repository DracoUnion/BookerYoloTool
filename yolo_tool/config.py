"""
Configuration management for the YOLO tool.

Mirrors the original MFC app's 运行库 (runtime library) layout:

  运行库\
    cfg\yolov3.cfg
    cfg\yolov3-tiny.cfg
    data\yolov3.names
    data\yolov3.data
  运行库V4\
    cfg\yolov4-tiny-dota2.cfg
    data\yolov4-tiny-dota2_last.weights
    yolo_train_dota2.cmd
    startcmd.exe
    dota2\...

All paths are resolved relative to an install root, so the tool can be
shipped with a bundled runtime dir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Directory names used by the original installer
RUNTIME_DIR_V3 = "运行库"
RUNTIME_DIR_V4 = "运行库V4"
DATA_DIR = "data"
CFG_DIR = "cfg"

# Supported image formats (from the original Open/Save dialog filter)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".ico", ".cur")


@dataclass
class YOLOConfig:
    """Describes a YOLO model (darknet) variant to load / train / run."""

    name: str
    version: str  # "v3" | "v4"
    cfg_path: Path
    names_path: Path
    weights_path: Optional[Path] = None
    data_config_path: Optional[Path] = None  # *.data (train cfg)
    runtime_root: Optional[Path] = None
    train_command: Optional[Path] = None      # e.g. yolo_train_dota2.cmd
    extra: dict = field(default_factory=dict)

    @property
    def runtime_dir(self) -> Path:
        if self.runtime_root is not None:
            return self.runtime_root
        return Path(self.version)

    def is_valid(self) -> bool:
        """Cfg + names must exist for inference; weights optional (train)."""
        return self.cfg_path.exists() and self.names_path.exists()


class ConfigRegistry:
    """Discovers / registers YOLO model configs under an install root."""

    def __init__(self, install_root: Optional[Path] = None):
        self.install_root = Path(install_root) if install_root else Path.cwd()

    # -- discovery --------------------------------------------------------
    def discover_default_v3(self) -> YOLOConfig:
        root = self.install_root / RUNTIME_DIR_V3
        return YOLOConfig(
            name="yolov3",
            version="v3",
            cfg_path=root / CFG_DIR / "yolov3.cfg",
            names_path=root / DATA_DIR / "yolov3.names",
            weights_path=root / "yolov3.weights",
            data_config_path=root / DATA_DIR / "yolov3.data",
            runtime_root=root,
        )

    def discover_default_v4(self) -> YOLOConfig:
        root = self.install_root / RUNTIME_DIR_V4
        data_dir = root / "dota2"
        return YOLOConfig(
            name="yolov4-tiny-dota2",
            version="v4",
            cfg_path=root / CFG_DIR / "yolov4-tiny-dota2.cfg",
            names_path=data_dir / "dota2.names",
            weights_path=data_dir / "yolov4-tiny-dota2_last.weights",
            data_config_path=data_dir / "dota2.data",
            train_command=root / "yolo_train_dota2.cmd",
            runtime_root=root,
        )

    def load_cfg_by_path(self, cfg_path: str | Path) -> YOLOConfig:
        """Build a config from an explicit .cfg path (user-selected)."""
        cfg = Path(cfg_path)
        # names sits in a sibling data dir
        d = cfg.parent
        names = d.parent / DATA_DIR / f"{cfg.stem}.names"
        if not names.exists():
            names = d / f"{cfg.stem}.names"
        weights = d / f"{cfg.stem}.weights"
        return YOLOConfig(
            name=cfg.stem,
            version="v3" if "v3" in cfg.stem else "v4",
            cfg_path=cfg,
            names_path=names,
            weights_path=weights if weights.exists() else None,
        )
