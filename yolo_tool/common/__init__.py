"""
Common zero-Qt building blocks for the YOLO tool.

This package never imports PySide6, so the service/task layers stay fully
headless-testable. Only ``yolo_tool.mainwindow`` (the UI) may touch Qt.
"""

from yolo_tool.common.exceptions import YoloToolError
from yolo_tool.common.events import EventHub
from yolo_tool.common.context import AppContext

__all__ = ["YoloToolError", "EventHub", "AppContext"]