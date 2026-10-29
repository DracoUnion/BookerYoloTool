"""Offscreen UI smoke test: main window builds via bootstrap without blocking.

Requires PySide6 + a display-less Qt platform (offscreen). Run with:

    QT_QPA_PLATFORM=offscreen python -m unittest tests/test_ui_smoke.py
"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestUiSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_builds(self):
        from yolo_tool.bootstrap import build_context
        from yolo_tool.mainwindow import YoloMainWindow

        ctx = build_context(install_root=str(ROOT))
        w = YoloMainWindow(ROOT, ctx=ctx)
        self.assertIsNotNone(w.facade)
        self.assertIsNotNone(w.tasks)
        self.assertIn("PySide6 + ultralytics", w.windowTitle())
        w.close()

    def test_task_bridge_wired_to_window(self):
        """A done event published on the main thread must reach the Qt bridge."""
        from yolo_tool.bootstrap import build_context
        from yolo_tool.mainwindow import YoloMainWindow

        ctx = build_context(install_root=str(ROOT))
        w = YoloMainWindow(ROOT, ctx=ctx)
        received = []
        w.bridge.task_done.connect(lambda label, task: received.append((label, task.result)))

        class _T:
            label = "smoke"
            result = "ok"

        # The window's _on_task_event is subscribed to the '*'-channel of the
        # EventHub via tasks.subscribe(...). Publishing on the main thread
        # invokes it synchronously, which re-emits the Qt signal in-thread.
        ctx.events.publish("done", task=_T(), label="smoke", type="done", pct=100)
        w.close()

        self.assertEqual(received, [("smoke", "ok")])

    def test_bootstrap_context_not_none(self):
        from yolo_tool.bootstrap import build_context

        ctx = build_context(install_root=str(ROOT))
        self.assertTrue(ctx.events is not None)
        self.assertTrue(ctx.services is not None)


if __name__ == "__main__":
    unittest.main()