"""Console entry point:  python yolo_tool_gui.py [install_root]"""

import sys
from pathlib import Path

from yolo_tool.mainwindow import YoloTool


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    return YoloTool(root).run()


if __name__ == "__main__":
    raise SystemExit(main())
