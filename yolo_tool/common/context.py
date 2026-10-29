"""
Application context (zero Qt).

``AppContext`` is assembled by :func:`yolo_tool.bootstrap.build_context` and
handed to the UI layer. The UI never imports Qt into services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from yolo_tool.common.events import EventHub

if TYPE_CHECKING:
    from yolo_tool.services.facade import ServiceFacade
    from yolo_tool.tasks import TaskManager


@dataclass
class AppContext:
    """Bundle of run-time services injected at bootstrap time."""

    events: EventHub
    tasks: "TaskManager"
    services: "ServiceFacade"
    extras: dict[str, Any] = field(default_factory=dict)

    def extra(self, key: str, default: Any = None) -> Any:
        """Read an optional bootstrap-injected dependency."""
        return self.extras.get(key, default)