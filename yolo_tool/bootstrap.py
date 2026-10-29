"""
Application bootstrap for the YOLO tool.

Assembles the zero-Qt service graph (EventHub + TaskManager + ServiceFacade)
into an :class:`AppContext` and returns it. The UI receives the context and
adapts task events into Qt signals; it never constructs services itself.
"""

from __future__ import annotations

from typing import Any, Optional

from yolo_tool.common.context import AppContext
from yolo_tool.common.events import EventHub
from yolo_tool.config import ConfigRegistry
from yolo_tool.services.facade import ServiceFacade
from yolo_tool.tasks import TaskManager


def build_context(
    *,
    install_root: Optional[str] = None,
    max_workers: int = 2,
    extras: Optional[dict[str, Any]] = None,
) -> AppContext:
    """Assemble services and return an :class:`AppContext`."""
    events = EventHub()
    tasks = TaskManager(max_workers=max_workers, events=events)
    registry = ConfigRegistry(install_root or ".")

    services = ServiceFacade(
        config_provider=registry,
        events=events,
        tasks=tasks,
    )
    ctx = AppContext(
        events=events,
        tasks=tasks,
        services=services,
        extras=extras or {},
    )
    ctx.extras.setdefault("registry", registry)
    return ctx