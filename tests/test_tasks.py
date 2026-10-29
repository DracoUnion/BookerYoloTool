"""Unit tests for the zero-Qt task layer (yolo_tool.tasks / common.events).

These never import PySide6, so they run headless.
"""

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yolo_tool.common.events import EventHub  # noqa: E402
from yolo_tool.tasks import TaskManager, TaskState  # noqa: E402


class TestEventHub(unittest.TestCase):

    def test_sub_and_publish(self):
        hub = EventHub()
        seen = []
        unsub = hub.subscribe("demo", lambda ev: seen.append(ev))
        hub.publish("demo", x=1)
        unsub()
        hub.publish("demo", x=2)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["type"], "demo")
        self.assertEqual(seen[0]["x"], 1)

    def test_wildcard_subscriber(self):
        hub = EventHub()
        seen = []
        hub.subscribe("*", lambda ev: seen.append(ev["type"]))
        hub.publish("a", v=1)
        hub.publish("b", v=2)
        self.assertEqual(seen, ["a", "b"])


class TestTaskManager(unittest.TestCase):

    def test_submit_runs_fn_and_emits_done(self):
        tasks = TaskManager(max_workers=2)
        events = []

        def record(ev):
            events.append(ev["type"])

        tasks.subscribe(record)

        def work(*, cancel_event, **kwargs):
            time.sleep(0.05)
            return 42

        task = tasks.submit(work, label="add")
        deadline = time.time() + 3
        while task.state not in (TaskState.DONE, TaskState.FAILED) and time.time() < deadline:
            time.sleep(0.01)
        tasks.shutdown()

        self.assertEqual(task.result, 42)
        self.assertIn("started", events)
        self.assertIn("done", events)

    def test_failed_task_sets_error(self):
        tasks = TaskManager(max_workers=2)

        def boom(*, cancel_event, **kwargs):
            raise ValueError("boom")

        task = tasks.submit(boom, label="fail")
        deadline = time.time() + 2
        while task.state not in (TaskState.DONE, TaskState.FAILED) and time.time() < deadline:
            time.sleep(0.01)
        tasks.shutdown()
        self.assertEqual(task.state, TaskState.FAILED)
        self.assertEqual(task.error, "boom")

    def test_cancel_sets_cooperative_flag(self):
        tasks = TaskManager(max_workers=1)
        task = tasks.submit(
            lambda *, cancel_event, **kwargs: cancel_event.wait(2),
            label="block",
        )
        time.sleep(0.05)
        ok = tasks.cancel(task.id)
        # Cooperative cancellation: for a RUNNING task the flag is set; the
        # fn observes it on its next check. The flag is the contract.
        self.assertTrue(ok)
        self.assertTrue(task.cancel_event.is_set())
        tasks.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()