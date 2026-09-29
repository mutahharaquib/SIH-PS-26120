"""In-process pub/sub for state updates. Modules couple only through published state."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable

Handler = Callable[[str, Any], None]


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._latest: dict[str, Any] = {}
        self._lock = threading.RLock()

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Subscribe to a topic ('*' = all). Returns an unsubscribe function."""
        with self._lock:
            self._subs[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._subs[topic]:
                    self._subs[topic].remove(handler)

        return unsubscribe

    def publish(self, topic: str, payload: Any) -> None:
        with self._lock:
            self._latest[topic] = payload
            handlers = list(self._subs.get(topic, [])) + list(self._subs.get("*", []))
        for h in handlers:
            h(topic, payload)

    def latest(self, topic: str) -> Any:
        with self._lock:
            return self._latest.get(topic)


bus = Bus()
