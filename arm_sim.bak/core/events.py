import heapq
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    ROBOT_EMIT = auto()
    ZONE_EMIT = auto()
    ARRIVE_ZONE = auto()
    ZONE_DONE = auto()
    ARRIVE_CENTRAL = auto()
    CENTRAL_DONE = auto()
    POLICY_TICK = auto()


@dataclass(frozen=True)
class Event:
    time: float
    etype: EventType
    payload: Any
    seq: int


class EventQueue:
    def __init__(self):
        self._heap = []

    def push(self, event: Event):
        heapq.heappush(self._heap, (event.time, event.seq, event))

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[2]

    def peek_time(self):
        if not self._heap:
            return None
        return self._heap[0][0]

    def __len__(self):
        return len(self._heap)
