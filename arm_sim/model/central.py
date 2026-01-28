from collections import deque
import random

from ..core.events import EventType


class CentralSupervisor:
    def __init__(self, sim, recorder, service_rate_msgs_s, rng=None, service_time_jitter=0.0):
        self.sim = sim
        self.recorder = recorder
        self.service_rate_msgs_s = float(service_rate_msgs_s)
        self.rng = rng or random.Random(0)
        self.service_time_jitter = float(service_time_jitter)
        self.queue = deque()
        self.busy = False
        self.busy_time = 0.0
        self._busy_start = None
        self.max_queue_len = 0

    def on_arrive(self, msg):
        self.recorder.record_arrive_central(msg.msg_id, self.sim.now)
        self.queue.append(msg)
        if len(self.queue) > self.max_queue_len:
            self.max_queue_len = len(self.queue)
        if not self.busy:
            self._start_service()

    def _start_service(self):
        if not self.queue:
            self.busy = False
            return
        self.busy = True
        self._busy_start = self.sim.now
        msg = self.queue.popleft()
        if self.service_rate_msgs_s <= 0:
            service_time = 0.0
        else:
            service_time = 1.0 / self.service_rate_msgs_s
        if self.service_time_jitter > 0.0 and service_time > 0.0:
            jitter = self.rng.uniform(-self.service_time_jitter, self.service_time_jitter)
            service_time = max(0.0, service_time * (1.0 + jitter))
        self.sim.schedule(self.sim.now + service_time, EventType.CENTRAL_DONE, msg)

    def on_done(self, msg):
        self.recorder.record_central_done(msg.msg_id, self.sim.now)
        if self._busy_start is not None:
            self.busy_time += self.sim.now - self._busy_start
            self._busy_start = None
        self.busy = False
        if self.queue:
            self._start_service()

    def finalize(self, end_time):
        if self.busy and self._busy_start is not None:
            self.busy_time += end_time - self._busy_start
            self._busy_start = end_time
