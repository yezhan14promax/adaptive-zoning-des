from collections import deque

from .network import sample_delay_s
from ..core.events import EventType


class ZoneController:
    def __init__(
        self,
        zone_id,
        sim,
        recorder,
        base_ms,
        jitter_ms,
        rng,
        central,
        zone_service_rate_msgs_s,
        behavior,
        service_time_jitter=0.0,
    ):
        self.zone_id = int(zone_id)
        self.sim = sim
        self.recorder = recorder
        self.base_ms = float(base_ms)
        self.jitter_ms = float(jitter_ms)
        self.rng = rng
        self.central = central
        self.zone_service_rate_msgs_s = float(zone_service_rate_msgs_s)
        self.service_time_jitter = float(service_time_jitter)
        self.behavior = behavior
        self.ingress = deque()
        self.busy = False
        self.busy_time = 0.0
        self._busy_start = None
        self.max_queue_len = 0

    def on_arrive(self, msg):
        self.recorder.record_arrive_zone(msg.msg_id, self.sim.now)
        self.recorder.record_admit_zone(msg.msg_id, self.zone_id)
        self.behavior.on_arrival(self, msg, self.sim.now)

    def on_done(self, token):
        self.behavior.on_done(self, token, self.sim.now)

    def enqueue(self, msg):
        self.ingress.append(msg)
        if len(self.ingress) > self.max_queue_len:
            self.max_queue_len = len(self.ingress)

    def queue_len(self):
        return len(self.ingress) + (1 if self.busy else 0)

    def start_service_if_idle(self):
        if self.busy or not self.ingress:
            return None
        self.busy = True
        self._busy_start = self.sim.now
        msg = self.ingress.popleft()
        self.recorder.record_zone_start(msg.msg_id, self.sim.now)
        if self.zone_service_rate_msgs_s <= 0:
            service_time = 0.0
        else:
            service_time = 1.0 / self.zone_service_rate_msgs_s
        if self.service_time_jitter > 0.0 and service_time > 0.0:
            jitter = self.rng.uniform(-self.service_time_jitter, self.service_time_jitter)
            service_time = max(0.0, service_time * (1.0 + jitter))
        token = {"zone_id": self.zone_id, "msg": msg}
        self.sim.schedule(self.sim.now + service_time, EventType.ZONE_DONE, token)
        return token

    def finish_service(self, msg, now):
        self.recorder.record_zone_done(msg.msg_id, now, self.zone_id)
        if self._busy_start is not None:
            self.busy_time += now - self._busy_start
        self.busy = False
        self._busy_start = None

    def finalize(self, end_time):
        if self.busy and self._busy_start is not None:
            self.busy_time += end_time - self._busy_start
            self._busy_start = end_time

    def forward_to_central(self, msg):
        delay_s = sample_delay_s(self.base_ms, self.jitter_ms, self.rng)
        self.sim.schedule(self.sim.now + delay_s, EventType.ARRIVE_CENTRAL, msg)
