from .message import StateMsg
from .network import sample_delay_s
from ..core.events import EventType


class Robot:
    def __init__(
        self,
        robot_id,
        state_rate_hz,
        sim,
        router,
        recorder,
        base_ms,
        jitter_ms,
        rng,
        telemetry=None,
        delay_model=None,
        emit_jitter_s=0.0,
        poisson_arrivals=False,
        arrival_rng=None,
        delay_rng=None,
    ):
        self.robot_id = int(robot_id)
        self.state_rate_hz = float(state_rate_hz)
        self.sim = sim
        self.router = router
        self.recorder = recorder
        self.base_ms = float(base_ms)
        self.jitter_ms = float(jitter_ms)
        self.rng = rng
        self.arrival_rng = arrival_rng or rng
        self.delay_rng = delay_rng or rng
        self.telemetry = telemetry
        self.delay_model = delay_model
        self.emit_jitter_s = float(emit_jitter_s)
        self.poisson_arrivals = bool(poisson_arrivals)
        self._seq = 0

    def schedule_first(self, start_time):
        self.sim.schedule(start_time, EventType.ROBOT_EMIT, self.robot_id)

    def on_emit(self):
        self._seq += 1
        msg_id = self.robot_id * 1000000 + self._seq
        msg = StateMsg(msg_id=msg_id, robot_id=self.robot_id, emit_time=self.sim.now, size_bytes=0)
        zone_id = self.router.zone_for_robot(self.robot_id)
        self.recorder.record_emit(
            msg.msg_id, self.sim.now, self.robot_id, emit_zone_id=zone_id, home_zone_id=zone_id
        )
        if self.telemetry is not None:
            self.telemetry.record_emit(self.robot_id)
        if self.delay_model is not None:
            delay_s = self.delay_model.delay_s(self.robot_id, zone_id, self.delay_rng)
        else:
            delay_s = sample_delay_s(self.base_ms, self.jitter_ms, self.delay_rng)
        if self.telemetry is not None:
            delay_s += self.telemetry.penalty_delay_s(self.sim.now, self.robot_id)
        payload = {"msg": msg, "zone_id": zone_id}
        self.sim.schedule(self.sim.now + delay_s, EventType.ARRIVE_ZONE, payload)

        if self.state_rate_hz > 0:
            if self.poisson_arrivals:
                interval = self.arrival_rng.expovariate(self.state_rate_hz)
            else:
                interval = 1.0 / self.state_rate_hz
                jitter = 0.0
                if self.emit_jitter_s > 0.0:
                    jitter = self.arrival_rng.uniform(-self.emit_jitter_s, self.emit_jitter_s)
                interval = max(0.0, interval + jitter)
            next_time = self.sim.now + interval
            self.sim.schedule(next_time, EventType.ROBOT_EMIT, self.robot_id)
