from .message import StateMsg
from ..core.events import EventType


class ZoneArrivalProcess:
    def __init__(
        self,
        zone_id,
        rate_hz,
        sim,
        router,
        recorder,
        rng,
        home_robot_ids,
        telemetry=None,
        delay_model=None,
        arrival_rng=None,
        delay_rng=None,
    ):
        self.zone_id = int(zone_id)
        self.rate_hz = float(rate_hz)
        self.sim = sim
        self.router = router
        self.recorder = recorder
        self.rng = rng
        self.arrival_rng = arrival_rng or rng
        self.delay_rng = delay_rng or rng
        self.home_robot_ids = list(home_robot_ids or [])
        self.telemetry = telemetry
        self.delay_model = delay_model
        self._seq = 0

    def schedule_first(self, start_time):
        self.sim.schedule(start_time, EventType.ZONE_EMIT, {"zone_id": self.zone_id})

    def on_emit(self, now):
        if self.rate_hz <= 0.0:
            return
        self._seq += 1
        if self.home_robot_ids:
            robot_id = self.arrival_rng.choice(self.home_robot_ids)
        else:
            robot_id = self.zone_id * 1000000 + self._seq
        msg_id = self.zone_id * 1000000 + self._seq
        msg = StateMsg(msg_id=msg_id, robot_id=robot_id, emit_time=now, size_bytes=0)
        home_zone_id = self.zone_id
        self.recorder.record_emit(
            msg.msg_id, now, robot_id, emit_zone_id=home_zone_id, home_zone_id=home_zone_id
        )
        if self.telemetry is not None:
            self.telemetry.record_emit(robot_id)

        route_zone_id = (
            self.router.zone_for_robot(robot_id)
            if self.router is not None
            else home_zone_id
        )
        delay_s = 0.0
        if self.delay_model is not None:
            delay_s = self.delay_model.delay_s(robot_id, route_zone_id, self.delay_rng)
        if self.telemetry is not None:
            delay_s += self.telemetry.penalty_delay_s(now, robot_id)
        payload = {"msg": msg, "zone_id": route_zone_id}
        self.sim.schedule(now + delay_s, EventType.ARRIVE_ZONE, payload)

        interval = self.arrival_rng.expovariate(self.rate_hz)
        self.sim.schedule(now + interval, EventType.ZONE_EMIT, {"zone_id": self.zone_id})
