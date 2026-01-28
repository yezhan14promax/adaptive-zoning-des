import random

from arm_sim.behavior.base import ZoneBehavior
from arm_sim.core.events import EventType
from arm_sim.core.sim import Simulator
from arm_sim.metrics.kpis import mean_p95_ms
from arm_sim.metrics.recorder import Recorder
from arm_sim.model.central import CentralSupervisor
from arm_sim.model.robot import Robot
from arm_sim.model.topology import Topology
from arm_sim.model.zone import ZoneController
from arm_sim.routing.s0_centralized import CentralizedRouter


class ForwardBehavior(ZoneBehavior):
    def on_arrival(self, zone, msg, now):
        zone.recorder.record_zone_done(msg.msg_id, now, zone.zone_id)
        zone.forward_to_central(msg)

    def on_done(self, zone, token, now):
        return None


def main():
    n_robots = 10
    n_zones = 2
    duration_s = 2.0
    state_rate_hz = 50
    robot_to_zone_base_ms = 5
    robot_to_zone_jitter_ms = 2
    zone_to_central_base_ms = 5
    zone_to_central_jitter_ms = 2
    zone_service_rate_msgs_s = 200
    central_service_rate_msgs_s = 2 * zone_service_rate_msgs_s

    rng = random.Random(123)
    sim = Simulator()
    recorder = Recorder()

    mapping = {robot_id: robot_id % n_zones for robot_id in range(n_robots)}
    topology = Topology(mapping, n_zones)
    router = CentralizedRouter(topology)

    central = CentralSupervisor(sim, recorder, central_service_rate_msgs_s)
    zones = [
        ZoneController(
            zone_id=i,
            sim=sim,
            recorder=recorder,
            base_ms=zone_to_central_base_ms,
            jitter_ms=zone_to_central_jitter_ms,
            rng=rng,
            central=central,
            zone_service_rate_msgs_s=zone_service_rate_msgs_s,
            behavior=ForwardBehavior(),
        )
        for i in range(n_zones)
    ]
    robots = [
        Robot(
            robot_id=i,
            state_rate_hz=state_rate_hz,
            sim=sim,
            router=router,
            recorder=recorder,
            base_ms=robot_to_zone_base_ms,
            jitter_ms=robot_to_zone_jitter_ms,
            rng=rng,
            telemetry=None,
        )
        for i in range(n_robots)
    ]

    def handle_robot_emit(robot_id):
        robots[robot_id].on_emit()

    def handle_arrive_zone(payload):
        msg = payload["msg"]
        zone_id = payload["zone_id"]
        zones[zone_id].on_arrive(msg)

    def handle_zone_done(token):
        zone_id = token["zone_id"]
        zones[zone_id].on_done(token)

    def handle_arrive_central(msg):
        central.on_arrive(msg)

    def handle_central_done(msg):
        central.on_done(msg)

    sim.handlers[EventType.ROBOT_EMIT] = handle_robot_emit
    sim.handlers[EventType.ARRIVE_ZONE] = handle_arrive_zone
    sim.handlers[EventType.ZONE_DONE] = handle_zone_done
    sim.handlers[EventType.ARRIVE_CENTRAL] = handle_arrive_central
    sim.handlers[EventType.CENTRAL_DONE] = handle_central_done

    print(f"n_robots={n_robots} n_zones={n_zones} state_rate_hz={state_rate_hz}")
    print(
        "robot_to_zone_ms="
        f"{robot_to_zone_base_ms}+/-{robot_to_zone_jitter_ms} "
        "zone_to_central_ms="
        f"{zone_to_central_base_ms}+/-{zone_to_central_jitter_ms}"
    )
    print(
        "zone_service_rate_msgs_s="
        f"{zone_service_rate_msgs_s} central_service_rate_msgs_s={central_service_rate_msgs_s}"
    )
    print("latency_def=emit_to_central_done")

    for robot in robots:
        robot.schedule_first(0.0)

    sim.run(duration_s)

    central.finalize(duration_s)
    for zone in zones:
        zone.finalize(duration_s)

    mean_ms, p95_ms, completed = mean_p95_ms(recorder)
    max_zone_q = [zone.max_queue_len for zone in zones]
    max_zone_q_overall = max(max_zone_q) if max_zone_q else 0
    max_central_q = central.max_queue_len
    if duration_s > 0:
        central_busy_fraction = central.busy_time / duration_s
        zone_busy_fraction = [zone.busy_time / duration_s for zone in zones]
    else:
        central_busy_fraction = 0.0
        zone_busy_fraction = [0.0 for _ in zones]
    print(
        f"completed={completed} mean_ms={mean_ms:.3f} p95_ms={p95_ms:.3f} "
        f"max_zone_q={max_zone_q} max_zone_q_max={max_zone_q_overall} "
        f"max_central_q={max_central_q} central_busy_fraction={central_busy_fraction:.3f} "
        f"zone_busy_fraction={zone_busy_fraction}"
    )


if __name__ == "__main__":
    main()
