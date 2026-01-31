import math


class SpatialModel:
    def __init__(
        self,
        robot_to_zone,
        zones_count,
        rng,
        zone_scale=10.0,
        robot_sigma=0.5,
        base_ms=3.0,
        k_ms_per_unit=1.0,
        jitter_ms=1.0,
    ):
        self.rng = rng
        self.zone_scale = float(zone_scale)
        self.robot_sigma = float(robot_sigma)
        self.base_ms = float(base_ms)
        self.k_ms_per_unit = float(k_ms_per_unit)
        self.jitter_ms = float(jitter_ms)

        self.zone_pos = {}
        for zone_id in range(int(zones_count)):
            self.zone_pos[zone_id] = (
                self.rng.uniform(0.0, self.zone_scale),
                self.rng.uniform(0.0, self.zone_scale),
            )

        self.robot_pos = {}
        for robot_id, zone_id in robot_to_zone.items():
            self.robot_pos[robot_id] = self._near_zone(zone_id)

    def _near_zone(self, zone_id):
        zx, zy = self.zone_pos[zone_id]
        return (
            zx + self.rng.gauss(0.0, self.robot_sigma),
            zy + self.rng.gauss(0.0, self.robot_sigma),
        )

    def move_robot_to_zone(self, robot_id, zone_id):
        self.robot_pos[robot_id] = self._near_zone(zone_id)

    def distance(self, robot_id, zone_id):
        rx, ry = self.robot_pos[robot_id]
        zx, zy = self.zone_pos[zone_id]
        return math.hypot(rx - zx, ry - zy)

    def delay_ms(self, robot_id, zone_id, rng):
        dist = self.distance(robot_id, zone_id)
        jitter = rng.uniform(-self.jitter_ms, self.jitter_ms)
        total_ms = self.base_ms + self.k_ms_per_unit * dist + jitter
        if total_ms < 0.0:
            total_ms = 0.0
        return total_ms

    def delay_s(self, robot_id, zone_id, rng):
        return self.delay_ms(robot_id, zone_id, rng) / 1000.0
