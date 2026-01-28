from .base import Router


class AdaptiveRouter(Router):
    def __init__(self, topology):
        self._cache = dict(topology.robot_to_zone)

    def zone_for_robot(self, robot_id):
        return self._cache[robot_id]

    def on_policy_update(self, topology):
        self._cache = dict(topology.robot_to_zone)
