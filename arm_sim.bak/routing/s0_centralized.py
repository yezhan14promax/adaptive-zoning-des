from .base import Router


class CentralizedRouter(Router):
    def __init__(self, topology):
        self.topology = topology

    def zone_for_robot(self, robot_id):
        return self.topology.zone_for_robot(robot_id)
