class Topology:
    def __init__(self, robot_to_zone, zones_count, neighbors=None):
        self.robot_to_zone = dict(robot_to_zone)
        self.zones_count = int(zones_count)
        self.neighbors = dict(neighbors) if neighbors is not None else {}

    def zone_for_robot(self, robot_id):
        return self.robot_to_zone[robot_id]

    def reassign(self, robot_id, new_zone_id):
        self.robot_to_zone[robot_id] = new_zone_id

    def neighbors_for(self, zone_id):
        return self.neighbors.get(zone_id)
