class Router:
    def zone_for_robot(self, robot_id):
        raise NotImplementedError

    def on_policy_update(self, topology):
        return None
