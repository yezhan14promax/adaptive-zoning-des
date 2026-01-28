class Recorder:
    def __init__(self):
        self.records = {}

    def _ensure(self, msg_id):
        if msg_id not in self.records:
            self.records[msg_id] = {
                "emit_time": None,
                "emit_robot_id": None,
                "emit_zone_id": None,
                "home_zone_id": None,
                "admit_zone_id": None,
                "arrive_zone": None,
                "zone_start": None,
                "zone_done": None,
                "edge_done_time": None,
                "edge_done_zone_id": None,
                "arrive_central": None,
                "central_done": None,
                "central_done_time": None,
                "final_done_time": None,
                "final_done_stage": None,
                "final_done_zone_id": None,
            }
        return self.records[msg_id]

    def record_emit(self, msg_id, time_s, robot_id=None, emit_zone_id=None, home_zone_id=None):
        record = self._ensure(msg_id)
        record["emit_time"] = time_s
        if robot_id is not None:
            record["emit_robot_id"] = int(robot_id)
        if emit_zone_id is not None:
            record["emit_zone_id"] = int(emit_zone_id)
        if home_zone_id is not None:
            record["home_zone_id"] = int(home_zone_id)
        elif emit_zone_id is not None:
            record["home_zone_id"] = int(emit_zone_id)

    def record_arrive_zone(self, msg_id, time_s):
        self._ensure(msg_id)["arrive_zone"] = time_s

    def record_admit_zone(self, msg_id, zone_id):
        self._ensure(msg_id)["admit_zone_id"] = int(zone_id)

    def record_zone_start(self, msg_id, time_s):
        self._ensure(msg_id)["zone_start"] = time_s

    def record_zone_done(self, msg_id, time_s, zone_id=None):
        record = self._ensure(msg_id)
        record["zone_done"] = time_s
        record["edge_done_time"] = time_s
        if zone_id is not None:
            record["edge_done_zone_id"] = int(zone_id)
        record["final_done_time"] = time_s
        record["final_done_stage"] = "edge"
        record["final_done_zone_id"] = record.get("edge_done_zone_id")

    def record_arrive_central(self, msg_id, time_s):
        self._ensure(msg_id)["arrive_central"] = time_s

    def record_central_done(self, msg_id, time_s):
        record = self._ensure(msg_id)
        record["central_done"] = time_s
        record["central_done_time"] = time_s
        if record.get("final_done_time") is None:
            record["final_done_time"] = time_s
            record["final_done_stage"] = "central"
            record["final_done_zone_id"] = None
