from dataclasses import dataclass


@dataclass
class StateMsg:
    msg_id: int
    robot_id: int
    emit_time: float
    size_bytes: int = 0
