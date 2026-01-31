def _apply_target_zone(topology, target_zone, hotspot_ratio, spatial_model=None):
    total = len(topology.robot_to_zone)
    desired = int(total * hotspot_ratio)
    current = sum(1 for z in topology.robot_to_zone.values() if z == target_zone)
    move_needed = max(0, desired - current)
    if move_needed == 0:
        return False

    candidates = [
        robot_id
        for robot_id in sorted(topology.robot_to_zone.keys())
        if topology.robot_to_zone[robot_id] != target_zone
    ]
    for robot_id in candidates[:move_needed]:
        topology.reassign(robot_id, target_zone)
        if spatial_model is not None:
            spatial_model.move_robot_to_zone(robot_id, target_zone)
    return move_needed > 0


def _apply_target_zones(topology, target_zones, extra_robots_per_zone, baseline, spatial_model=None):
    if not target_zones:
        return False
    baseline_counts = {}
    for zone_id in target_zones:
        baseline_counts[zone_id] = sum(1 for z in baseline.values() if z == zone_id)

    non_target = [
        robot_id
        for robot_id in sorted(topology.robot_to_zone.keys())
        if topology.robot_to_zone[robot_id] not in target_zones
    ]
    cursor = 0
    changed = False
    for zone_id in target_zones:
        desired = baseline_counts.get(zone_id, 0) + extra_robots_per_zone
        current = sum(1 for z in topology.robot_to_zone.values() if z == zone_id)
        move_needed = max(0, desired - current)
        for robot_id in non_target[cursor : cursor + move_needed]:
            topology.reassign(robot_id, zone_id)
            if spatial_model is not None:
                spatial_model.move_robot_to_zone(robot_id, zone_id)
            changed = True
        cursor += move_needed
    return changed


def _restore_baseline(topology, baseline, spatial_model=None):
    for robot_id, zone_id in baseline.items():
        topology.reassign(robot_id, zone_id)
        if spatial_model is not None:
            spatial_model.move_robot_to_zone(robot_id, zone_id)


def apply_hotspot(topology, now, params, spatial_model=None):
    state = params.setdefault(
        "state",
        {"applied": False, "reverted": False, "baseline": None},
    )
    if state["baseline"] is None:
        state["baseline"] = dict(topology.robot_to_zone)

    action = params.get("action")
    if action == "start":
        if state["applied"]:
            return False
        if "target_zones" in params and "extra_robots_per_zone" in params:
            changed = _apply_target_zones(
                topology,
                params["target_zones"],
                params["extra_robots_per_zone"],
                state["baseline"],
                spatial_model,
            )
        else:
            changed = _apply_target_zone(
                topology, params["target_zone"], params["hotspot_ratio"], spatial_model
            )
        state["applied"] = True
        return changed

    if action == "end":
        if state["reverted"]:
            return False
        _restore_baseline(topology, state["baseline"], spatial_model)
        state["reverted"] = True
        return True

    return False
