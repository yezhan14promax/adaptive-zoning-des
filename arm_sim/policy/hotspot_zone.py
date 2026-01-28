from .base import Policy


class RobotTelemetry:
    def __init__(
        self,
        robot_ids,
        alpha_ema,
        heavy_fraction,
        cost_factor_light,
        cost_factor_heavy,
        cost_factor_mode,
        migrate_penalty_ms,
        migrate_penalty_ttl_s,
        spatial_model,
        rng,
    ):
        self.alpha_ema = float(alpha_ema)
        self.cost_factor_light = float(cost_factor_light)
        self.cost_factor_heavy = float(cost_factor_heavy)
        self.cost_factor_mode = cost_factor_mode
        self.migrate_penalty_s = float(migrate_penalty_ms) / 1000.0
        self.migrate_penalty_ttl_s = float(migrate_penalty_ttl_s)
        self.spatial = spatial_model

        self.cost_factor = {}
        self.ema_rate = {}
        self.count_sec = {}
        self.migrated_until_time = {}
        self._last_update_time = 0.0

        robot_ids = list(sorted(robot_ids))
        if self.cost_factor_mode == "lognormal":
            for robot_id in robot_ids:
                self.cost_factor[robot_id] = self.cost_factor_light * rng.lognormvariate(0.0, 0.5)
        else:
            heavy_count = int(len(robot_ids) * float(heavy_fraction))
            heavy_set = set(rng.sample(robot_ids, k=heavy_count)) if heavy_count > 0 else set()
            for robot_id in robot_ids:
                self.cost_factor[robot_id] = (
                    self.cost_factor_heavy if robot_id in heavy_set else self.cost_factor_light
                )

        for robot_id in robot_ids:
            self.ema_rate[robot_id] = 0.0
            self.count_sec[robot_id] = 0

    def record_emit(self, robot_id):
        if robot_id in self.count_sec:
            self.count_sec[robot_id] += 1

    def update_rates(self, now):
        dt = now - self._last_update_time
        if dt <= 0:
            return
        for robot_id, count in self.count_sec.items():
            current_rate = count / dt
            prev = self.ema_rate.get(robot_id, 0.0)
            self.ema_rate[robot_id] = self.alpha_ema * current_rate + (1 - self.alpha_ema) * prev
            self.count_sec[robot_id] = 0
        self._last_update_time = now

    def weight(self, robot_id):
        return self.ema_rate.get(robot_id, 0.0) * self.cost_factor.get(robot_id, 0.0)

    def penalty_delay_s(self, now, robot_id):
        until = self.migrated_until_time.get(robot_id, 0.0)
        if now < until:
            return self.migrate_penalty_s
        return 0.0

    def delay_ms(self, robot_id, zone_id, rng):
        if self.spatial is None:
            return 0.0
        return self.spatial.delay_ms(robot_id, zone_id, rng)

    def mark_migrated(self, robot_id, now):
        self.migrated_until_time[robot_id] = now + self.migrate_penalty_ttl_s

    def move_robot_to_zone(self, robot_id, zone_id):
        if self.spatial is not None:
            self.spatial.move_robot_to_zone(robot_id, zone_id)


class HotspotZonePolicy(Policy):
    def __init__(
        self,
        policy_period_s,
        q_high,
        q_low,
        move_k,
        cooldown_s,
        beta_capacity,
        budget_gamma,
        candidate_sample_m,
        p2c_k,
        dmax_ms,
        disable_fallback=False,
        disable_budget=False,
        fixed_k=False,
        move_k_fixed=5,
        total_arrival_rate=None,
        weight_w500=1.0,
        weight_w1000=2.0,
        weight_w1500=4.0,
        weight_scale=1.0,
        min_gain=200.0,
    ):
        self.policy_period_s = float(policy_period_s)
        self.q_high = int(q_high)
        self.q_low = int(q_low)
        self.move_k = int(move_k)
        self.cooldown_s = float(cooldown_s)
        self.beta_capacity = float(beta_capacity)
        self.budget_gamma = float(budget_gamma)
        self.candidate_sample_m = int(candidate_sample_m)
        self.p2c_k = int(p2c_k)
        self.dmax_ms = float(dmax_ms)
        self.disable_fallback = bool(disable_fallback)
        self.disable_budget = bool(disable_budget)
        self.fixed_k = bool(fixed_k)
        self.move_k_fixed = int(move_k_fixed)

        self._last_action_time = {}
        self._overloaded_flag = {}
        self._tokens = 0.0
        self._total_arrival_rate = float(total_arrival_rate) if total_arrival_rate else 0.0
        self.tokens_per_s = self.budget_gamma * self._total_arrival_rate
        self.weight_w500 = float(weight_w500)
        self.weight_w1000 = float(weight_w1000)
        self.weight_w1500 = float(weight_w1500)
        self.weight_scale = float(weight_scale)
        self.min_gain = float(min_gain)

        self.reassign_ops = 0
        self.reconfig_action_count = 0
        self.migrated_robots_total = 0
        self.migrated_weight_total = 0.0
        self.rejected_no_feasible_target = 0
        self.rejected_budget = 0
        self.rejected_safety = 0
        self.fallback_attempts = 0
        self.fallback_success = 0
        self.dmax_rejects = 0
        self.feasible_ratio_sum = 0.0
        self.feasible_ratio_count = 0

    def on_interval_start(self, now, zones, topology, router, telemetry, rng):
        if telemetry is None:
            return
        telemetry.update_rates(now)
        if self.disable_budget:
            self._tokens = float("inf")
        else:
            self._tokens = max(0.0, self.tokens_per_s * self.policy_period_s)

        for zone in zones:
            if zone.zone_id not in self._last_action_time:
                self._last_action_time[zone.zone_id] = -1.0e9
            if zone.zone_id not in self._overloaded_flag:
                self._overloaded_flag[zone.zone_id] = False

    def on_tick(self, now, zones, topology, router, telemetry, rng):
        if not zones or telemetry is None:
            return
        queue_lens = {zone.zone_id: zone.queue_len() for zone in zones}

        def _score(q_len):
            return self.weight_scale * (
                self.weight_w500 * max(0.0, q_len - 500.0)
                + self.weight_w1000 * max(0.0, q_len - 1000.0)
                + self.weight_w1500 * max(0.0, q_len - 1500.0)
            )

        def _benefit(donor_q, recv_q, weight):
            before = _score(donor_q) + _score(recv_q)
            after = _score(max(0.0, donor_q - weight)) + _score(recv_q + weight)
            return before - after

        gain_threshold = self.min_gain / max(self.weight_scale, 1.0e-6)

        for zone_id, q_len in queue_lens.items():
            flag = self._overloaded_flag.get(zone_id, False)
            if not flag and q_len >= self.q_high:
                flag = True
            elif flag and q_len <= self.q_low:
                flag = False
            self._overloaded_flag[zone_id] = flag

        donors = [
            zone_id
            for zone_id in queue_lens
            if self._overloaded_flag.get(zone_id, False)
            and now - self._last_action_time.get(zone_id, -1.0e9) >= self.cooldown_s
        ]
        receiver_threshold = max(self.q_low, self.q_high)
        receivers = [
            zone_id for zone_id, qlen in queue_lens.items() if qlen <= receiver_threshold
        ]

        if zones:
            self.feasible_ratio_sum += len(receivers) / float(len(zones))
            self.feasible_ratio_count += 1

        if not donors:
            return
        if not receivers:
            self.rejected_no_feasible_target += 1
            return

        available_budget = float("inf") if self.disable_budget else self._tokens
        if available_budget <= 0.0:
            self.rejected_budget += 1
            return

        receiver_state = {zone_id: queue_lens[zone_id] for zone_id in receivers}
        donor_state = {zone_id: queue_lens[zone_id] for zone_id in donors}
        donors_sorted = sorted(donors, key=lambda zid: (-_score(queue_lens[zid]), zid))

        def _robot_weight(robot_id):
            weight = telemetry.weight(robot_id)
            return weight if weight > 0.0 else 1.0

        candidates = []
        max_count = self.move_k_fixed if self.fixed_k else self.move_k
        for donor_id in donors_sorted:
            robot_weights = [
                (_robot_weight(robot_id), robot_id)
                for robot_id, zone_id in topology.robot_to_zone.items()
                if zone_id == donor_id
            ]
            if not robot_weights:
                continue
            robot_weights.sort(key=lambda pair: (-pair[0], pair[1]))
            for weight, robot_id in robot_weights[:max_count]:
                best = None
                best_ratio = None
                best_gain = None
                for receiver_id, recv_q in receiver_state.items():
                    if recv_q + weight > receiver_threshold:
                        continue
                    if self.dmax_ms < 1.0e8:
                        delay_ms = telemetry.delay_ms(robot_id, receiver_id, rng)
                        if delay_ms > self.dmax_ms:
                            self.dmax_rejects += 1
                            continue
                    gain = _benefit(donor_state[donor_id], recv_q, weight)
                    if gain < gain_threshold:
                        continue
                    ratio = gain / max(weight, 1.0e-6)
                    if best_ratio is None or ratio > best_ratio:
                        best_ratio = ratio
                        best_gain = gain
                        best = receiver_id
                if best is None:
                    continue
                candidates.append(
                    {
                        "ratio": best_ratio,
                        "gain": best_gain,
                        "weight": weight,
                        "robot_id": robot_id,
                        "donor_id": donor_id,
                        "receiver_id": best,
                    }
                )

        if not candidates:
            self.rejected_no_feasible_target += 1
            return

        candidates.sort(
            key=lambda item: (-item["ratio"], -item["gain"], item["robot_id"])
        )

        moved_any = False
        moved_weight_total = 0.0
        moved_robot_count = 0
        remaining_budget = available_budget
        moved_robots = set()

        for cand in candidates:
            if remaining_budget <= 0.0:
                break
            robot_id = cand["robot_id"]
            if robot_id in moved_robots:
                continue
            donor_id = cand["donor_id"]
            receiver_id = cand["receiver_id"]
            donor_q = donor_state.get(donor_id, 0.0)
            recv_q = receiver_state.get(receiver_id, 0.0)
            weight = cand["weight"]
            if recv_q + weight > receiver_threshold:
                continue
            gain = _benefit(donor_q, recv_q, weight)
            if gain < gain_threshold:
                continue
            if not self.disable_budget and weight > remaining_budget:
                continue
            topology.reassign(robot_id, receiver_id)
            telemetry.mark_migrated(robot_id, now)
            telemetry.move_robot_to_zone(robot_id, receiver_id)
            donor_state[donor_id] = max(0.0, donor_q - weight)
            receiver_state[receiver_id] = recv_q + weight
            remaining_budget -= weight
            moved_any = True
            moved_weight_total += weight
            moved_robot_count += 1
            moved_robots.add(robot_id)
            self._last_action_time[donor_id] = now

        if not moved_any:
            return

        router.on_policy_update(topology)
        self.reassign_ops += 1
        self.reconfig_action_count += 1
        self.migrated_robots_total += moved_robot_count
        self.migrated_weight_total += moved_weight_total
        if not self.disable_budget:
            self._tokens = max(0.0, self._tokens - moved_weight_total)
