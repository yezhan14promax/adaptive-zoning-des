from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class S2Profile:
    profile_id: str
    budget_ratio: float
    weight_scale: float
    cooldown_s: int


class PolicyS2:
    name = "S2"

    def __init__(self, N: int, profile: S2Profile, cloud_zone: int):
        self.N = N
        self.profile = profile
        self.cloud_zone = cloud_zone

    def _offload_prob(self, home_zone: int, hotspot_zones: List[int]) -> float:
        base = max(0.0, min(1.0, self.profile.budget_ratio))
        scale = max(0.1, self.profile.weight_scale)
        cooldown_penalty = 1.0 + (self.profile.cooldown_s / 10.0)
        if home_zone in hotspot_zones:
            p = base * scale / cooldown_penalty
        else:
            p = base * (scale * 0.25) / cooldown_penalty
        return max(0.0, min(0.95, p))

    def route(self, arrival, policy_rng, hotspot_zones):
        p_offload = self._offload_prob(arrival.home_zone, hotspot_zones)
        if policy_rng.random() < p_offload:
            return self.cloud_zone, True
        return arrival.home_zone, False
