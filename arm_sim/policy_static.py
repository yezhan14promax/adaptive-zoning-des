from __future__ import annotations


class StaticEdgePolicy:
    name = "Static-edge"

    def __init__(self, N: int):
        self.N = N

    def route(self, arrival, policy_rng, hotspot_zones):
        return arrival.home_zone, False
