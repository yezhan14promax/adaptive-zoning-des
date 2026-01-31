from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict


@dataclass
class RNGStreams:
    arrival_rng: random.Random
    service_rng: random.Random
    policy_rng: random.Random


def make_rng_streams(seed: int) -> RNGStreams:
    arrival_rng = random.Random(seed + 101)
    service_rng = random.Random(seed + 503)
    policy_rng = random.Random(seed + 907)
    return RNGStreams(arrival_rng=arrival_rng, service_rng=service_rng, policy_rng=policy_rng)


def rng_seed_manifest(seed: int) -> Dict[str, int]:
    return {
        "arrival_seed": seed + 101,
        "service_seed": seed + 503,
        "policy_seed": seed + 907,
    }
