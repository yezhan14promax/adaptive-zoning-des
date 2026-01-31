from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple


@dataclass
class HotspotConfig:
    phi: float = 0.25
    hotspot_skew: float = 6.0
    hotspot_window: Tuple[int, int] = (60, 140)
    hotspot_selection_mode: str = "seeded_random"


@dataclass
class ServiceConfig:
    mu_zone: float = 20.0
    mu_cloud: float = 120.0
    edge_cost: float = 1.0
    cloud_cost: float = 3.0


@dataclass
class ArrivalConfig:
    arrival_mode: str = "zone_poisson"
    lambda_mean: float = 15.0
    sim_duration_s: int = 200


@dataclass
class WindowingConfig:
    window_s: int = 30
    queue_overload_threshold: int = 1000
    queue_overload_thresholds: List[int] = field(default_factory=lambda: [500, 1000, 1500])


@dataclass
class ProfileGridConfig:
    budget_ratios: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.4])
    weight_scales: List[float] = field(default_factory=lambda: [0.8, 1.2])
    cooldown_s: List[int] = field(default_factory=lambda: [0, 5])


@dataclass
class SelectionConfig:
    balanced_threshold: float = 0.3
    strong_threshold: float = 0.1


@dataclass
class ExperimentConfig:
    N: int = 16
    scale_Ns: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    emit_debug_artifacts: bool = False
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)
    hotspot: HotspotConfig = field(default_factory=HotspotConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    profile_grid: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DerivedHotspotParams:
    N: int
    k_hot: int
    phi_actual: float
    lambda_hot: float
    lambda_cold: float
    hotspot_zones: List[int]


@dataclass
class RunParams:
    base_config: Dict
    derived_hotspot: Dict[int, DerivedHotspotParams]
    rng_streams: Dict[str, int]
    cohort_time_rule: str = "arrival_time"
    completion_ratio_clamped: bool = False
    module_imports_ok: bool = True
    rng_streams_independent: bool = True


def expand_config(overrides: Dict | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if key == "seeds" and isinstance(value, str):
            cfg.seeds = [int(s) for s in value.split(",") if s.strip() != ""]
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def compute_hotspot_params(N: int, phi: float, hotspot_skew: float, lambda_mean: float, hotspot_zone_selector) -> DerivedHotspotParams:
    k_hot = max(1, int(round(phi * N)))
    phi_actual = k_hot / float(N)
    denom = phi_actual * hotspot_skew + (1.0 - phi_actual)
    lambda_cold = lambda_mean / denom
    lambda_hot = hotspot_skew * lambda_cold
    hotspot_zones = hotspot_zone_selector(N, k_hot)
    return DerivedHotspotParams(
        N=N,
        k_hot=k_hot,
        phi_actual=phi_actual,
        lambda_hot=lambda_hot,
        lambda_cold=lambda_cold,
        hotspot_zones=hotspot_zones,
    )
