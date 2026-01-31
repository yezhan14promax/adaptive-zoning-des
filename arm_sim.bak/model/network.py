def sample_delay_s(base_ms, jitter_ms, rng):
    jitter = rng.uniform(-jitter_ms, jitter_ms)
    total_ms = base_ms + jitter
    if total_ms < 0:
        total_ms = 0.0
    return total_ms / 1000.0
