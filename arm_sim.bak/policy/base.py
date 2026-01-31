class Policy:
    def on_interval_start(self, now, zones, topology, router, telemetry, rng):
        return None

    def on_tick(self, now, zones, topology, router, telemetry, rng):
        raise NotImplementedError
