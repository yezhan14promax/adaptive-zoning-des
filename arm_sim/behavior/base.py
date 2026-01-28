class ZoneBehavior:
    def on_arrival(self, zone, msg, now):
        raise NotImplementedError

    def on_done(self, zone, token, now):
        raise NotImplementedError
