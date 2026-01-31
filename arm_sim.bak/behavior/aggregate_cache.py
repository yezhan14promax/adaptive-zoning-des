from .base import ZoneBehavior


class AggregateCacheBehavior(ZoneBehavior):
    def on_arrival(self, zone, msg, now):
        zone.enqueue(msg)
        zone.start_service_if_idle()

    def on_done(self, zone, token, now):
        msg = token["msg"]
        zone.finish_service(msg, now)
        zone.forward_to_central(msg)
        zone.start_service_if_idle()
