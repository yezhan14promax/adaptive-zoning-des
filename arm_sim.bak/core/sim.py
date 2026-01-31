from .events import Event, EventQueue


class Simulator:
    def __init__(self):
        self.now = 0.0
        self.event_queue = EventQueue()
        self.handlers = {}
        self._seq = 0

    def schedule(self, time, etype, payload):
        self._seq += 1
        event = Event(time=time, etype=etype, payload=payload, seq=self._seq)
        self.event_queue.push(event)

    def run(self, until_s):
        while len(self.event_queue) > 0:
            next_time = self.event_queue.peek_time()
            if next_time is None or next_time > until_s:
                break
            event = self.event_queue.pop()
            self.now = event.time
            handler = self.handlers.get(event.etype)
            if handler is not None:
                handler(event.payload)
        self.now = until_s
