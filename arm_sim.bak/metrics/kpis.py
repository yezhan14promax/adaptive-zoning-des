import math


def _completed_latencies_ms(recorder):
    latencies = []
    for record in recorder.records.values():
        emit_time = record.get("emit_time")
        done_time = record.get("central_done")
        if emit_time is None or done_time is None:
            continue
        latencies.append((done_time - emit_time) * 1000.0)
    return latencies


def mean_p95_ms(recorder):
    latencies = _completed_latencies_ms(recorder)
    if not latencies:
        return 0.0, 0.0, 0
    latencies.sort()
    n = len(latencies)
    mean_val = sum(latencies) / n
    index = int(math.ceil(0.95 * n) - 1)
    p95_val = latencies[index]
    return mean_val, p95_val, n
