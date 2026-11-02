"""A performance model on paper: predict the numbers before running anything.

Steady state, continuous batching. Per request: P prompt tokens, O answer tokens.
Prefill stalls the whole batch, so a share of the clock is lost to it:
    f = 1 - rate * P * prefill_s_per_token          (time left for decode steps)
B requests decode together and each step takes a + b * B * L, where L is the
average cached tokens per request. Decode must produce rate * (O - 1) tokens
per second, and the batch makes B / step tokens per second of decode time:
    rate * (O - 1) = f * B / (a + b * B * L)
Solving for B:
    B = rate * (O - 1) * a / (f - rate * (O - 1) * b * L)
The batch can never be bigger than the batch cap or the KV memory allows:
    B_max = min(max_batch, capacity / (P + O))
At B_max the same equation, solved for the rate, gives the most the server can finish:
    s = B_max / (a + b * B_max * L)
    max_rate = s / ((O - 1) + P * prefill_s_per_token * s)
"""
from . import config
from .metrics import percentile


def mean_shape(requests):
    p = sum(r.prompt for r in requests) / len(requests)
    o = sum(r.output for r in requests) / len(requests)
    return p, o


def batch_limit(p, o, max_batch=config.MAX_BATCH, capacity=config.KV_CAPACITY_TOKENS):
    return min(max_batch, capacity / (p + o))


def max_rate(p, o, max_batch=config.MAX_BATCH, capacity=config.KV_CAPACITY_TOKENS):
    """Most requests per second the server can finish, given its batch and memory limits."""
    a, b = config.DECODE_BASE_S, config.DECODE_S_PER_KV_TOKEN
    l_avg = p + (o - 1) / 2
    bm = batch_limit(p, o, max_batch, capacity)
    s = bm / (a + b * bm * l_avg)
    return s / ((o - 1) + p * config.PREFILL_S_PER_TOKEN * s)


def predict(rate, p, o, max_batch=config.MAX_BATCH, capacity=config.KV_CAPACITY_TOKENS):
    """Predicted batch size, step time, latency and tokens per second at `rate`."""
    a, b = config.DECODE_BASE_S, config.DECODE_S_PER_KV_TOKEN
    l_avg = p + (o - 1) / 2
    top = max_rate(p, o, max_batch, capacity)
    if rate >= top:
        return {"saturated": True, "max_rate": top, "tokens_per_s": top * o}
    f = 1 - rate * p * config.PREFILL_S_PER_TOKEN
    batch = rate * (o - 1) * a / (f - rate * (o - 1) * b * l_avg)
    step = a + b * batch * l_avg
    latency = p * config.PREFILL_S_PER_TOKEN + (o - 1) * step / f
    return {"saturated": False, "max_rate": top, "batch": batch, "step": step,
            "latency": latency, "tokens_per_s": rate * o}


def predict_percentiles(rate, requests, max_batch=config.MAX_BATCH, capacity=config.KV_CAPACITY_TOKENS):
    """Predicted p50 and p99 latency, or None past saturation.

    The steady state (step time and the share of the clock left for decode) is
    worked out once from the average request. Each request's own lengths then
    give its own latency, and the percentiles are read off that list.
    """
    p, o = mean_shape(requests)
    m = predict(rate, p, o, max_batch, capacity)
    if m["saturated"]:
        return None
    f = 1 - rate * p * config.PREFILL_S_PER_TOKEN
    lat = [r.prompt * config.PREFILL_S_PER_TOKEN + (r.output - 1) * m["step"] / f for r in requests]
    return percentile(lat, 50), percentile(lat, 99)


if __name__ == "__main__":
    import sys

    from .workload import make_workload

    p, o = mean_shape(make_workload(4000, 5.0))
    print(f"mean prompt {p:.0f} tokens, mean answer {o:.0f} tokens")
    print(f"most requests per second the server can finish: {max_rate(p, o):.2f}")
    print(f"{'rate':>5} {'batch':>6} {'step ms':>8} {'latency s':>10} {'tok/s':>7}")
    for rate in [float(x) for x in sys.argv[1:]] or [1, 4, 8, 10, 12]:
        m = predict(rate, p, o)
        if m["saturated"]:
            print(f"{rate:5.1f}   saturated, cannot keep up")
        else:
            print(f"{rate:5.1f} {m['batch']:6.1f} {m['step'] * 1000:8.1f} {m['latency']:10.2f} {m['tokens_per_s']:7.0f}")
