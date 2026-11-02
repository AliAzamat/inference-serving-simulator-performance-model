import math


def percentile(values, p):
    """Nearest-rank percentile: the smallest value with p percent at or below it."""
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(run):
    results = run.results
    latencies = [r.latency for r in results]
    ttfts = [r.ttft for r in results]
    tokens = sum(r.request.output for r in results)
    span = max(r.done for r in results) - min(r.request.arrival for r in results)
    return {
        "n": len(results),
        "p50": percentile(latencies, 50),
        "p99": percentile(latencies, 99),
        "mean": sum(latencies) / len(latencies),
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p99": percentile(ttfts, 99),
        "tokens_per_s": tokens / span,
        "peak_batch": run.peak_batch,
    }
