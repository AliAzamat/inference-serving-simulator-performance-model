import sys

from . import config
from .metrics import summarize
from .model import mean_shape, predict_percentiles, predict
from .sim import simulate
from .workload import make_workload


def compare(rates, n=4000, seed=1):
    """Predict first, then run the simulator, and put the two side by side."""
    rows = []
    for rate in rates:
        reqs = make_workload(n, rate, seed)
        p, o = mean_shape(reqs)
        guess = predict(rate, p, o)
        pct = predict_percentiles(rate, reqs)
        got = summarize(simulate(reqs))
        row = {"rate": rate, "saturated": guess["saturated"], "got": got,
               "pred_tokens": guess["tokens_per_s"]}
        if pct:
            row["pred_p50"], row["pred_p99"] = pct
        rows.append(row)
    return rows


def calibrate(rates, n=4000, seed=1):
    """How much the model under-predicts p99: the median of measured / predicted."""
    ratios = sorted(r["got"]["p99"] / r["pred_p99"] for r in compare(rates, n, seed) if "pred_p99" in r)
    return ratios[len(ratios) // 2]


def rate_for_p99(target, factor, requests_for_shape, step=0.25):
    """Highest arrival rate whose calibrated predicted p99 stays under `target` seconds."""
    best, rate = 0.0, step
    while True:
        pct = predict_percentiles(rate, requests_for_shape)
        if pct is None or pct[1] * factor > target:
            return best
        best, rate = rate, rate + step


def p99_on_seeds(rate, seeds, n=4000):
    """Measured p99 at `rate` on each seed: one fresh workload per seed."""
    return [summarize(simulate(make_workload(n, rate, s)))["p99"] for s in seeds]


def print_rows(rows):
    print(f"{'rate':>5} | {'p50 pred':>8} {'p50 got':>8} | {'p99 pred':>8} {'p99 got':>8} | {'tok/s pred':>10} {'tok/s got':>9}")
    for r in rows:
        g = r["got"]
        if "pred_p50" in r:
            print(f"{r['rate']:5.1f} | {r['pred_p50']:8.2f} {g['p50']:8.2f} | {r['pred_p99']:8.2f} {g['p99']:8.2f} | "
                  f"{r['pred_tokens']:10.0f} {g['tokens_per_s']:9.0f}")
        else:
            print(f"{r['rate']:5.1f} | {'saturated':>8} {g['p50']:8.2f} | {'saturated':>8} {g['p99']:8.2f} | "
                  f"{r['pred_tokens']:10.0f} {g['tokens_per_s']:9.0f}")


if __name__ == "__main__":
    rates = [float(x) for x in sys.argv[1:]] or [1, 4, 8, 10, 12, 13, 14]
    print_rows(compare(rates))
    factor = calibrate([4, 8, 10])
    print(f"\np99 runs {factor:.2f}x above the model (median over rates 4, 8, 10)")
    shape = make_workload(4000, 5.0, seed=7)
    fresh = range(2, 22)  # twenty seeds the calibration never used
    for target in (4.0, 6.0, 8.0):
        rate = rate_for_p99(target, factor, shape)
        p99s = p99_on_seeds(rate, fresh)
        held = sum(p < target for p in p99s)
        print(f"p99 target {target:.0f}s: serve up to {rate:.2f} req/s, seed 7 alone {p99s[5]:.2f}s, "
              f"held on {held} of {len(p99s)} new seeds, worst {max(p99s):.2f}s")
