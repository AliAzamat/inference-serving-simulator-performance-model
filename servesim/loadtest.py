import sys

from . import config
from .metrics import summarize
from .sim import simulate
from .workload import make_workload


def sweep(rates, n=2000, seed=1, max_batch=config.MAX_BATCH,
          capacity=config.KV_CAPACITY_TOKENS, policy="fifo"):
    """Run one simulation per arrival rate and summarize each."""
    rows = []
    for rate in rates:
        run = simulate(make_workload(n, rate, seed), max_batch, capacity, policy)
        row = summarize(run)
        row["rate"] = rate
        row["peak_kv"] = run.peak_kv
        rows.append(row)
    return rows


def find_knee(rows, factor=3.0):
    """The first rate whose p99 is more than `factor` times the p99 at the lowest rate."""
    base = rows[0]["p99"]
    for r in rows:
        if r["p99"] > factor * base:
            return r["rate"]
    return None


def print_table(rows):
    print(f"{'rate':>6} {'p50 s':>8} {'p99 s':>8} {'ttft p99':>9} {'tok/s':>8} {'batch':>6} {'kv':>7}")
    for r in rows:
        print(f"{r['rate']:6.2f} {r['p50']:8.2f} {r['p99']:8.2f} {r['ttft_p99']:9.2f} "
              f"{r['tokens_per_s']:8.0f} {r['peak_batch']:6d} {r['peak_kv']:7d}")


if __name__ == "__main__":
    # usage: python3 -m servesim.loadtest <policy> <capacity> <rate> [<rate> ...]
    args = sys.argv[1:]
    policy = args.pop(0) if args else "fifo"
    capacity = int(args.pop(0)) if args else config.KV_CAPACITY_TOKENS
    rates = [float(x) for x in args] or [1, 4, 8, 10, 12, 13, 14, 16]
    rows = sweep(rates, capacity=capacity, policy=policy)
    print_table(rows)
    print("knee at rate:", find_knee(rows))
