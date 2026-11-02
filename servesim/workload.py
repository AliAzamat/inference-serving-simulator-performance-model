import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    id: int
    arrival: float
    prompt: int
    output: int


def make_workload(n, rate, seed=1):
    """n requests with Poisson arrivals at `rate` per second.

    The lengths use the same random stream whatever the rate is, so two
    workloads with the same seed differ only in when requests arrive.
    """
    rng = random.Random(seed)
    t = 0.0
    requests = []
    for i in range(n):
        t += rng.expovariate(rate)
        if rng.random() < 0.7:  # a short chat turn
            prompt, output = rng.randint(100, 400), rng.randint(50, 200)
        else:  # a long document with a short answer
            prompt, output = rng.randint(1500, 3000), rng.randint(20, 100)
        requests.append(Request(i, t, prompt, output))
    return requests


def describe(requests):
    n = len(requests)
    return {
        "n": n,
        "mean_prompt": sum(r.prompt for r in requests) / n,
        "mean_output": sum(r.output for r in requests) / n,
        "max_total": max(r.prompt + r.output for r in requests),
    }
