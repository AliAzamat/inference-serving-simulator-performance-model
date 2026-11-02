from collections import deque
from dataclasses import dataclass, field

from . import config


@dataclass
class Result:
    request: object
    admitted: float     # when its prefill started
    first_token: float  # when the first token came out
    done: float         # when the last token came out

    @property
    def latency(self):
        return self.done - self.request.arrival

    @property
    def ttft(self):
        return self.first_token - self.request.arrival


@dataclass
class Run:
    results: list = field(default_factory=list)
    peak_batch: int = 0
    peak_kv: int = 0    # most reserved KV tokens at any moment


@dataclass
class Active:
    request: object
    admitted: float
    first_token: float
    kv: int        # tokens this request holds in the cache right now
    left: int      # decode steps still to run
    reserved: int  # tokens set aside for it: prompt plus the whole answer


def _retire(running, t, run):
    """Move every request with no steps left out of the batch."""
    still = []
    for a in running:
        if a.left == 0:
            run.results.append(Result(a.request, a.admitted, a.first_token, t))
        else:
            still.append(a)
    running[:] = still


def pick(waiting, running, max_batch, capacity, policy):
    """Index in `waiting` of the next request to admit, or None.

    fifo: only the head of the line may go, so a big request blocks everyone behind it.
    skip: take the first request in line that fits in the memory that is left.
    """
    if len(running) >= max_batch:
        return None
    used = sum(a.reserved for a in running)
    for i, r in enumerate(waiting):
        if used + r.prompt + r.output <= capacity:
            return i
        if policy == "fifo":
            return None
    return None


def simulate(requests, max_batch=config.MAX_BATCH, capacity=config.KV_CAPACITY_TOKENS, policy="fifo"):
    """Continuous batching with a KV memory budget and an admission policy."""
    if policy not in ("fifo", "skip"):
        raise ValueError(f"unknown policy {policy!r}")
    too_big = [r for r in requests if r.prompt + r.output > capacity]
    if too_big:
        raise ValueError(f"request {too_big[0].id} can never fit in {capacity} KV tokens")
    pending = deque(sorted(requests, key=lambda r: r.arrival))
    waiting = []
    running = []
    run = Run()
    t = 0.0
    while pending or waiting or running:
        if not running and not waiting and pending[0].arrival > t:
            t = pending[0].arrival  # idle: jump to the next arrival
        while pending and pending[0].arrival <= t:
            waiting.append(pending.popleft())
        while True:
            i = pick(waiting, running, max_batch, capacity, policy)
            if i is None:
                break
            r = waiting.pop(i)
            admitted = t
            t += config.prefill_time(r.prompt)  # prefill stalls the whole batch
            running.append(Active(r, admitted, t, r.prompt, r.output - 1, r.prompt + r.output))
            while pending and pending[0].arrival <= t:
                waiting.append(pending.popleft())
        run.peak_batch = max(run.peak_batch, len(running))
        run.peak_kv = max(run.peak_kv, sum(a.reserved for a in running))
        _retire(running, t, run)
        if running:
            t += config.decode_step_time(sum(a.kv for a in running))
            for a in running:
                a.kv += 1
                a.left -= 1
            _retire(running, t, run)
    return run


def simulate_serial(requests):
    """The baseline: one request at a time, in arrival order."""
    t = 0.0
    run = Run(peak_batch=1)
    for r in sorted(requests, key=lambda r: r.arrival):
        t = max(t, r.arrival)
        admitted = t
        t += config.prefill_time(r.prompt)
        first = t
        kv = r.prompt
        for _ in range(r.output - 1):  # the first token came from prefill
            t += config.decode_step_time(kv)
            kv += 1
        run.results.append(Result(r, admitted, first, t))
    return run
