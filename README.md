# LLM Serving Simulator and Performance Model

**A deterministic simulator of an LLM inference server (continuous batching, a KV-cache memory budget, and two admission policies) paired with a closed-form performance model that predicts latency and capacity *before* the simulator runs. Then it checks where the prediction holds and where it doesn't.**

Pure Python standard library: no GPU, no model weights, no dependencies. It has 41 unit tests. The costs come from a made-up cost model chosen for this project (see `servesim/config.py`), so the numbers describe that model, not any real hardware.

## Prediction vs. measurement

`python3 -m servesim.compare` predicts each row from the model, then runs the simulator on 4,000 requests:

```text
 rate | p50 pred  p50 got | p99 pred  p99 got | tok/s pred tok/s got
  1.0 |     1.27     1.29 |     2.53     2.61 |        105       103
  4.0 |     1.56     1.58 |     3.15     3.58 |        420       411
  8.0 |     2.31     2.26 |     4.71     6.08 |        839       820
 10.0 |     3.05     2.89 |     6.24     8.27 |       1049      1023
 12.0 |     4.50     4.06 |     9.28    10.70 |       1259      1226
 13.0 | saturated     5.53 | saturated    13.16 |       1277      1312
 14.0 | saturated     9.96 | saturated    20.75 |       1277      1365
```

Rate is requests per second. Latency is in seconds from arrival to the last token.

- **Where the model holds:** median latency is within 20% and throughput within 5% below saturation (both checked by tests).
- **Where it misses:** the p99 tail. The model predicts each request's latency from the *average* steady state, so it can't see queueing bursts, and it under-predicts p99 by up to about 1.3× near the knee.
- **Its ceiling is a little conservative.** The model says the server tops out at 12.17 req/s. The simulator keeps scaling a bit further, levelling off at about 1,370 tokens/s (roughly 13 req/s).

The model's predictions were written down before the comparison was run, in [`PREDICTIONS.md`](PREDICTIONS.md). The derivation is in the docstring of [`servesim/model.py`](servesim/model.py).

## What's simulated

- **Cost model:** prefill time grows with prompt tokens, and each decode step costs a fixed 12 ms plus a little per token held in the KV cache. Prefill stalls the whole batch.
- **Serial baseline:** one request at a time. It caps out around 80 tokens/s; at just 1 request/s, median latency is already 320 s and the queue grows without bound.
- **Continuous batching:** up to 64 requests decode together, and new requests join between decode steps.
- **KV-cache budget:** 60,000 tokens. A request reserves its prompt plus its full answer up front, so memory never goes over budget.
- **Admission policies:** `fifo` lets only the head of the line in, so a large request blocks everyone behind it. `skip` admits the first waiting request that fits. A test shows the head-of-line blocking directly. On the default workload the gap is small: with a 20,000-token budget at 6 req/s, p99 time to first token is 0.85 s for `fifo` vs. 0.72 s for `skip`.
- **Workload:** seeded Poisson arrivals. 70% are short chat turns (100–400 prompt tokens, 50–200 answer tokens) and 30% are long documents with short answers (1,500–3,000 prompt tokens, 20–100 answer tokens). The seed fixes request lengths independently of rate, so changing the rate changes only arrival times.

## Load test

`python3 -m servesim.loadtest`, using FIFO admission and the default 60k-token cache:

```text
  rate    p50 s    p99 s  ttft p99    tok/s  batch      kv
  1.00     1.31     2.60      0.16      104      8   13617
  4.00     1.60     3.51      0.23      414     24   29504
  8.00     2.28     5.87      0.33      824     53   55615
 10.00     2.90     7.88      0.51     1027     64   59975
 12.00     3.97     9.92      2.01     1228     64   59985
 13.00     5.01    10.82      2.87     1319     64   59988
 14.00     8.57    14.65      6.62     1369     64   59981
 16.00    17.69    30.50     24.03     1371     64   59993
knee at rate: 10
```

The knee is the first rate whose p99 is more than 3× the p99 at the lowest rate. At 10 req/s the batch hits its 64-request cap and the KV cache is full. Past that point throughput stops growing and the queue absorbs everything.

## Capacity planning on unseen seeds

The model's p99 is calibrated with the median measured/predicted ratio (1.29×) at rates 4, 8 and 10 on seed 1. The calibrated model then picks the highest rate expected to meet a p99 target, and that plan is checked on 20 seeds the calibration never saw:

```text
p99 target 4s: serve up to 3.75 req/s, seed 7 alone 3.38s, held on 20 of 20 new seeds, worst 3.51s
p99 target 6s: serve up to 8.00 req/s, seed 7 alone 5.20s, held on 18 of 20 new seeds, worst 6.06s
p99 target 8s: serve up to 10.25 req/s, seed 7 alone 6.90s, held on 16 of 20 new seeds, worst 9.25s
```

Far below the knee the plan holds on every seed. Near the knee, one fresh seed (seed 7) says the plan is safe while the twenty-seed check says it isn't. A test shows that backing off to 9.25 req/s holds the 8 s target on all 20 seeds.

## Quick start

Requires Python 3 and nothing else. Tested on Python 3.13.

```bash
git clone https://github.com/AliAzamat/inference-serving-simulator-performance-model.git
cd inference-serving-simulator-performance-model

python3 -m unittest                                  # 41 tests, about 15 s
python3 -m servesim.model                            # predictions only (matches PREDICTIONS.md)
python3 -m servesim.loadtest                         # sweep: fifo, 60k KV tokens
python3 -m servesim.loadtest skip 20000 4 6 8        # <policy> <kv capacity> <rates...>
python3 -m servesim.compare                          # prediction vs measurement + capacity plan
```

## Project layout

| Path | Purpose |
|---|---|
| `servesim/config.py` | Cost model and limits (prefill and decode costs, KV capacity, batch cap) |
| `servesim/workload.py` | Seeded Poisson workload generator |
| `servesim/sim.py` | Event loop: serial baseline, continuous batching, KV budget, `fifo`/`skip` admission |
| `servesim/metrics.py` | Nearest-rank percentiles, latency/TTFT/throughput summary |
| `servesim/loadtest.py` | Rate sweep and knee detection |
| `servesim/model.py` | Closed-form performance model and its derivation |
| `servesim/compare.py` | Prediction vs. simulation, p99 calibration, capacity planning on unseen seeds |
| `PREDICTIONS.md` | Model output recorded before the comparison |

## Limitations

- **The costs are invented.** Prefill and decode costs are round numbers chosen for the exercise, not measured from a GPU. The method carries over to real hardware, but the specific numbers don't.
- **Several real serving behaviours are left out:** paged or shared KV cache, chunked prefill, preemption, speculative decoding, and multiple replicas.
- **The tail correction is a single factor.** One measured/predicted ratio is applied at every rate, but the real gap widens near the knee, which is why the 6 s and 8 s plans don't hold on every seed.
