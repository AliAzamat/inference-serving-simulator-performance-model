import unittest

from servesim import config
from servesim.sim import simulate, simulate_serial
from servesim.workload import Request, make_workload


def by_id(run):
    return {r.request.id: r for r in run.results}


class SerialTests(unittest.TestCase):
    def test_one_request_timing(self):
        r = Request(0, 1.0, 400, 3)
        res = simulate_serial([r]).results[0]
        prefill = config.prefill_time(400)
        step1 = config.decode_step_time(400)
        step2 = config.decode_step_time(401)
        self.assertAlmostEqual(res.first_token, 1.0 + prefill)
        self.assertAlmostEqual(res.done, 1.0 + prefill + step1 + step2)

    def test_second_request_waits_for_the_first(self):
        a = Request(0, 0.0, 400, 50)
        b = Request(1, 0.001, 400, 50)
        res = simulate_serial([a, b]).results
        self.assertAlmostEqual(res[1].admitted, res[0].done)
        self.assertGreater(res[1].ttft, res[0].ttft)

    def test_idle_server_starts_at_arrival(self):
        a = Request(0, 0.0, 100, 2)
        b = Request(1, 100.0, 100, 2)
        res = simulate_serial([a, b]).results
        self.assertEqual(res[1].admitted, 100.0)


class ContinuousTests(unittest.TestCase):
    def test_batch_of_one_matches_the_serial_baseline(self):
        reqs = make_workload(60, 0.5)
        serial = by_id(simulate_serial(reqs))
        one = by_id(simulate(reqs, max_batch=1))
        for i in serial:
            self.assertAlmostEqual(serial[i].done, one[i].done)

    def test_every_request_finishes_once(self):
        reqs = make_workload(300, 8.0)
        run = simulate(reqs)
        self.assertEqual(sorted(r.request.id for r in run.results), list(range(300)))

    def test_batch_cap_is_respected(self):
        run = simulate(make_workload(300, 20.0), max_batch=4)
        self.assertLessEqual(run.peak_batch, 4)

    def test_batching_beats_one_at_a_time(self):
        reqs = make_workload(300, 20.0)
        serial = max(r.done for r in simulate_serial(reqs).results)
        batched = max(r.done for r in simulate(reqs).results)
        self.assertLess(batched, serial / 5)

    def test_a_late_arrival_joins_between_steps(self):
        long_one = Request(0, 0.0, 100, 500)
        late = Request(1, 0.5, 100, 5)
        res = by_id(simulate([long_one, late]))
        self.assertLess(res[1].done, res[0].done)  # it did not wait for the long one

    def test_single_token_answer(self):
        res = simulate([Request(0, 0.0, 100, 1)]).results[0]
        self.assertAlmostEqual(res.done, res.first_token)


class MemoryBudgetTests(unittest.TestCase):
    def test_reserved_memory_never_passes_the_budget(self):
        reqs = make_workload(400, 14.0)
        for policy in ("fifo", "skip"):
            run = simulate(reqs, capacity=30000, policy=policy)
            self.assertLessEqual(run.peak_kv, 30000)
            self.assertEqual(len(run.results), 400)

    def test_two_big_requests_take_turns(self):
        a = Request(0, 0.0, 1000, 200)
        b = Request(1, 0.0, 1000, 200)
        res = by_id(simulate([a, b], capacity=1500))
        self.assertGreaterEqual(res[1].admitted, res[0].done)

    def test_request_that_can_never_fit_is_refused(self):
        with self.assertRaises(ValueError):
            simulate([Request(0, 0.0, 900, 200)], capacity=1000)

    def test_unknown_policy_is_refused(self):
        with self.assertRaises(ValueError):
            simulate([Request(0, 0.0, 10, 10)], policy="lifo")

    def test_fifo_blocks_behind_a_big_request_and_skip_does_not(self):
        running = Request(0, 0.0, 600, 300)    # holds 900 of 1000
        big = Request(1, 0.001, 600, 300)      # cannot fit beside it
        small = Request(2, 0.002, 50, 20)      # fits in the 100 left
        fifo = by_id(simulate([running, big, small], capacity=1000, policy="fifo"))
        skip = by_id(simulate([running, big, small], capacity=1000, policy="skip"))
        self.assertGreaterEqual(fifo[2].admitted, fifo[0].done)
        self.assertLess(skip[2].admitted, skip[0].done)


if __name__ == "__main__":
    unittest.main()
