import unittest

from servesim.workload import make_workload, describe


class WorkloadTests(unittest.TestCase):
    def test_same_seed_same_requests(self):
        self.assertEqual(make_workload(50, 5.0, seed=3), make_workload(50, 5.0, seed=3))

    def test_arrivals_increase(self):
        reqs = make_workload(200, 5.0)
        times = [r.arrival for r in reqs]
        self.assertEqual(times, sorted(times))

    def test_rate_changes_arrivals_not_lengths(self):
        slow = make_workload(100, 1.0)
        fast = make_workload(100, 9.0)
        self.assertEqual([(r.prompt, r.output) for r in slow], [(r.prompt, r.output) for r in fast])
        self.assertGreater(slow[-1].arrival, fast[-1].arrival)

    def test_average_arrival_rate(self):
        reqs = make_workload(4000, 10.0)
        measured = len(reqs) / reqs[-1].arrival
        self.assertAlmostEqual(measured, 10.0, delta=0.6)

    def test_describe(self):
        d = describe(make_workload(1000, 5.0))
        self.assertEqual(d["n"], 1000)
        self.assertGreater(d["mean_prompt"], 600)


if __name__ == "__main__":
    unittest.main()
