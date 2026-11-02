import unittest

from servesim.loadtest import find_knee, sweep


class LoadTestTests(unittest.TestCase):
    def test_sweep_returns_one_row_per_rate(self):
        rows = sweep([1, 4], n=300)
        self.assertEqual([r["rate"] for r in rows], [1, 4])

    def test_more_load_never_makes_p99_better(self):
        rows = sweep([1, 6, 12, 16], n=600)
        p99 = [r["p99"] for r in rows]
        self.assertEqual(p99, sorted(p99))

    def test_throughput_levels_off(self):
        rows = sweep([14, 20], n=1500)
        self.assertAlmostEqual(rows[0]["tokens_per_s"], rows[1]["tokens_per_s"], delta=rows[1]["tokens_per_s"] * 0.05)

    def test_find_knee(self):
        rows = [{"rate": 1, "p99": 2.0}, {"rate": 2, "p99": 3.0}, {"rate": 3, "p99": 7.0}]
        self.assertEqual(find_knee(rows), 3)
        self.assertIsNone(find_knee(rows[:2]))


if __name__ == "__main__":
    unittest.main()
