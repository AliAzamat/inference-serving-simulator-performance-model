import unittest

from servesim.compare import calibrate, p99_on_seeds, rate_for_p99
from servesim.workload import make_workload

FRESH = range(2, 22)  # twenty seeds; calibration uses seed 1


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.factor = calibrate([4, 8, 10], seed=1)
        cls.shape = make_workload(4000, 5.0, seed=7)

    def test_the_model_runs_under_the_tail(self):
        self.assertGreater(self.factor, 1.0)

    def test_far_below_the_knee_the_plan_holds_on_every_new_seed(self):
        rate = rate_for_p99(4.0, self.factor, self.shape)
        self.assertGreater(rate, 1.0)
        self.assertLess(max(p99_on_seeds(rate, FRESH)), 4.0)

    def test_one_new_seed_overstates_the_plan_near_the_knee(self):
        rate = rate_for_p99(8.0, self.factor, self.shape)
        self.assertLess(p99_on_seeds(rate, [7])[0], 8.0)       # seed 7 alone says it holds
        self.assertGreater(max(p99_on_seeds(rate, FRESH)), 8.0)  # twenty seeds say it does not

    def test_the_backed_off_plan_holds_on_every_new_seed(self):
        self.assertLess(max(p99_on_seeds(9.25, FRESH)), 8.0)

    def test_a_tighter_target_allows_less_load(self):
        self.assertLess(rate_for_p99(4.0, self.factor, self.shape), rate_for_p99(8.0, self.factor, self.shape))


if __name__ == "__main__":
    unittest.main()
