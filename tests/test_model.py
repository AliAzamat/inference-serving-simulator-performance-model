import unittest

from servesim.model import batch_limit, max_rate, mean_shape, predict
from servesim.workload import make_workload


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.p, self.o = mean_shape(make_workload(4000, 5.0))

    def test_more_load_means_a_bigger_batch_and_slower_answers(self):
        low, high = predict(2, self.p, self.o), predict(10, self.p, self.o)
        self.assertLess(low["batch"], high["batch"])
        self.assertLess(low["latency"], high["latency"])

    def test_past_the_maximum_rate_the_model_says_saturated(self):
        top = max_rate(self.p, self.o)
        self.assertTrue(predict(top * 1.01, self.p, self.o)["saturated"])
        self.assertFalse(predict(top * 0.9, self.p, self.o)["saturated"])

    def test_predicted_batch_stays_inside_the_limit(self):
        top = max_rate(self.p, self.o)
        m = predict(top * 0.999, self.p, self.o)
        self.assertLessEqual(m["batch"], batch_limit(self.p, self.o) * 1.01)

    def test_a_smaller_memory_budget_lowers_the_maximum_rate(self):
        big = max_rate(self.p, self.o, capacity=60000)
        small = max_rate(self.p, self.o, capacity=20000)
        self.assertLess(small, big)

    def test_tokens_per_second_is_rate_times_answer_length(self):
        m = predict(4, self.p, self.o)
        self.assertAlmostEqual(m["tokens_per_s"], 4 * self.o)


if __name__ == "__main__":
    unittest.main()
