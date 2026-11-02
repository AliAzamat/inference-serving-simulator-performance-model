import unittest

from servesim.metrics import percentile


class PercentileTests(unittest.TestCase):
    def test_median_of_odd_list(self):
        self.assertEqual(percentile([5, 1, 3], 50), 3)

    def test_p99_is_the_top_of_a_hundred(self):
        self.assertEqual(percentile(list(range(1, 101)), 99), 99)

    def test_p100_is_the_max(self):
        self.assertEqual(percentile([4, 9, 2], 100), 9)

    def test_one_value(self):
        self.assertEqual(percentile([7], 99), 7)


if __name__ == "__main__":
    unittest.main()
