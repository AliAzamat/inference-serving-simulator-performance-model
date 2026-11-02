import unittest

from servesim.compare import compare


def err(predicted, measured):
    return abs(predicted - measured) / measured


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {r["rate"]: r for r in compare([1, 4, 8, 12, 14], n=3000)}

    def test_median_latency_within_20_percent_below_saturation(self):
        for rate in (1, 4, 8, 12):
            r = self.rows[rate]
            self.assertLess(err(r["pred_p50"], r["got"]["p50"]), 0.20, rate)

    def test_throughput_within_5_percent_below_saturation(self):
        for rate in (1, 4, 8, 12):
            r = self.rows[rate]
            self.assertLess(err(r["pred_tokens"], r["got"]["tokens_per_s"]), 0.05, rate)

    def test_p99_prediction_is_a_floor_not_a_match(self):
        for rate in (8, 12):
            r = self.rows[rate]
            self.assertLess(r["pred_p99"], r["got"]["p99"])

    def test_model_calls_saturation_where_the_simulator_stops_scaling(self):
        self.assertTrue(self.rows[14]["saturated"])
        self.assertFalse(self.rows[12]["saturated"])
        ceiling = self.rows[14]["pred_tokens"]
        self.assertLess(err(ceiling, self.rows[14]["got"]["tokens_per_s"]), 0.10)


if __name__ == "__main__":
    unittest.main()
