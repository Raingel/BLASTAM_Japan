import unittest

import numpy as np

import run_10_years
import run_blastam_assessment


def base_arrays(temp=22.0):
    return (
        np.full(120, temp, dtype=float),
        np.zeros(120, dtype=float),
        np.zeros(120, dtype=float),
        np.zeros(120, dtype=float),
    )


class KoshimizuModelTest(unittest.TestCase):
    def assert_model_case(self, configure, check):
        for module in (run_blastam_assessment, run_10_years):
            temp, wind, rain, sun = base_arrays()
            configure(temp, wind, rain, sun)
            leaf_wet, result = module.koshimizu_model(temp, wind, rain, sun)
            check(self, leaf_wet, result)

    def test_early_morning_wind_rule_does_not_apply_all_night(self):
        def configure(temp, wind, rain, sun):
            rain[89] = 0.5
            wind[92] = 3.0

        def check(case, leaf_wet, result):
            case.assertIs(leaf_wet[20], True)

        self.assert_model_case(configure, check)

    def test_exact_four_mm_rain_is_invalidating(self):
        def configure(temp, wind, rain, sun):
            rain[89] = 0.5
            rain[95] = 4.0

        def check(case, leaf_wet, result):
            case.assertEqual(leaf_wet[23], -2)

        self.assert_model_case(configure, check)

    def test_two_consecutive_three_mm_rain_hours_are_invalidating(self):
        def configure(temp, wind, rain, sun):
            rain[89] = 0.5
            rain[96] = 3.0
            rain[97] = 3.0

        def check(case, leaf_wet, result):
            case.assertEqual(leaf_wet[0], -2)

        self.assert_model_case(configure, check)

    def test_twenty_one_degrees_requires_eleven_wet_hours(self):
        def configure(temp, wind, rain, sun):
            temp[:] = 21.0
            rain[89] = 0.5
            sun[98] = 0.3

        def check(case, leaf_wet, result):
            case.assertEqual(result['wet_period_hrs'], 10)
            case.assertEqual(result['blast_score'], 4)

        self.assert_model_case(configure, check)

    def test_rain_with_exact_three_mps_wind_is_treated_as_two_mps(self):
        def configure(temp, wind, rain, sun):
            rain[89] = 0.5
            wind[97:100] = 3.0
            rain[98] = 0.5

        def check(case, leaf_wet, result):
            case.assertIs(leaf_wet[2], True)

        self.assert_model_case(configure, check)


if __name__ == '__main__':
    unittest.main()
