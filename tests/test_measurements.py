import unittest

from measurements import (
    BuretCalibration,
    aqueous_molarity_at_temperature,
    apparent_mass_bias_percent,
    apparent_mass_from_true,
    compare_means_from_stats,
    concentration_after_evaporation,
    confidence_interval_mean,
    f_test_variances_from_stats,
    gravitational_reading_at_height,
    grubbs_test,
    ideal_gas_density_g_per_ml,
    inverse_linear_interpolate,
    linear_least_squares,
    log10_least_squares,
    mean,
    normal_probability_between,
    one_sample_t_test,
    paired_t_test,
    pooled_standard_deviation,
    qcm_frequency_shift_hz,
    relative_humidity_percent,
    sample_standard_deviation,
    standard_deviation_of_mean,
    surface_mass_from_loading,
    t_critical_two_tailed,
    true_mass_from_apparent,
    water_density_g_per_ml,
    water_mass_to_volume_ml,
    water_vapor_pressure_from_humidity,
    x_from_log10_calibration_y,
)


class BuoyancyTests(unittest.TestCase):
    def test_density_equal_to_balance_weights_has_no_correction(self):
        self.assertAlmostEqual(true_mass_from_apparent(10.0, 8.0), 10.0)

    def test_water_true_mass_from_apparent(self):
        true_mass = true_mass_from_apparent(5.3974, water_density_g_per_ml(25.0))
        self.assertAlmostEqual(true_mass, 5.4031, places=4)

    def test_apparent_mass_from_true(self):
        apparent = apparent_mass_from_true(1.6780, object_density_g_per_ml=3.988)
        self.assertAlmostEqual(apparent, 1.6777, places=4)

    def test_apparent_mass_bias_for_khp(self):
        self.assertAlmostEqual(apparent_mass_bias_percent(1.636), -0.0584, places=4)


class GlasswareCalibrationTests(unittest.TestCase):
    def test_water_mass_to_volume_matches_buret_example_factor(self):
        self.assertAlmostEqual(water_mass_to_volume_ml(9.984, 24.0), 10.022, places=3)

    def test_buret_calibration_interval(self):
        calibration = BuretCalibration.from_water_masses(
            readings_ml=[(0.03, 10.01), (10.01, 19.90), (19.90, 30.06)],
            apparent_water_masses_g=[9.984, 9.835, 10.071],
            temp_c=24.0,
        )
        self.assertAlmostEqual(calibration.intervals[0].correction_ml, 0.042, places=3)


class TemperatureAndGasTests(unittest.TestCase):
    def test_aqueous_molarity_at_lower_temperature(self):
        corrected = aqueous_molarity_at_temperature(0.05138, 24.0, 16.0)
        self.assertAlmostEqual(corrected, 0.05146, places=5)

    def test_helium_density_from_ideal_gas_law(self):
        density = ideal_gas_density_g_per_ml(4.0026, temp_c=20.0, pressure_bar=1.0)
        self.assertAlmostEqual(density, 0.000164, places=6)

    def test_gravity_reading_at_higher_floor(self):
        reading = gravitational_reading_at_height(100.0000, height_change_m=30.0)
        self.assertAlmostEqual(reading, 99.9991, places=4)


class OtherMeasurementTests(unittest.TestCase):
    def test_humidity_helpers(self):
        partial_pressure = water_vapor_pressure_from_humidity(42.0, 2330.0)
        self.assertAlmostEqual(partial_pressure, 978.6)
        self.assertAlmostEqual(relative_humidity_percent(partial_pressure, 2330.0), 42.0)

    def test_qcm_helpers(self):
        mass_ug = surface_mass_from_loading(area_mm2=33.0, loading_ug_per_cm2=0.090)
        self.assertAlmostEqual(mass_ug, 0.0297)
        self.assertAlmostEqual(qcm_frequency_shift_hz(90.0, 10.0), 900.0)

    def test_concentration_after_evaporation(self):
        self.assertAlmostEqual(concentration_after_evaporation(1.0, 100.0, 10.0), 1.111111, places=6)


class StatisticsTests(unittest.TestCase):
    def test_summary_statistics_and_confidence_interval(self):
        values = [116.0, 97.9, 114.2, 106.8, 108.3]

        self.assertAlmostEqual(mean(values), 108.64)
        self.assertAlmostEqual(sample_standard_deviation(values), 7.1402, places=4)
        self.assertAlmostEqual(standard_deviation_of_mean(values), 3.1932, places=4)

        interval = confidence_interval_mean(values, confidence=0.90)
        self.assertAlmostEqual(interval.critical_value, 2.1318, places=4)
        self.assertAlmostEqual(interval.lower, 101.8326, places=4)
        self.assertAlmostEqual(interval.upper, 115.4474, places=4)

    def test_gaussian_probability_areas(self):
        self.assertAlmostEqual(normal_probability_between(-1, 1), 0.682689, places=6)
        self.assertAlmostEqual(normal_probability_between(-2, 2), 0.954500, places=6)
        self.assertAlmostEqual(normal_probability_between(0, 0.5), 0.191462, places=6)

    def test_grubbs_identifies_candidate_but_does_not_reject(self):
        result = grubbs_test([116.0, 97.9, 114.2, 106.8, 108.3], confidence=0.90)

        self.assertEqual(result.outlier_index, 1)
        self.assertEqual(result.outlier_value, 97.9)
        self.assertAlmostEqual(result.statistic, 1.5042, places=4)
        self.assertAlmostEqual(result.critical_value, 1.6714, places=4)
        self.assertFalse(result.significant)

    def test_f_test_variances_from_summary_stats(self):
        result = f_test_variances_from_stats(0.025, 4, 0.039, 4, confidence=0.95)

        self.assertAlmostEqual(result.statistic, 2.4336, places=4)
        self.assertAlmostEqual(result.critical_value, 9.2766, places=4)
        self.assertFalse(result.significant)

    def test_two_sample_t_test_from_summary_stats(self):
        result = compare_means_from_stats(1.392, 0.025, 4, 1.346, 0.039, 4, confidence=0.95)

        self.assertAlmostEqual(pooled_standard_deviation(0.025, 4, 0.039, 4), 0.03276, places=5)
        self.assertAlmostEqual(result.statistic, 1.9860, places=4)
        self.assertAlmostEqual(result.critical_value, 2.4469, places=4)
        self.assertFalse(result.significant)

    def test_one_sample_and_paired_t_tests(self):
        one_sample = one_sample_t_test([98.4, 97.2, 94.6, 96.2], expected_mean=98.6, confidence=0.95)
        paired = paired_t_test([4.70, 3.81], [4.84, 3.82], confidence=0.95)

        self.assertAlmostEqual(one_sample.statistic, -2.4871, places=4)
        self.assertFalse(one_sample.significant)
        self.assertAlmostEqual(paired.difference, -0.075, places=3)
        self.assertFalse(paired.significant)

    def test_distribution_critical_values(self):
        self.assertAlmostEqual(t_critical_two_tailed(0.95, 4), 2.7764, places=4)
        self.assertAlmostEqual(t_critical_two_tailed(0.90, 4), 2.1318, places=4)


class LinearCalibrationTests(unittest.TestCase):
    def test_linear_least_squares_returns_slope_intercept_and_uncertainties(self):
        fit = linear_least_squares(
            [3.0, 10.0, 20.0, 30.0, 40.0],
            [-0.074, -1.411, -2.584, -3.750, -5.407],
        )

        self.assertAlmostEqual(fit.slope, -0.13789, places=5)
        self.assertAlmostEqual(fit.intercept, 0.19534, places=5)
        self.assertAlmostEqual(fit.slope_uncertainty, 0.006635, places=6)
        self.assertAlmostEqual(fit.intercept_uncertainty, 0.16276, places=5)
        self.assertAlmostEqual(fit.r_squared, 0.99310, places=5)

    def test_inverse_calibration_prediction(self):
        fit = linear_least_squares(
            [0.00, 9.36, 18.72, 28.08, 37.44],
            [0.446, 0.676, 0.883, 1.086, 1.280],
        )
        prediction = fit.inverse_prediction(0.973, confidence=0.95)

        self.assertAlmostEqual(prediction.x_value, 23.1703, places=4)
        self.assertAlmostEqual(prediction.standard_uncertainty, 0.6034, places=4)
        self.assertIsNotNone(prediction.confidence_interval)
        self.assertAlmostEqual(prediction.confidence_interval.half_width, 1.9204, places=4)

    def test_log10_calibration_prediction(self):
        fit = log10_least_squares(
            [0.0100, 0.0299, 0.117, 0.311, 1.02],
            [0.215, 0.846, 2.65, 7.41, 20.8],
        )

        self.assertAlmostEqual(fit.slope, 0.97493, places=5)
        self.assertAlmostEqual(x_from_log10_calibration_y(fit, 99.9), 4.7448, places=4)

    def test_inverse_linear_interpolation_for_nonlinear_calibration(self):
        points = [(0.0, 0.10), (5.0, 0.32), (10.0, 0.61)]

        self.assertAlmostEqual(inverse_linear_interpolate(0.350, points), 5.5172, places=4)


if __name__ == "__main__":
    unittest.main()
