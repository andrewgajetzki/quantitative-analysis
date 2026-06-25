import unittest

from measurements import (
    BuretCalibration,
    aqueous_molarity_at_temperature,
    apparent_mass_bias_percent,
    apparent_mass_from_true,
    concentration_after_evaporation,
    gravitational_reading_at_height,
    ideal_gas_density_g_per_ml,
    qcm_frequency_shift_hz,
    relative_humidity_percent,
    surface_mass_from_loading,
    true_mass_from_apparent,
    water_density_g_per_ml,
    water_mass_to_volume_ml,
    water_vapor_pressure_from_humidity,
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


if __name__ == "__main__":
    unittest.main()
