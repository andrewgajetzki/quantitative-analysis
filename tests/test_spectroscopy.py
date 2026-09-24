import unittest

from measurements import linear_least_squares
from spectroscopy import (
    absorbance_from_percent_transmittance,
    beer_lambert_concentration,
    beer_lambert_molar_absorptivity,
    calibration_concentrations_from_stock,
    concentration_from_calibration_signal,
    corrected_absorbance,
    dilution_corrected_signals,
    frequency_from_wavelength_nm,
    mass_concentration_from_molar_concentration,
    molar_absorptivity_from_mass_calibration_slope,
    molar_concentration_from_mass_concentration,
    percent_transmittance_from_absorbance,
    photon_energy_j_from_wavelength_nm,
    photon_energy_kj_per_mol_from_wavelength_nm,
    protein_molar_absorptivity_a280,
    standard_addition_from_added_amounts,
    two_line_endpoint,
    wavelength_nm_from_frequency_hz,
    wavelength_nm_from_wavenumber_cm,
    wavenumber_from_wavelength_nm,
)


class ElectromagneticRadiationTests(unittest.TestCase):
    def test_visible_photon_properties_from_wavelength(self):
        wavelength_nm = 562.0

        self.assertAlmostEqual(frequency_from_wavelength_nm(wavelength_nm), 5.334385e14, delta=5.0e8)
        self.assertAlmostEqual(wavenumber_from_wavelength_nm(wavelength_nm), 17793.5943, places=4)
        self.assertAlmostEqual(photon_energy_j_from_wavelength_nm(wavelength_nm), 3.534601e-19, places=25)
        self.assertAlmostEqual(photon_energy_kj_per_mol_from_wavelength_nm(wavelength_nm), 212.8587, places=4)
        self.assertAlmostEqual(wavelength_nm_from_frequency_hz(5.3343853736654806e14), 562.0, places=8)
        self.assertAlmostEqual(wavelength_nm_from_wavenumber_cm(17793.59430604982), 562.0, places=8)


class AbsorbanceAndBeerLawTests(unittest.TestCase):
    def test_absorbance_transmittance_conversions(self):
        self.assertAlmostEqual(percent_transmittance_from_absorbance(0.822), 15.06607, places=5)
        self.assertAlmostEqual(absorbance_from_percent_transmittance(15.06607066186742), 0.822, places=12)

    def test_blank_corrected_molar_absorptivity_and_concentration(self):
        absorbance = corrected_absorbance(0.624, blank_absorbance=0.029)
        epsilon = beer_lambert_molar_absorptivity(absorbance, 3.96e-4, path_length_cm=1.000)
        epsilon_from_direct_question = beer_lambert_molar_absorptivity(0.822, 2.31e-5, path_length_cm=1.000)
        concentration = beer_lambert_concentration(absorbance, epsilon, path_length_cm=1.000)

        self.assertAlmostEqual(absorbance, 0.595)
        self.assertAlmostEqual(epsilon, 1502.5253, places=4)
        self.assertAlmostEqual(epsilon_from_direct_question, 35584.4156, places=4)
        self.assertAlmostEqual(concentration, 3.96e-4, places=12)

    def test_mass_concentration_and_molar_absorptivity_conversions(self):
        concentration = molar_concentration_from_mass_concentration(15.0, 4568.0, unit="ug/mL")
        self.assertAlmostEqual(concentration, 3.2837e-6, places=10)
        self.assertAlmostEqual(
            mass_concentration_from_molar_concentration(concentration, 4568.0, unit="ug/mL"),
            15.0,
            places=12,
        )

        dna_fit = linear_least_squares([2.5, 5.0, 10.0], [0.080, 0.149, 0.307])
        epsilon = molar_absorptivity_from_mass_calibration_slope(dna_fit.slope, 4568.0, path_length_cm=1.0)
        self.assertAlmostEqual(dna_fit.slope, 0.03045714, places=8)
        self.assertAlmostEqual(epsilon, 139128.2286, places=4)

    def test_protein_a280_estimated_molar_absorptivity(self):
        epsilon = protein_molar_absorptivity_a280(
            tryptophan_count=8,
            tyrosine_count=26,
            disulfide_count=19,
        )

        self.assertAlmostEqual(epsilon, 85115.0)


class SpectrophotometricCalibrationTests(unittest.TestCase):
    def test_standard_concentrations_from_stock_dilutions(self):
        concentrations = calibration_concentrations_from_stock(
            stock_concentration=1.40,
            aliquot_volumes=[0.0, 10.0, 20.0, 30.0, 40.0],
            final_volume=2000.0,
        )

        self.assertEqual(concentrations, (0.0, 0.007, 0.014, 0.021, 0.028))

    def test_concentration_from_calibration_signal_with_dilution(self):
        fit = linear_least_squares([0.0, 1.0, 2.0], [0.100, 0.300, 0.500])

        self.assertAlmostEqual(concentration_from_calibration_signal(fit, 0.300), 1.0)
        self.assertAlmostEqual(
            concentration_from_calibration_signal(
                fit,
                0.300,
                sample_aliquot_volume=5.00,
                final_volume=50.00,
            ),
            10.0,
        )

    def test_dilution_corrected_photometric_endpoint(self):
        volumes_ul = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 70, 80]
        absorbances = [0.227, 0.256, 0.286, 0.316, 0.345, 0.370, 0.399, 0.422, 0.443, 0.448, 0.449, 0.450, 0.447]
        corrected = dilution_corrected_signals(
            absorbances,
            initial_volume=2.025,
            added_volumes=[volume / 1000.0 for volume in volumes_ul],
        )
        endpoint = two_line_endpoint(
            list(zip(volumes_ul[:6], corrected[:6])),
            list(zip(volumes_ul[7:], corrected[7:])),
        )

        self.assertAlmostEqual(corrected[0], 0.227)
        self.assertAlmostEqual(corrected[-1], 0.46465926, places=8)
        self.assertAlmostEqual(endpoint.x_value, 43.4384, places=4)
        self.assertAlmostEqual(endpoint.y_value, 0.4447, places=4)

    def test_standard_addition_from_added_amounts_and_sample_mass(self):
        result = standard_addition_from_added_amounts(
            added_amounts=[4.0, 8.0, 12.0, 16.0],
            signals=[0.800, 1.090, 1.359, 1.637],
            blank_signal=0.029,
            sample_aliquot_volume=1.00,
            total_sample_volume=500.0,
            sample_mass=1.12,
        )

        self.assertAlmostEqual(result.fit.slope, 0.0695, places=4)
        self.assertAlmostEqual(result.x_intercept, -7.1583, places=4)
        self.assertAlmostEqual(result.unknown_amount, 7.1583, places=4)
        self.assertAlmostEqual(result.original_sample_amount, 3579.1367, places=4)
        self.assertAlmostEqual(result.amount_per_sample_mass, 3195.6578, places=4)


if __name__ == "__main__":
    unittest.main()
