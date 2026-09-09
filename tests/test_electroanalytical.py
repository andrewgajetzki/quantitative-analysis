import math
import unittest

from electrochemistry import (
    FARADAY_CONSTANT_C_PER_MOL,
    R_J_PER_MOL_K,
    STANDARD_TEMPERATURE_K,
    amperometric_titration_endpoint,
    concentration_from_cottrell_current,
    concentration_from_diffusion_limited_current,
    concentration_from_randles_sevcik_peak_current,
    conductometric_titration_endpoint,
    conductivity_from_resistance,
    coulometric_analysis,
    coulometric_analyte_moles_from_charge,
    coulometric_charge_for_analyte,
    coulometric_generation_time,
    cottrell_current,
    cyclic_voltammetry_formal_potential,
    cyclic_voltammetry_peak_separation,
    diffusion_limited_current,
    molar_conductivity,
    randles_sevcik_peak_current,
    resistance_from_conductivity,
    reversible_cv_electron_count_from_peak_separation,
    voltammetric_standard_addition_concentration,
)


class CoulometricAnalysisTests(unittest.TestCase):
    def test_constant_current_coulometric_analysis_returns_moles_and_molarity(self):
        result = coulometric_analysis(
            current_a=0.0500,
            time_s=612.0,
            electrons_per_mole_analyte=2,
            sample_volume_ml=25.00,
        )
        expected_moles = 30.6 / FARADAY_CONSTANT_C_PER_MOL / 2.0

        self.assertAlmostEqual(result.charge_c, 30.6)
        self.assertAlmostEqual(result.analyte_charge_c, 30.6)
        self.assertAlmostEqual(result.moles_electrons, 30.6 / FARADAY_CONSTANT_C_PER_MOL)
        self.assertAlmostEqual(result.analyte_moles, expected_moles)
        self.assertAlmostEqual(result.analyte_molarity, expected_moles / 0.02500)

    def test_coulometric_blank_efficiency_charge_and_time_helpers(self):
        corrected = coulometric_analyte_moles_from_charge(
            charge_c=30.6,
            electrons_per_mole_analyte=2,
            current_efficiency=0.995,
            blank_charge_c=0.2,
        )
        required_charge = coulometric_charge_for_analyte(1.00e-4, 2, current_efficiency=0.995)
        generation_time = coulometric_generation_time(1.00e-4, 0.0500, 2)

        self.assertAlmostEqual(corrected, 30.4 * 0.995 / FARADAY_CONSTANT_C_PER_MOL / 2)
        self.assertAlmostEqual(required_charge, 2.0 * FARADAY_CONSTANT_C_PER_MOL * 1.00e-4 / 0.995)
        self.assertAlmostEqual(generation_time, 385.9413, places=4)


class VoltammetryTests(unittest.TestCase):
    def test_diffusion_limited_current_and_concentration_round_trip(self):
        current = diffusion_limited_current(
            electrons_transferred=1,
            electrode_area_cm2=0.100,
            diffusion_coefficient_cm2_s=7.00e-6,
            concentration_mol_l=1.00e-3,
            diffusion_layer_thickness_cm=0.00500,
        )
        concentration = concentration_from_diffusion_limited_current(
            current,
            electrons_transferred=1,
            electrode_area_cm2=0.100,
            diffusion_coefficient_cm2_s=7.00e-6,
            diffusion_layer_thickness_cm=0.00500,
        )

        self.assertAlmostEqual(current, 1.35079465e-5)
        self.assertAlmostEqual(concentration, 1.00e-3)

    def test_cottrell_current_and_concentration_round_trip(self):
        current = cottrell_current(
            electrons_transferred=1,
            electrode_area_cm2=0.0500,
            diffusion_coefficient_cm2_s=7.00e-6,
            concentration_mol_l=1.00e-3,
            time_s=10.0,
        )
        concentration = concentration_from_cottrell_current(
            current,
            electrons_transferred=1,
            electrode_area_cm2=0.0500,
            diffusion_coefficient_cm2_s=7.00e-6,
            time_s=10.0,
        )

        expected = (
            FARADAY_CONSTANT_C_PER_MOL
            * 0.0500
            * 1.00e-6
            * math.sqrt(7.00e-6)
            / math.sqrt(math.pi * 10.0)
        )
        self.assertAlmostEqual(current, expected)
        self.assertAlmostEqual(concentration, 1.00e-3)

    def test_randles_sevcik_peak_current_and_concentration_round_trip(self):
        current = randles_sevcik_peak_current(
            electrons_transferred=1,
            electrode_area_cm2=0.0707,
            diffusion_coefficient_cm2_s=7.60e-6,
            concentration_mol_l=1.00e-3,
            scan_rate_v_s=0.100,
        )
        concentration = concentration_from_randles_sevcik_peak_current(
            current,
            electrons_transferred=1,
            electrode_area_cm2=0.0707,
            diffusion_coefficient_cm2_s=7.60e-6,
            scan_rate_v_s=0.100,
        )

        expected = (
            0.4463
            * FARADAY_CONSTANT_C_PER_MOL
            * 0.0707
            * 1.00e-6
            * math.sqrt(
                FARADAY_CONSTANT_C_PER_MOL
                * 7.60e-6
                * 0.100
                / (R_J_PER_MOL_K * STANDARD_TEMPERATURE_K)
            )
        )
        self.assertAlmostEqual(current, expected)
        self.assertAlmostEqual(concentration, 1.00e-3)

    def test_voltammetric_standard_addition(self):
        unknown = voltammetric_standard_addition_concentration(
            initial_current_a=10.0e-6,
            final_current_a=27.27272727272727e-6,
            sample_volume_ml=10.00,
            standard_volume_ml=1.00,
            standard_concentration=1.00e-4,
        )

        self.assertAlmostEqual(unknown, 5.00e-6)

    def test_cyclic_voltammetry_peak_relationships(self):
        formal_potential = cyclic_voltammetry_formal_potential(0.310, 0.250)
        peak_separation = cyclic_voltammetry_peak_separation(0.310, 0.250)
        electron_count = reversible_cv_electron_count_from_peak_separation(peak_separation)

        self.assertAlmostEqual(formal_potential, 0.280)
        self.assertAlmostEqual(peak_separation, 0.060)
        self.assertAlmostEqual(electron_count, 0.98598916)


class EndpointAndConductometryTests(unittest.TestCase):
    def test_amperometric_titration_endpoint_from_line_intersection(self):
        endpoint = amperometric_titration_endpoint(
            before_endpoint_points=((0.0, 1.0), (5.0, 2.0), (8.0, 2.6)),
            after_endpoint_points=((12.0, 3.8), (15.0, 5.0), (20.0, 7.0)),
        )

        self.assertAlmostEqual(endpoint.endpoint_volume_ml, 10.0)
        self.assertAlmostEqual(endpoint.endpoint_signal, 3.0)
        self.assertAlmostEqual(endpoint.before_slope, 0.2)
        self.assertAlmostEqual(endpoint.after_slope, 0.4)

    def test_conductivity_molar_conductivity_and_conductometric_endpoint(self):
        conductivity = conductivity_from_resistance(
            resistance_ohm=500.0,
            cell_constant_cm_inverse=1.000,
        )
        resistance = resistance_from_conductivity(conductivity, cell_constant_cm_inverse=1.000)
        molar = molar_conductivity(conductivity, concentration_mol_l=0.0100)
        endpoint = conductometric_titration_endpoint(
            before_endpoint_points=((0.0, 10.0), (10.0, 9.0), (20.0, 8.0)),
            after_endpoint_points=((30.0, 13.0), (40.0, 18.0), (50.0, 23.0)),
        )

        self.assertAlmostEqual(conductivity, 0.00200)
        self.assertAlmostEqual(resistance, 500.0)
        self.assertAlmostEqual(molar, 200.0)
        self.assertAlmostEqual(endpoint.endpoint_volume_ml, 20.0)
        self.assertAlmostEqual(endpoint.endpoint_signal, 8.0)


if __name__ == "__main__":
    unittest.main()
