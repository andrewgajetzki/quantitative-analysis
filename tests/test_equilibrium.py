import math
import unittest

from equilibrium import (
    buffer_ph,
    classify_ph,
    combine_equilibrium_constants,
    complex_concentration_from_free,
    complex_distribution_fractions,
    complex_species_concentrations,
    conjugate_base_hydrolysis_constants,
    conjugate_kb,
    delta_g_from_enthalpy_entropy,
    delta_g_standard_from_equilibrium_constant,
    delta_gas_moles,
    equilibrium_constant_from_delta_g_standard,
    equilibrium_constant_at_temperature,
    equilibrium_constant_from_delta_h_delta_s,
    equilibrium_direction,
    equilibrium_expression,
    formation_constant,
    free_metal_from_total_metal,
    free_metal_concentration,
    henry_law_concentration,
    henry_law_pressure,
    hydrogen_from_ph,
    ion_product,
    ka_from_pka,
    ksp_from_molar_solubility,
    molar_solubility_from_ksp,
    molar_solubility_with_common_ions,
    neutral_ph,
    partial_pressures_from_moles,
    ph_from_hydrogen,
    polyprotic_acid_distribution_fractions,
    polyprotic_species_concentrations,
    precipitation_threshold_concentration,
    pressure_change_shift,
    pressure_reaction_quotient,
    proton_transfer_direction,
    proton_transfer_equilibrium_constant,
    reaction_quotient,
    salt_solution_character,
    solve_equilibrium,
    spontaneity_from_delta_g,
    successive_to_cumulative_constants,
    strong_acid_ph,
    strong_base_ph,
    temperature_change_shift,
    weak_acid_hydrogen_concentration,
    weak_acid_ph,
    weak_base_hydroxide_concentration,
    weak_base_ph,
    will_precipitate,
    kc_from_kp,
    kp_from_kc,
)


class EquilibriumConstantTests(unittest.TestCase):
    def test_reaction_quotient_and_direction(self):
        stoichiometry = {"HI": 2, "H2": -1, "I2": -1}
        concentrations = {"HI": 1.0, "H2": 0.10, "I2": 0.10}

        self.assertAlmostEqual(reaction_quotient(concentrations, stoichiometry), 100.0)
        self.assertEqual(equilibrium_direction(100.0, 50.0), "reverse")
        self.assertEqual(equilibrium_direction(10.0, 50.0), "forward")
        self.assertEqual(equilibrium_direction(50.0, 50.0), "at_equilibrium")

    def test_combine_equilibrium_constants_for_reversed_and_scaled_reactions(self):
        self.assertAlmostEqual(combine_equilibrium_constants((6.2e-10, -1)), 1.0 / 6.2e-10)
        self.assertAlmostEqual(combine_equilibrium_constants((2.0, 1), (3.0, 2)), 18.0)

    def test_equilibrium_expression(self):
        expression = equilibrium_expression({"HI": 2, "H2": -1, "I2": -1})

        self.assertEqual(expression, "[HI]^2 / [H2] [I2]")

    def test_solve_simple_ice_table(self):
        result = solve_equilibrium(
            {"A": 1.0, "B": 0.0},
            {"B": 1, "A": -1},
            equilibrium_constant=4.0,
        )

        self.assertAlmostEqual(result.extent, 0.8)
        self.assertAlmostEqual(result.concentrations["A"], 0.2)
        self.assertAlmostEqual(result.concentrations["B"], 0.8)
        self.assertAlmostEqual(result.reaction_quotient, 4.0)

    def test_solve_equilibrium_can_shift_reverse(self):
        result = solve_equilibrium(
            {"A": 0.10, "B": 1.0},
            {"B": 1, "A": -1},
            equilibrium_constant=4.0,
        )

        self.assertAlmostEqual(result.concentrations["B"] / result.concentrations["A"], 4.0)
        self.assertLess(result.extent, 0.0)


class AcidBaseTests(unittest.TestCase):
    def test_neutral_ph_and_classification_with_nonstandard_kw(self):
        kw_100_c = 5.5e-13

        self.assertAlmostEqual(neutral_ph(kw_100_c), 6.130, places=3)
        self.assertEqual(classify_ph(7.00, kw_100_c), "basic")
        self.assertEqual(classify_ph(neutral_ph(kw_100_c), kw_100_c), "neutral")

    def test_ph_and_strong_acid_base_helpers(self):
        self.assertAlmostEqual(hydrogen_from_ph(3.25), 5.623413251903491e-4)
        self.assertAlmostEqual(ph_from_hydrogen(0.0100), 2.0)
        self.assertAlmostEqual(strong_acid_ph(0.0100), 2.0)
        self.assertAlmostEqual(strong_base_ph(0.0350), 12.544068044350276)

    def test_weak_acid_helpers(self):
        hydrogen = weak_acid_hydrogen_concentration(0.100, 1.8e-5)

        self.assertAlmostEqual(hydrogen, 0.001332670973077975)
        self.assertAlmostEqual(weak_acid_ph(0.100, 1.8e-5), 2.875, places=3)

    def test_weak_base_helpers(self):
        hydroxide = weak_base_hydroxide_concentration(0.100, 1.8e-5)

        self.assertAlmostEqual(hydroxide, 0.001332670973077975)
        self.assertAlmostEqual(weak_base_ph(0.100, 1.8e-5), 11.125, places=3)

    def test_conjugate_constants_and_buffer_ph(self):
        ka = ka_from_pka(4.76)

        self.assertAlmostEqual(ka * conjugate_kb(ka), 1.0e-14)
        self.assertAlmostEqual(buffer_ph(0.20, 0.20, 4.76), 4.76)
        self.assertAlmostEqual(buffer_ph(0.10, 1.0, 4.76), 5.76)

    def test_proton_transfer_and_polyprotic_conjugate_base_constants(self):
        kb_values = conjugate_base_hydrolysis_constants((6.2e-5, 2.3e-6))

        self.assertAlmostEqual(proton_transfer_equilibrium_constant(4.76, 9.25), 10.0**4.49)
        self.assertEqual(proton_transfer_direction(4.76, 9.25), "forward")
        self.assertAlmostEqual(kb_values[0], 1.0e-14 / 2.3e-6)
        self.assertAlmostEqual(kb_values[1], 1.0e-14 / 6.2e-5)

    def test_salt_hydrolysis_classification(self):
        self.assertEqual(salt_solution_character().character, "neutral")
        self.assertEqual(salt_solution_character(cation_ka=5.6e-10).character, "acidic")
        self.assertEqual(salt_solution_character(anion_kb=1.5e-11).character, "basic")
        self.assertEqual(salt_solution_character(cation_ka=5.6e-10, anion_kb=1.5e-11).character, "acidic")

    def test_polyprotic_acid_distribution(self):
        fractions = polyprotic_acid_distribution_fractions(1.0e-8, (4.5e-7, 4.7e-11))

        self.assertAlmostEqual(sum(fractions), 1.0)
        self.assertAlmostEqual(fractions[2], 0.004576782835441395)

        species = polyprotic_species_concentrations(
            2.0e-5,
            1.0e-8,
            (4.5e-7, 4.7e-11),
            ("H2CO3", "HCO3-", "CO3^2-"),
        )
        self.assertAlmostEqual(species["CO3^2-"], 9.153565670882791e-8)


class SolubilityTests(unittest.TestCase):
    def test_ksp_and_molar_solubility_for_one_to_one_salt(self):
        solubility = molar_solubility_from_ksp(1.8e-10, {"Ag+": 1, "Cl-": 1})

        self.assertAlmostEqual(solubility, math.sqrt(1.8e-10))
        self.assertAlmostEqual(ksp_from_molar_solubility(solubility, {"Ag+": 1, "Cl-": 1}), 1.8e-10)

    def test_ksp_and_molar_solubility_for_one_to_two_salt(self):
        solubility = molar_solubility_from_ksp(3.9e-11, {"Ca2+": 1, "F-": 2})

        self.assertAlmostEqual(solubility, (3.9e-11 / 4.0) ** (1.0 / 3.0))
        self.assertAlmostEqual(ksp_from_molar_solubility(solubility, {"Ca2+": 1, "F-": 2}), 3.9e-11)

    def test_precipitation_decision(self):
        no_precipitate = will_precipitate({"Ag+": 1.0e-5, "Cl-": 1.0e-5}, 1.8e-10, {"Ag+": 1, "Cl-": 1})
        precipitate = will_precipitate({"Ag+": 1.0e-4, "Cl-": 1.0e-4}, 1.8e-10, {"Ag+": 1, "Cl-": 1})

        self.assertAlmostEqual(ion_product({"Ag+": 1.0e-4, "Cl-": 1.0e-4}, {"Ag+": 1, "Cl-": 1}), 1.0e-8)
        self.assertFalse(no_precipitate.precipitates)
        self.assertTrue(precipitate.precipitates)

    def test_common_ion_solubility_and_precipitation_threshold(self):
        solubility = molar_solubility_with_common_ions(
            1.8e-10,
            {"Ag+": 1, "Cl-": 1},
            {"Cl-": 0.0100},
        )

        self.assertAlmostEqual(solubility, 1.8e-8, places=10)
        self.assertAlmostEqual(
            precipitation_threshold_concentration(1.8e-10, 0.0100),
            1.8e-8,
        )


class ComplexThermoHenryTests(unittest.TestCase):
    def test_complex_formation_helpers(self):
        kf = formation_constant(1.0e-9, 0.10, 0.010, ligand_coefficient=2)

        self.assertAlmostEqual(kf, 1.0e9, delta=1.0)
        self.assertAlmostEqual(free_metal_concentration(0.010, 0.10, kf, ligand_coefficient=2), 1.0e-9)
        self.assertAlmostEqual(complex_concentration_from_free(1.0e-9, 0.10, kf, ligand_coefficient=2), 0.010)

    def test_cumulative_complex_formation_helpers(self):
        cumulative = successive_to_cumulative_constants((1.0e2, 1.0e3, 1.0e4))
        fractions = complex_distribution_fractions(0.10, cumulative)

        self.assertEqual(cumulative, (1.0e2, 1.0e5, 1.0e9))
        self.assertAlmostEqual(sum(fractions), 1.0)
        self.assertAlmostEqual(fractions[0], 1.0 / (1.0 + 10.0 + 1000.0 + 1.0e6))

        species = complex_species_concentrations(0.0100, 0.10, cumulative, ("M", "ML", "ML2", "ML3"))
        self.assertAlmostEqual(species["M"], free_metal_from_total_metal(0.0100, 0.10, cumulative))
        self.assertGreater(species["ML3"], 0.009)

    def test_delta_g_and_equilibrium_constant_round_trip(self):
        equilibrium_constant = equilibrium_constant_from_delta_g_standard(-59_000.0, 298.15)

        self.assertGreater(equilibrium_constant, 1.0e10)
        self.assertAlmostEqual(
            delta_g_standard_from_equilibrium_constant(equilibrium_constant, 298.15),
            -59_000.0,
            places=8,
        )

    def test_enthalpy_entropy_and_vant_hoff_helpers(self):
        delta_g_value = delta_g_from_enthalpy_entropy(10_000.0, 50.0, 298.15)
        equilibrium_constant = equilibrium_constant_from_delta_h_delta_s(10_000.0, 50.0, 298.15)
        warmer_k = equilibrium_constant_at_temperature(1.0e-5, 298.15, 350.0, -40_000.0)

        self.assertAlmostEqual(delta_g_value, -4907.5)
        self.assertEqual(spontaneity_from_delta_g(delta_g_value), "forward")
        self.assertAlmostEqual(
            equilibrium_constant,
            equilibrium_constant_from_delta_g_standard(delta_g_value, 298.15),
        )
        self.assertLess(warmer_k, 1.0e-5)

    def test_henrys_law_helpers(self):
        concentration = henry_law_concentration(3.0e-8, 0.20)

        self.assertAlmostEqual(concentration, 6.0e-9)
        self.assertAlmostEqual(henry_law_pressure(concentration, 3.0e-8), 0.20)

    def test_gas_equilibrium_helpers(self):
        stoichiometry = {"NH3": 2, "N2": -1, "H2": -3}
        pressures = partial_pressures_from_moles({"N2": 1.0, "H2": 3.0, "NH3": 0.5}, 10.0)
        kc = kc_from_kp(2.0e-4, delta_gas_moles(stoichiometry), 500.0)

        self.assertAlmostEqual(sum(pressures.values()), 10.0)
        self.assertAlmostEqual(kp_from_kc(kc, delta_gas_moles(stoichiometry), 500.0), 2.0e-4)
        self.assertAlmostEqual(pressure_reaction_quotient(pressures, stoichiometry), 0.001875)

    def test_le_chatelier_shift_helpers(self):
        self.assertEqual(pressure_change_shift({"CF4": 1, "F2": -2}, pressure_increases=True), "forward")
        self.assertEqual(pressure_change_shift({"HI": 2, "H2": -1, "I2": -1}, pressure_increases=True), "no_shift")
        self.assertEqual(temperature_change_shift(-59_000.0, temperature_increases=True), "reverse")
        self.assertEqual(temperature_change_shift(59_000.0, temperature_increases=True), "forward")


if __name__ == "__main__":
    unittest.main()
