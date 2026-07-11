import math
import unittest

from equilibrium import (
    buffer_ph,
    combine_equilibrium_constants,
    complex_concentration_from_free,
    conjugate_kb,
    delta_g_standard_from_equilibrium_constant,
    equilibrium_constant_from_delta_g_standard,
    equilibrium_direction,
    formation_constant,
    free_metal_concentration,
    henry_law_concentration,
    henry_law_pressure,
    hydrogen_from_ph,
    ion_product,
    ka_from_pka,
    ksp_from_molar_solubility,
    molar_solubility_from_ksp,
    molar_solubility_with_common_ions,
    ph_from_hydrogen,
    precipitation_threshold_concentration,
    reaction_quotient,
    solve_equilibrium,
    strong_acid_ph,
    strong_base_ph,
    weak_acid_hydrogen_concentration,
    weak_acid_ph,
    weak_base_hydroxide_concentration,
    weak_base_ph,
    will_precipitate,
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

    def test_delta_g_and_equilibrium_constant_round_trip(self):
        equilibrium_constant = equilibrium_constant_from_delta_g_standard(-59_000.0, 298.15)

        self.assertGreater(equilibrium_constant, 1.0e10)
        self.assertAlmostEqual(
            delta_g_standard_from_equilibrium_constant(equilibrium_constant, 298.15),
            -59_000.0,
            places=8,
        )

    def test_henrys_law_helpers(self):
        concentration = henry_law_concentration(3.0e-8, 0.20)

        self.assertAlmostEqual(concentration, 6.0e-9)
        self.assertAlmostEqual(henry_law_pressure(concentration, 3.0e-8), 0.20)


if __name__ == "__main__":
    unittest.main()
