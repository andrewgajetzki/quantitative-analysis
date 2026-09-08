import math
import unittest

from redox import (
    RedoxCouple,
    SATURATED_CALOMEL_ELECTRODE_V,
    analyte_molarity_from_redox_titration,
    balance_half_reaction,
    combine_half_reactions,
    indicator_cell_voltage,
    iodine_moles_from_thiosulfate,
    molarity_from_normality,
    normality_from_molarity,
    oxidation_number,
    redox_couple_potential,
    redox_equivalence_volume_ml,
    redox_gran_equivalence_volume,
    redox_indicator_is_suitable,
    redox_indicator_transition_range,
    redox_titration_state,
    species_charge,
    winkler_oxygen_mg_per_l,
    winkler_oxygen_moles_from_thiosulfate,
)


class RedoxFormulaTests(unittest.TestCase):
    def test_species_charge_parser_handles_monatomic_and_polyatomic_ions(self):
        self.assertEqual(species_charge("Fe2+"), 2)
        self.assertEqual(species_charge("Ce4+"), 4)
        self.assertEqual(species_charge("MnO4-"), -1)
        self.assertEqual(species_charge("Cr2O7^2-"), -2)
        self.assertEqual(species_charge("NH4+"), 1)
        self.assertEqual(species_charge("I3-"), -1)

    def test_oxidation_number_from_formula_and_charge(self):
        self.assertAlmostEqual(oxidation_number("MnO4-", "Mn"), 7.0)
        self.assertAlmostEqual(oxidation_number("Cr2O7^2-", "Cr"), 6.0)
        self.assertAlmostEqual(oxidation_number("Fe2O3", "Fe"), 3.0)


class RedoxBalancingTests(unittest.TestCase):
    def test_balances_common_acidic_half_reactions(self):
        permanganate = balance_half_reaction("MnO4-", "Mn2+", medium="acidic")
        dichromate = balance_half_reaction("Cr2O7^2-", "Cr3+", medium="acidic")

        self.assertEqual(permanganate.equation(), "MnO4- + 8 H+ + 5 e- -> Mn2+ + 4 H2O")
        self.assertEqual(dichromate.equation(), "Cr2O7^2- + 14 H+ + 6 e- -> 2 Cr3+ + 7 H2O")

    def test_balances_basic_half_reaction(self):
        permanganate_to_manganese_dioxide = balance_half_reaction(
            "MnO4-",
            "MnO2",
            medium="basic",
        )

        self.assertEqual(
            permanganate_to_manganese_dioxide.equation(),
            "MnO4- + 2 H2O + 3 e- -> MnO2 + 4 OH-",
        )

    def test_combines_oxidation_and_reduction_half_reactions(self):
        tin = balance_half_reaction("Sn2+", "Sn4+")
        cerium = balance_half_reaction("Ce4+", "Ce3+")
        reaction = combine_half_reactions(tin, cerium)

        self.assertEqual(tin.equation(), "Sn2+ -> Sn4+ + 2 e-")
        self.assertEqual(cerium.equation(), "Ce4+ + e- -> Ce3+")
        self.assertEqual(reaction.equation(), "Sn2+ + 2 Ce4+ -> Sn4+ + 2 Ce3+")


class RedoxStoichiometryTests(unittest.TestCase):
    def test_equivalents_normality_and_endpoint_molarity(self):
        self.assertAlmostEqual(normality_from_molarity(0.0200, 5), 0.1000)
        self.assertAlmostEqual(molarity_from_normality(0.1000, 5), 0.0200)

        equivalence_volume = redox_equivalence_volume_ml(
            analyte_molarity=0.00500,
            analyte_volume_ml=20.00,
            titrant_molarity=0.0200,
            analyte_electrons=2,
            titrant_electrons=1,
        )
        analyte_molarity = analyte_molarity_from_redox_titration(
            titrant_molarity=0.0200,
            titrant_volume_ml=10.00,
            sample_volume_ml=20.00,
            analyte_electrons=2,
            titrant_electrons=1,
        )

        self.assertAlmostEqual(equivalence_volume, 10.00)
        self.assertAlmostEqual(analyte_molarity, 0.00500)

    def test_iodometric_thiosulfate_and_winkler_stoichiometry(self):
        iodine_moles = iodine_moles_from_thiosulfate(0.0100, 25.00)
        oxygen_moles = winkler_oxygen_moles_from_thiosulfate(0.0100, 25.00)
        oxygen_mg_l = winkler_oxygen_mg_per_l(0.0100, 25.00, sample_volume_ml=300.0)

        self.assertAlmostEqual(iodine_moles, 1.25e-4)
        self.assertAlmostEqual(oxygen_moles, 6.25e-5)
        self.assertAlmostEqual(oxygen_mg_l, 6.6664, places=4)


class RedoxPotentialTests(unittest.TestCase):
    def test_redox_couple_potential_and_reference_voltage(self):
        potential = redox_couple_potential(
            standard_potential_v=0.767,
            electrons_transferred=1,
            oxidized_activity=0.0100,
            reduced_activity=0.00100,
        )

        self.assertAlmostEqual(potential, 0.82615935, places=8)
        self.assertAlmostEqual(
            indicator_cell_voltage(potential, SATURATED_CALOMEL_ELECTRODE_V),
            0.58515935,
            places=8,
        )

    def test_redox_titration_curve_before_at_and_after_equivalence(self):
        iron = RedoxCouple("Fe3+", "Fe2+", electrons=1, standard_potential_v=0.767)
        cerium = RedoxCouple("Ce4+", "Ce3+", electrons=1, standard_potential_v=1.440)

        before = redox_titration_state(0.100, 25.00, 0.100, 12.50, iron, cerium)
        equivalence = redox_titration_state(
            0.100,
            25.00,
            0.100,
            25.00,
            iron,
            cerium,
            reference_electrode_potential_v=SATURATED_CALOMEL_ELECTRODE_V,
        )
        after = redox_titration_state(0.100, 25.00, 0.100, 30.00, iron, cerium)

        self.assertEqual(before.stage, "before equivalence")
        self.assertAlmostEqual(before.indicator_potential_v, 0.767)
        self.assertEqual(equivalence.stage, "at equivalence")
        self.assertAlmostEqual(equivalence.indicator_potential_v, 1.1035)
        self.assertAlmostEqual(equivalence.cell_voltage_v, 0.8625)
        self.assertEqual(after.stage, "after equivalence")
        self.assertAlmostEqual(
            after.indicator_potential_v,
            1.440 + 0.0591593496911508 * math.log10(0.20),
            places=8,
        )

    def test_redox_titration_supports_oxidizing_analyte_with_reducing_titrant(self):
        cerium = RedoxCouple("Ce4+", "Ce3+", electrons=1, standard_potential_v=1.440)
        copper = RedoxCouple("Cu2+", "Cu+", electrons=1, standard_potential_v=0.153)

        before = redox_titration_state(
            0.0100,
            100.0,
            0.0400,
            12.50,
            cerium,
            copper,
            analyte_initial_form="oxidized",
        )
        after = redox_titration_state(
            0.0100,
            100.0,
            0.0400,
            30.00,
            cerium,
            copper,
            analyte_initial_form="oxidized",
        )

        self.assertAlmostEqual(before.equivalence_volume_ml, 25.00)
        self.assertAlmostEqual(before.indicator_potential_v, 1.440)
        self.assertAlmostEqual(
            after.indicator_potential_v,
            0.153 + 0.0591593496911508 * math.log10(5.0),
            places=8,
        )

    def test_redox_indicator_transition_and_suitability(self):
        transition = redox_indicator_transition_range(1.06, electrons_transferred=1)

        self.assertAlmostEqual(transition.lower_v, 1.00084065, places=8)
        self.assertAlmostEqual(transition.upper_v, 1.11915935, places=8)
        self.assertTrue(redox_indicator_is_suitable(1.1035, 1.06))
        self.assertFalse(redox_indicator_is_suitable(0.80, 1.06))

    def test_redox_gran_equivalence_volume_after_equivalence(self):
        formal_potential = 1.440
        slope = 0.0591593496911508
        volumes = (12.0, 15.0, 20.0)
        potentials = tuple(formal_potential + slope * math.log10((volume - 10.0) / 10.0) for volume in volumes)

        gran = redox_gran_equivalence_volume(
            volumes,
            potentials,
            formal_potential_v=formal_potential,
            electrons_transferred=1,
            branch="after",
        )

        self.assertAlmostEqual(gran.slope, 0.1000)
        self.assertAlmostEqual(gran.intercept, -1.0000)
        self.assertAlmostEqual(gran.equivalence_volume_ml, 10.00)


if __name__ == "__main__":
    unittest.main()
