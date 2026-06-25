import unittest

from formula import Formula
from solutions import Solution, dilution_volume, molarity_from_mass
from stoichiometry import Reaction
from units import mass, volume


class FormulaTests(unittest.TestCase):
    def test_formula_mass_with_parentheses(self):
        self.assertAlmostEqual(Formula("Ca(NO3)2").mass, 164.088, places=3)

    def test_hydrate_formula_mass(self):
        self.assertAlmostEqual(Formula("CuSO4.5H2O").mass, 249.685, places=3)

    def test_charge_annotation_does_not_change_mass(self):
        self.assertAlmostEqual(Formula("Ba2+").mass, Formula("Ba").mass)


class SolutionTests(unittest.TestCase):
    def test_molarity_from_mass_and_volume(self):
        self.assertAlmostEqual(molarity_from_mass("NaOH", 4.00, 1.00), 0.100, places=3)

    def test_weight_percent_solution_molarity(self):
        solution = Solution.from_weight_percent("HBr", 48.0, 1.50)
        self.assertAlmostEqual(solution.molarity, 8.90, places=2)

    def test_dilution_volume(self):
        self.assertAlmostEqual(dilution_volume(18.0, 1.00, 1000), 55.6, places=1)


class StoichiometryTests(unittest.TestCase):
    def test_limiting_reagent(self):
        reaction = Reaction.from_equation("2 H2 + O2 -> 2 H2O")
        result = reaction.limiting_reagent({"H2": 3.0, "O2": 1.0})
        self.assertEqual(result.limiting_species, "O2")
        self.assertAlmostEqual(result.product_moles("H2O"), 2.0)

    def test_reaction_parser_keeps_charged_species_labels(self):
        reaction = Reaction.from_equation("Ba2+ + SO4 -> BaSO4")
        self.assertIn("Ba2+", reaction.reactants)


class UnitTests(unittest.TestCase):
    def test_mass_conversion(self):
        self.assertAlmostEqual(mass.convert(1.0, "lb", "g"), 453.59237)

    def test_volume_conversion(self):
        self.assertAlmostEqual(volume.convert(1.0, "mL", "L"), 0.001)


if __name__ == "__main__":
    unittest.main()
