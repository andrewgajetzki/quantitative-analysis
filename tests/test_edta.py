import unittest

from edta import (
    back_edta_titration,
    direct_edta_assay,
    displacement_edta_assay,
    edta_conditional_formation_constant,
    edta_equivalence_volume_ml,
    edta_species_concentrations_from_ph,
    edta_titration_curve,
    edta_titration_state,
    edta_y4_fraction_from_ph,
    free_metal_fraction_with_complexing_agent,
    metal_buffer_free_metal_concentration,
    metal_indicator_color,
    metal_indicator_complex_fraction,
    metal_molarity_from_edta_titration,
    p_metal_from_concentration,
)


class EDTAFractionTests(unittest.TestCase):
    def test_edta_y4_fraction_from_ph(self):
        self.assertAlmostEqual(edta_y4_fraction_from_ph(3.50), 3.2965522853505096e-10)
        self.assertAlmostEqual(edta_y4_fraction_from_ph(9.00), 0.052061390276683495)
        self.assertAlmostEqual(edta_y4_fraction_from_ph(10.00), 0.35480563065008114)
        self.assertAlmostEqual(edta_y4_fraction_from_ph(10.50), 0.634926339630923)

    def test_edta_species_concentrations_from_ph(self):
        species = edta_species_concentrations_from_ph(0.0010, 4.00)

        self.assertAlmostEqual(species["H2Y2-"], 9.4867020363152e-4)
        self.assertAlmostEqual(species["HY3-"], 6.564797809130118e-6)
        self.assertAlmostEqual(species["Y4-"], 3.6106387950215648e-12)

    def test_conditional_formation_constant_includes_ph_and_metal_side_reaction(self):
        metal_alpha = free_metal_fraction_with_complexing_agent(1.00, (1.0e4, 1.0e8, 1.0e12, 1.0e13))

        self.assertAlmostEqual(metal_alpha, 9.09082643876716e-14)
        self.assertAlmostEqual(
            edta_conditional_formation_constant(5.0e10, 10.00),
            1.7740281532504055e10,
        )
        self.assertAlmostEqual(
            edta_conditional_formation_constant(1.0e18, 9.00, free_metal_fraction=metal_alpha),
            4732.810631662499,
            places=3,
        )


class EDTATitrationTests(unittest.TestCase):
    def test_equivalence_volume_and_direct_endpoint_molarity(self):
        self.assertAlmostEqual(edta_equivalence_volume_ml(0.0010, 100.0, 0.0010), 100.0)
        self.assertAlmostEqual(metal_molarity_from_edta_titration(0.0100, 25.00, 50.00), 0.00500)

    def test_titration_state_before_at_and_after_equivalence(self):
        before = edta_titration_state(0.0010, 100.0, 0.0010, 50.0, 1.0e10)
        equivalence = edta_titration_state(0.0010, 100.0, 0.0010, 100.0, 1.0e10)
        after = edta_titration_state(0.0010, 100.0, 0.0010, 150.0, 1.0e10)

        self.assertEqual(before.stage, "before_equivalence")
        self.assertAlmostEqual(before.free_metal_concentration, 3.333334333332734e-4)
        self.assertAlmostEqual(before.p_metal, 3.4771211244314153)

        self.assertEqual(equivalence.stage, "at_equivalence")
        self.assertAlmostEqual(equivalence.free_metal_concentration, 2.2355680334014887e-7)
        self.assertAlmostEqual(equivalence.free_edta_concentration, equivalence.free_metal_concentration)
        self.assertAlmostEqual(equivalence.p_metal, 6.65061210902956)

        self.assertEqual(after.stage, "after_equivalence")
        self.assertAlmostEqual(after.free_metal_concentration, 1.9999970000075009e-10)
        self.assertAlmostEqual(after.free_edta_concentration, 2.0000019999969994e-4)
        self.assertAlmostEqual(after.p_metal, 9.698970655776602)

    def test_titration_curve_returns_states_for_each_volume(self):
        curve = edta_titration_curve(0.0010, 100.0, 0.0010, (0.0, 100.0, 150.0), 1.0e10)

        self.assertEqual(len(curve), 3)
        self.assertEqual([state.stage for state in curve], ["before_equivalence", "at_equivalence", "after_equivalence"])

    def test_metal_buffer_free_metal_and_p_metal(self):
        free_metal = metal_buffer_free_metal_concentration(0.0100, 0.0500, 1.0e8)

        self.assertAlmostEqual(free_metal, 2.0e-9)
        self.assertAlmostEqual(p_metal_from_concentration(free_metal), 8.698970004336019)


class EDTAIndicatorAndAssayTests(unittest.TestCase):
    def test_metal_indicator_fraction_and_color(self):
        self.assertAlmostEqual(metal_indicator_complex_fraction(1.0e-5, 1.0e6), 0.9090909090909091)
        self.assertEqual(metal_indicator_color(1.0e-3, 1.0e6, "blue", "red"), "red")
        self.assertEqual(metal_indicator_color(1.0e-9, 1.0e6, "blue", "red"), "blue")
        self.assertEqual(metal_indicator_color(1.0e-6, 1.0e6, "blue", "red"), "mixed")

    def test_direct_back_and_displacement_assays(self):
        direct = direct_edta_assay(0.0100, 25.00, sample_volume_ml=50.00)
        back = back_edta_titration(0.0500, 25.00, 0.02127, 25.63, sample_volume_ml=50.00)
        displacement = displacement_edta_assay(0.0100, 10.00, sample_volume_ml=25.00)

        self.assertAlmostEqual(direct.edta_moles, 2.500e-4)
        self.assertAlmostEqual(direct.analyte_molarity, 0.00500)

        self.assertAlmostEqual(back.edta_added_moles, 0.0012500000000000002)
        self.assertAlmostEqual(back.excess_edta_moles, 0.0005451501)
        self.assertAlmostEqual(back.edta_consumed_moles, 0.0007048499000000002)
        self.assertAlmostEqual(back.analyte_molarity, 0.014096998000000003)

        self.assertAlmostEqual(displacement.analyte_moles, 1.000e-4)
        self.assertAlmostEqual(displacement.analyte_molarity, 0.00400)


if __name__ == "__main__":
    unittest.main()
