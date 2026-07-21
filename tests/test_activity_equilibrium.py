import unittest

from activity_equilibrium import (
    AcidBaseComponent,
    concentration_equilibrium_constant_with_davies,
    davies_activity_coefficient,
    davies_concentration_pka,
    davies_log10_activity_coefficient,
    fit_pkas_from_mean_charge,
    ionic_strength_from_concentrations,
    polyprotic_activity_average_charge,
    polyprotic_activity_species_concentrations,
    solve_acid_base_mixture_activity,
)


class DaviesActivityTests(unittest.TestCase):
    def test_davies_activity_coefficients(self):
        self.assertAlmostEqual(davies_log10_activity_coefficient(1, 0.100), -0.10701881433618944)
        self.assertAlmostEqual(davies_activity_coefficient(1, 0.100), 0.7815939439468335)
        self.assertAlmostEqual(davies_activity_coefficient(2, 0.100), 0.37318548420827014)
        self.assertAlmostEqual(davies_activity_coefficient(0, 0.100), 1.0)

    def test_davies_concentration_pka_for_glycine(self):
        pka1 = davies_concentration_pka(2.350, acid_charge=1, conjugate_base_charge=0, ionic_strength=0.100)
        pka2 = davies_concentration_pka(9.778, acid_charge=0, conjugate_base_charge=-1, ionic_strength=0.100)

        self.assertAlmostEqual(pka1, 2.350)
        self.assertAlmostEqual(pka2, 9.563962371327621)

    def test_concentration_equilibrium_constant_with_davies(self):
        acid_k = concentration_equilibrium_constant_with_davies(
            1.0e-5,
            {"H+": 1, "A-": 1, "HA": -1},
            {"H+": 1, "A-": -1, "HA": 0},
            0.100,
        )
        fescn_k = concentration_equilibrium_constant_with_davies(
            1.40e2,
            {"FeSCN2+": 1, "Fe3+": -1, "SCN-": -1},
            {"FeSCN2+": 2, "Fe3+": 3, "SCN-": -1},
            0.005,
        )

        self.assertAlmostEqual(acid_k, 1.636958346625927e-5)
        self.assertAlmostEqual(fescn_k, 88.92424374612608)


class ActivityAcidBaseSolverTests(unittest.TestCase):
    def test_activity_corrected_species_concentrations_at_fixed_ph(self):
        species = polyprotic_activity_species_concentrations(
            0.0300,
            9.00,
            (10.0**-2.350, 10.0**-9.778),
            (1.0, 1.0, 1.0),
            ("H2G+", "HG", "G-"),
        )

        self.assertAlmostEqual(species["H2G+"], 5.756423948724854e-9)
        self.assertAlmostEqual(species["HG"], 0.025713001273602487)
        self.assertAlmostEqual(species["G-"], 0.004286992969973563)

    def test_solve_monoprotic_acid_with_added_strong_base(self):
        component = AcidBaseComponent.from_pkas(0.0100, (5.00,), (0, -1), ("HA", "A-"))
        ideal = solve_acid_base_mixture_activity(
            (component,),
            {"K+": 0.00050},
            {"K+": 1},
            ionic_strength=0.0,
        )
        corrected = solve_acid_base_mixture_activity(
            (component,),
            {"K+": 0.00050},
            {"K+": 1},
        )

        self.assertAlmostEqual(ideal.ph, 3.8385383731219918)
        self.assertAlmostEqual(corrected.ph, 3.8309105311054736)
        self.assertAlmostEqual(corrected.ionic_strength, 6.519366853275051e-4)
        self.assertAlmostEqual(corrected.species_concentrations["A-"], 6.519366214542348e-4)

    def test_solve_glycine_hydrochloride_at_fixed_ionic_strength(self):
        glycine = AcidBaseComponent.from_pkas(
            0.0300,
            (2.350, 9.778),
            (1, 0, -1),
            ("H2G+", "HG", "G-"),
        )
        result = solve_acid_base_mixture_activity(
            (glycine,),
            {"Cl-": 0.0150},
            {"Cl-": -1},
            ionic_strength=0.100,
        )

        self.assertAlmostEqual(result.ph, 2.6321670033576083)
        self.assertAlmostEqual(result.species_concentrations["H2G+"], 0.012015637523774601)
        self.assertAlmostEqual(result.species_concentrations["HG"], 0.017984360831547617)
        self.assertAlmostEqual(result.charge_balance_residual, 0.0, places=11)

    def test_ionic_strength_from_named_concentrations(self):
        strength = ionic_strength_from_concentrations(
            {"K+": 0.010, "H2PO4-": 0.010, "HPO4^2-": 0.005, "Na+": 0.010},
            {"K+": 1, "H2PO4-": -1, "HPO4^2-": -2, "Na+": 1},
        )

        self.assertAlmostEqual(strength, 0.025)


class PkaFitTests(unittest.TestCase):
    def test_fit_pkas_from_mean_charge_data(self):
        ph_values = (2.0, 3.0, 6.0, 9.0, 10.0)
        species_charges = (1, 0, -1)
        constants = (10.0**-2.350, 10.0**-9.778)
        measured_charges = tuple(
            polyprotic_activity_average_charge(ph, constants, species_charges, (1.0, 1.0, 1.0))
            for ph in ph_values
        )

        fit = fit_pkas_from_mean_charge(
            ph_values,
            measured_charges,
            species_charges,
            initial_pkas=(2.0, 10.0),
            initial_step=1.0,
        )

        self.assertAlmostEqual(fit.pkas[0], 2.350, places=5)
        self.assertAlmostEqual(fit.pkas[1], 9.778, places=5)
        self.assertLess(fit.sum_squares, 1.0e-12)


if __name__ == "__main__":
    unittest.main()
