import math
import unittest

from equilibrium import (
    activity_coefficient_from_ph,
    activity_product,
    activity_reaction_quotient,
    buffer_ph,
    buffer_capacity,
    buffer_ph_after_strong_acid,
    buffer_ph_after_strong_base,
    buffer_ph_from_amounts,
    charge_balance,
    classify_ph,
    combine_equilibrium_constants,
    complex_concentration_from_free,
    complex_distribution_fractions,
    complex_species_concentrations,
    conjugate_acid_ph,
    conjugate_base_ph,
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
    concentration_equilibrium_constant,
    debye_huckel_activity_coefficient,
    debye_huckel_activity_coefficients,
    debye_huckel_a_parameter,
    debye_huckel_b_parameter_pm,
    debye_huckel_log10_activity_coefficient,
    formation_constant,
    free_metal_from_total_metal,
    free_metal_concentration,
    henry_law_concentration,
    henry_law_pressure,
    hydrogen_from_ph,
    hydrogen_from_strong_acid,
    interpolate_activity_coefficient,
    ion_product,
    ionic_strength,
    ka_from_pka,
    ksp_from_molar_solubility,
    ksp_from_molar_solubility_with_activity,
    mass_balance,
    maximum_buffer_capacity,
    monoprotic_acid_mixture_ph,
    molar_solubility_from_ksp,
    molar_solubility_from_ksp_with_activity,
    molar_solubility_with_common_ions,
    molar_solubility_with_common_ions_and_activity,
    neutral_ph,
    partial_pressures_from_moles,
    ph_from_hydrogen,
    ph_from_hydrogen_concentration_activity,
    ph_from_hydroxide_concentration_activity,
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
    strong_acid_base_mixture_ph,
    strong_base_ph,
    strong_reagent_for_target_buffer_ph,
    target_buffer_base_acid_ratio,
    temperature_change_shift,
    weak_acid_hydrogen_concentration,
    weak_acid_fraction_dissociated,
    weak_acid_ka_from_fraction_dissociated,
    weak_acid_ph,
    weak_acid_pka_from_ph,
    weak_acid_strong_base_mixture_ph,
    weak_acid_ph_with_activity,
    weak_base_hydroxide_concentration,
    weak_base_fraction_protonated,
    weak_base_kb_from_ph,
    weak_base_ph,
    weak_base_strong_acid_mixture_ph,
    weak_base_ph_with_activity,
    will_precipitate,
    will_precipitate_with_activity,
    thermodynamic_equilibrium_constant,
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


class ActivityCoefficientTests(unittest.TestCase):
    def test_ionic_strength_and_debye_huckel_coefficients(self):
        concentrations = {"Mg2+": 0.025, "Cl-": 0.050}
        charges = {"Mg2+": 2, "Cl-": -1}
        sizes = {"Mg2+": 800, "Cl-": 300}

        self.assertAlmostEqual(ionic_strength(concentrations, charges), 0.075)
        self.assertAlmostEqual(debye_huckel_a_parameter(), 0.50930073492646)
        self.assertAlmostEqual(debye_huckel_b_parameter_pm(), 0.0032863875776494573)
        self.assertAlmostEqual(
            debye_huckel_log10_activity_coefficient(2, 0.075, 800),
            -0.32436470363290587,
        )

        coefficients = debye_huckel_activity_coefficients(concentrations, charges, sizes)
        self.assertAlmostEqual(coefficients["Mg2+"], 0.47384390243667895)
        self.assertAlmostEqual(coefficients["Cl-"], 0.7765606330939845)
        self.assertAlmostEqual(debye_huckel_activity_coefficient(0, 0.075), 1.0)

    def test_activity_quotients_and_constant_conversion(self):
        stoichiometry = {"Ca2+": 1, "CO3^2-": 1}
        concentrations = {"Ca2+": 1.0e-4, "CO3^2-": 1.0e-4}
        coefficients = {"Ca2+": 0.50, "CO3^2-": 0.40}

        self.assertAlmostEqual(activity_product(concentrations, stoichiometry, coefficients), 2.0e-9)
        self.assertAlmostEqual(activity_reaction_quotient(concentrations, stoichiometry, coefficients), 2.0e-9)
        self.assertAlmostEqual(
            concentration_equilibrium_constant(4.5e-9, stoichiometry, coefficients),
            2.25e-8,
        )
        self.assertAlmostEqual(
            thermodynamic_equilibrium_constant(2.25e-8, stoichiometry, coefficients),
            4.5e-9,
        )

    def test_interpolate_activity_coefficient(self):
        table = {0.01: 0.90, 0.05: 0.80}

        self.assertAlmostEqual(interpolate_activity_coefficient(0.03, table), 0.85)
        self.assertAlmostEqual(interpolate_activity_coefficient(0.001, table), 0.90)
        self.assertAlmostEqual(interpolate_activity_coefficient(0.10, table), 0.80)

    def test_charge_and_mass_balance_helpers(self):
        charge_result = charge_balance({"Mg2+": 0.025, "Cl-": 0.050}, {"Mg2+": 2, "Cl-": -1})
        mass_result = mass_balance(
            0.025,
            {"Mg2+": 0.020, "MgCl+": 0.005},
            {"Mg2+": 1, "MgCl+": 1},
        )

        self.assertTrue(charge_result.balanced)
        self.assertAlmostEqual(charge_result.positive_charge, charge_result.negative_charge)
        self.assertTrue(mass_result.balanced)
        self.assertAlmostEqual(mass_result.accounted_concentration, 0.025)


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
        self.assertAlmostEqual(ph_from_hydrogen_concentration_activity(0.0100, 0.83), 2.0809219076239263)
        self.assertAlmostEqual(ph_from_hydroxide_concentration_activity(0.0100, 0.76), 11.880813592280791)

    def test_dilute_strong_acid_and_activity_from_ph(self):
        self.assertAlmostEqual(hydrogen_from_strong_acid(1.0e-8, acidic_protons=2), 1.1050e-7, places=11)
        self.assertAlmostEqual(strong_acid_ph(1.0e-8, acidic_protons=2), 6.9566, places=4)
        self.assertAlmostEqual(activity_coefficient_from_ph(0.100, 1.092), 0.8091, places=4)

    def test_weak_acid_helpers(self):
        hydrogen = weak_acid_hydrogen_concentration(0.100, 1.8e-5)

        self.assertAlmostEqual(hydrogen, 0.001332670973077975)
        self.assertAlmostEqual(weak_acid_ph(0.100, 1.8e-5), 2.875, places=3)
        self.assertAlmostEqual(weak_acid_fraction_dissociated(0.100, 1.0e-5), 0.0099501, places=7)
        self.assertAlmostEqual(
            weak_acid_ka_from_fraction_dissociated(0.0450, 0.0060),
            1.6298e-6,
            places=10,
        )
        self.assertAlmostEqual(weak_acid_pka_from_ph(0.0450, 2.78), 4.1969, places=4)

    def test_weak_base_helpers(self):
        hydroxide = weak_base_hydroxide_concentration(0.100, 1.8e-5)

        self.assertAlmostEqual(hydroxide, 0.001332670973077975)
        self.assertAlmostEqual(weak_base_ph(0.100, 1.8e-5), 11.125, places=3)
        self.assertAlmostEqual(weak_base_fraction_protonated(0.100, 1.0e-5), 0.0099501, places=7)
        self.assertAlmostEqual(weak_base_kb_from_ph(0.100, 11.0), 1.0101e-5, places=9)

    def test_conjugate_salt_ph_helpers(self):
        self.assertAlmostEqual(conjugate_acid_ph(0.100, 1.0e-4), 5.5000, places=4)
        self.assertAlmostEqual(conjugate_base_ph(0.100, 1.75e-5), 8.8785, places=4)

    def test_weak_acid_and_base_activity_corrections(self):
        self.assertAlmostEqual(weak_acid_ph_with_activity(0.100, 1.8e-5, 0.83, 0.76), 2.8568995838485494)
        self.assertAlmostEqual(weak_base_ph_with_activity(0.100, 1.8e-5, 0.83, 0.76), 11.104835916056167)

    def test_conjugate_constants_and_buffer_ph(self):
        ka = ka_from_pka(4.76)

        self.assertAlmostEqual(ka * conjugate_kb(ka), 1.0e-14)
        self.assertAlmostEqual(buffer_ph(0.20, 0.20, 4.76), 4.76)
        self.assertAlmostEqual(buffer_ph(0.10, 1.0, 4.76), 5.76)
        self.assertAlmostEqual(buffer_ph(0.20, 0.20, 4.76, conjugate_base_activity_coefficient=0.80), 4.6631, places=4)

    def test_buffer_stoichiometry_and_capacity_helpers(self):
        self.assertAlmostEqual(buffer_ph_from_amounts(0.020, 0.030, 4.76), 4.9361, places=4)
        self.assertAlmostEqual(buffer_ph_after_strong_acid(0.050, 0.050, 4.76, 0.010), 4.5839, places=4)
        self.assertAlmostEqual(buffer_ph_after_strong_base(0.050, 0.050, 4.76, 0.010), 4.9361, places=4)
        self.assertAlmostEqual(weak_acid_strong_base_mixture_ph(0.050, 0.030, 1.0, 4.76), 4.9361, places=4)
        self.assertAlmostEqual(weak_base_strong_acid_mixture_ph(0.020, 0.010, 1.0, 1.0e-5), 9.0)
        self.assertAlmostEqual(strong_acid_base_mixture_ph(0.020, 0.020, 1.0), 7.0)

        target_ratio = target_buffer_base_acid_ratio(7.45, 7.55)
        self.assertAlmostEqual(target_ratio, 0.79433, places=5)
        adjustment = strong_reagent_for_target_buffer_ph(0.0125, 0.0, 7.55, 7.45)
        self.assertEqual(adjustment.reagent, "strong_base")
        self.assertAlmostEqual(adjustment.amount, 0.0055336, places=7)
        self.assertAlmostEqual(adjustment.ph, 7.45)

        ka = ka_from_pka(4.76)
        self.assertAlmostEqual(buffer_capacity(0.100, ka, 4.76, include_water=False), 0.057565, places=6)
        self.assertAlmostEqual(maximum_buffer_capacity(0.100), 0.057565, places=6)

    def test_exact_monoprotic_acid_charge_balance_buffer(self):
        ka = ka_from_pka(4.76)

        exact_buffer_ph = monoprotic_acid_mixture_ph(0.100, 0.100, 1.0, ka)
        exact_after_base = monoprotic_acid_mixture_ph(0.050, 0.050, 1.0, ka, strong_base_amount=0.010)

        self.assertAlmostEqual(exact_buffer_ph, 4.7602, places=4)
        self.assertAlmostEqual(exact_after_base, 4.9363, places=4)

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

    def test_activity_corrected_solubility(self):
        coefficients = {"Ag+": 0.75, "Cl-": 0.76}

        solubility = molar_solubility_from_ksp_with_activity(
            1.8e-10,
            {"Ag+": 1, "Cl-": 1},
            coefficients,
        )
        common_ion_solubility = molar_solubility_with_common_ions_and_activity(
            1.8e-10,
            {"Ag+": 1, "Cl-": 1},
            {"Cl-": 0.0100},
            coefficients,
        )

        self.assertAlmostEqual(solubility, 1.777046633277277e-5)
        self.assertAlmostEqual(
            ksp_from_molar_solubility_with_activity(solubility, {"Ag+": 1, "Cl-": 1}, coefficients),
            1.8e-10,
        )
        self.assertAlmostEqual(common_ion_solubility, 3.157884764606278e-8)
        self.assertTrue(
            will_precipitate_with_activity(
                {"Ag+": 1.0e-4, "Cl-": 1.0e-4},
                1.8e-10,
                {"Ag+": 1, "Cl-": 1},
                coefficients,
            ).precipitates
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
