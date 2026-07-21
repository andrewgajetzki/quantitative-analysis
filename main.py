"""Examples for exploring quantitative-analysis calculations.

Keep this file as a scratchpad: import the helpers you need, name each example
after the concept, and print the result with units.
"""

from formula import Formula
from equilibrium import (
    classify_ph,
    complex_concentration_from_free,
    delta_gas_moles,
    debye_huckel_activity_coefficients,
    delta_g_standard_from_equilibrium_constant,
    equilibrium_constant_from_delta_g_standard,
    equilibrium_direction,
    free_metal_from_total_metal,
    henry_law_concentration,
    ionic_strength,
    kp_from_kc,
    molar_solubility_from_ksp,
    molar_solubility_from_ksp_with_activity,
    neutral_ph,
    ph_from_hydrogen_concentration_activity,
    pressure_change_shift,
    reaction_quotient,
    salt_solution_character,
    solve_equilibrium,
    successive_to_cumulative_constants,
    weak_acid_ph,
    will_precipitate,
)
from edta import (
    back_edta_titration,
    edta_conditional_formation_constant,
    edta_titration_state,
    edta_y4_fraction_from_ph,
    free_metal_fraction_with_complexing_agent,
    metal_buffer_free_metal_concentration,
    metal_indicator_color,
    p_metal_from_concentration,
)
from measurements import (
    BuretCalibration,
    aqueous_molarity_at_temperature,
    apparent_mass_from_true,
    compare_means_from_stats,
    concentration_from_internal_standard,
    confidence_interval_mean,
    control_chart_status,
    detection_limit_concentration,
    grubbs_test,
    internal_standard_response_factor,
    linear_least_squares,
    normal_probability_between,
    single_standard_addition_concentration,
    standard_addition_from_added_concentrations,
    true_mass_from_apparent,
    water_density_g_per_ml,
)
from solutions import Solution, dilution_volume, serial_dilution, solute_mass_for_molarity
from stoichiometry import Reaction
from uncertainty import (
    Measurement,
    antilog10,
    format_measurement,
    format_significant_figures,
    significant_figures,
)


def demonstrate_solution_concepts() -> None:
    """Run representative solution-concentration calculations."""
    hbr = Solution.from_weight_percent(
        solute_formula="HBr",
        weight_percent=48.0,
        density_g_per_ml=1.50,
    )
    print("48.0 wt% HBr, density 1.50 g/mL")
    print(f"  formal concentration: {hbr.molarity:.3f} mol/L")
    print(f"  mass of solution with 36.0 g HBr: {hbr.solution_mass_for_solute(36.0):.1f} g")

    stock_volume_ml = dilution_volume(
        concentrated_molarity=18.0,
        dilute_molarity=1.00,
        dilute_volume_ml=1000,
    )
    print("\nDilution setup")
    print(f"  stock volume required: {stock_volume_ml:.1f} mL")


def demonstrate_stoichiometry() -> None:
    """Show limiting-reagent and yield calculations with reusable objects."""
    reaction = Reaction.from_equation("Ba2+ + SO4 -> BaSO4")
    result = reaction.limiting_reagent({"Ba2+": 23.2 / Formula("Ba(NO3)2").mass, "SO4": 0.0120})

    print("\nPrecipitation stoichiometry")
    print(f"  limiting species: {result.limiting_species}")
    print(f"  BaSO4 produced: {result.product_moles('BaSO4'):.4f} mol")


def demonstrate_equilibrium() -> None:
    """Show equilibrium, acid-base, solubility, and thermodynamic calculations."""
    hi_equilibrium = {"HI": 2, "H2": -1, "I2": -1}
    q_value = reaction_quotient({"HI": 1.0, "H2": 0.10, "I2": 0.10}, hi_equilibrium)
    print("\nChemical equilibrium")
    print(f"  Q for H2 + I2 <=> 2 HI: {q_value:.1f}")
    print(f"  shift when K = 50.0: {equilibrium_direction(q_value, 50.0)}")

    ice_result = solve_equilibrium({"A": 1.0, "B": 0.0}, {"B": 1, "A": -1}, 4.0)
    print(f"  ICE result for A <=> B, K = 4.0: [A] = {ice_result.concentrations['A']:.2f} M")
    print(f"  weak acid pH for 0.100 M HA, Ka = 1.8e-5: {weak_acid_ph(0.100, 1.8e-5):.3f}")
    print(f"  neutral pH when Kw = 5.5e-13: {neutral_ph(5.5e-13):.3f}")
    print(f"  pH 7.00 at that Kw is {classify_ph(7.00, 5.5e-13)}")
    print(f"  NH4Cl salt character: {salt_solution_character(cation_ka=5.6e-10).character}")

    mgcl2_concentrations = {"Mg2+": 0.025, "Cl-": 0.050}
    mgcl2_charges = {"Mg2+": 2, "Cl-": -1}
    mgcl2_sizes = {"Mg2+": 800, "Cl-": 300}
    mgcl2_strength = ionic_strength(mgcl2_concentrations, mgcl2_charges)
    mgcl2_gammas = debye_huckel_activity_coefficients(
        mgcl2_concentrations,
        mgcl2_charges,
        mgcl2_sizes,
    )
    print(f"  MgCl2 ionic strength: {mgcl2_strength:.3f} M")
    print(f"  gamma(Mg2+): {mgcl2_gammas['Mg2+']:.3f}, gamma(Cl-): {mgcl2_gammas['Cl-']:.3f}")
    print(f"  pH from 0.0100 M H+ with gamma=0.83: {ph_from_hydrogen_concentration_activity(0.0100, 0.83):.3f}")

    agcl_solubility = molar_solubility_from_ksp(1.8e-10, {"Ag+": 1, "Cl-": 1})
    agcl_activity_solubility = molar_solubility_from_ksp_with_activity(
        1.8e-10,
        {"Ag+": 1, "Cl-": 1},
        {"Ag+": 0.75, "Cl-": 0.76},
    )
    agcl_precipitate = will_precipitate({"Ag+": 1.0e-4, "Cl-": 1.0e-4}, 1.8e-10, {"Ag+": 1, "Cl-": 1})
    print(f"  AgCl molar solubility: {agcl_solubility:.2e} M")
    print(f"  AgCl activity-corrected solubility: {agcl_activity_solubility:.2e} M")
    print(f"  AgCl precipitates from 1.0e-4 M ions: {agcl_precipitate.precipitates}")

    complex_concentration = complex_concentration_from_free(1.0e-9, 0.10, 1.0e9, ligand_coefficient=2)
    cumulative_betas = successive_to_cumulative_constants((1.0e2, 1.0e3, 1.0e4))
    free_metal = free_metal_from_total_metal(0.0100, 0.10, cumulative_betas)
    equilibrium_constant = equilibrium_constant_from_delta_g_standard(-59_000.0, 298.15)
    delta_g_standard = delta_g_standard_from_equilibrium_constant(equilibrium_constant, 298.15)
    gas_stoichiometry = {"NH3": 2, "N2": -1, "H2": -3}
    kp = kp_from_kc(1.0e-3, delta_gas_moles(gas_stoichiometry), 500.0)
    dissolved_gas = henry_law_concentration(3.0e-8, 0.20)
    print(f"  complex concentration from Kf: {complex_concentration:.2e} M")
    print(f"  free metal from cumulative beta values: {free_metal:.2e} M")
    print(f"  K from Delta G standard = -59.0 kJ/mol: {equilibrium_constant:.2e}")
    print(f"  round-trip Delta G standard: {delta_g_standard / 1000.0:.1f} kJ/mol")
    print(f"  Kp from Kc for N2 + 3 H2 <=> 2 NH3: {kp:.2e}")
    print(f"  pressure increase shift for 2 F2(g) -> CF4(g): {pressure_change_shift({'CF4': 1, 'F2': -2}, True)}")
    print(f"  Henry's law dissolved gas: {dissolved_gas:.2e} M")


def demonstrate_edta_complexometry() -> None:
    """Show EDTA conditional constants, titration curves, and back titration."""
    alpha_y4 = edta_y4_fraction_from_ph(10.00)
    conditional_kf = edta_conditional_formation_constant(5.0e10, 10.00)
    print("\nEDTA complexometric titrations")
    print(f"  alpha_Y4- at pH 10.00: {alpha_y4:.3f}")
    print(f"  conditional Kf for CaY2- at pH 10.00: {conditional_kf:.2e}")

    metal_alpha = free_metal_fraction_with_complexing_agent(1.00, (1.0e4, 1.0e8, 1.0e12, 1.0e13))
    print(f"  free-metal fraction with 1.00 M auxiliary ligand: {metal_alpha:.2e}")

    free_metal = metal_buffer_free_metal_concentration(0.0100, 0.0500, 1.0e8)
    print(f"  EDTA metal buffer pM: {p_metal_from_concentration(free_metal):.3f}")

    for volume_ml in (50.0, 100.0, 150.0):
        state = edta_titration_state(0.0010, 100.0, 0.0010, volume_ml, 1.0e10)
        print(f"  pM after {volume_ml:.1f} mL EDTA: {state.p_metal:.3f} ({state.stage})")

    before_color = metal_indicator_color(1.0e-3, 1.0e6, "blue", "red")
    after_color = metal_indicator_color(1.0e-9, 1.0e6, "blue", "red")
    print(f"  metal indicator color before/after equivalence: {before_color} -> {after_color}")

    back_titration = back_edta_titration(0.0500, 25.00, 0.02127, 25.63, sample_volume_ml=50.00)
    print(f"  back-titration analyte concentration: {back_titration.analyte_molarity:.5f} M")


def demonstrate_measurement_corrections() -> None:
    """Show balance, temperature, and glassware calibration calculations."""
    water_mass_15 = true_mass_from_apparent(5.3974, water_density_g_per_ml(15.0))
    water_mass_25 = true_mass_from_apparent(5.3974, water_density_g_per_ml(25.0))
    print("\nBuoyancy correction")
    print(f"  true water mass at 15 C: {water_mass_15:.4f} g")
    print(f"  true water mass at 25 C: {water_mass_25:.4f} g")

    apparent_cscl = apparent_mass_from_true(1.6780, object_density_g_per_ml=3.988)
    print(f"  apparent CsCl mass for 1.6780 g true mass: {apparent_cscl:.4f} g")

    corrected_molarity = aqueous_molarity_at_temperature(0.05138, from_temp_c=24.0, to_temp_c=16.0)
    print("\nTemperature correction")
    print(f"  0.05138 M at 24 C becomes {corrected_molarity:.5f} M at 16 C")

    calibration = BuretCalibration.from_water_masses(
        readings_ml=[(0.03, 10.01), (10.01, 19.90), (19.90, 30.06)],
        apparent_water_masses_g=[9.984, 9.835, 10.071],
        temp_c=24.0,
    )
    print("\nBuret calibration")
    for interval in calibration.intervals:
        print(
            f"  {interval.initial_reading_ml:.2f}-{interval.final_reading_ml:.2f} mL: "
            f"actual {interval.actual_volume_ml:.3f} mL, correction {interval.correction_ml:+.3f} mL"
        )


def demonstrate_preparation_and_dilution() -> None:
    """Show standard-solution preparation and serial dilution calculations."""
    grams_k2so4 = solute_mass_for_molarity("K2SO4", molarity=0.1500, solution_volume_l=0.2500)
    diluted = serial_dilution(
        initial_concentration=0.001000,
        steps=[(5.00, 100.0), (10.00, 250.0)],
    )

    print("\nPreparation and dilution")
    print(f"  K2SO4 for 250.0 mL of 0.1500 M: {grams_k2so4:.4f} g")
    print(f"  serial dilution result: {diluted:.3e} M")


def demonstrate_experimental_error() -> None:
    """Show significant-figure and uncertainty-propagation calculations."""
    print("\nSignificant figures")
    print(f"  figures in 1.40 x 10^4: {significant_figures('1.40 x 10^4')}")
    print(f"  0.216500 rounded to 3 figures: {format_significant_figures('0.216500', 3)}")

    precipitate = Measurement(12.5296, 0.0003) - Measurement(12.4372, 0.0003)
    print("\nUncertainty propagation")
    print(f"  precipitate mass: {format_measurement(precipitate)} g")

    hydrogen_ion = antilog10(-Measurement(4.44, 0.04))
    print(f"  [H+] from pH 4.44 +/- 0.04: {format_measurement(hydrogen_ion, 2)} M")


def demonstrate_statistics_and_calibration() -> None:
    """Show statistics and calibration calculations."""
    replicate_values = [116.0, 97.9, 114.2, 106.8, 108.3]
    interval = confidence_interval_mean(replicate_values, confidence=0.90)
    outlier = grubbs_test(replicate_values, confidence=0.90)
    print("\nMeasurement statistics")
    print(f"  90% confidence interval: {interval.center:.2f} +/- {interval.half_width:.2f}")
    print(f"  Grubbs candidate: {outlier.outlier_value:.1f}, reject: {outlier.significant}")
    print(f"  Gaussian fraction within +/-1 sigma: {normal_probability_between(-1, 1):.3f}")

    comparison = compare_means_from_stats(1.392, 0.025, 4, 1.346, 0.039, 4)
    print(f"  two-method t statistic: {comparison.statistic:.2f}, significant: {comparison.significant}")

    fit = linear_least_squares(
        [0.00, 9.36, 18.72, 28.08, 37.44],
        [0.446, 0.676, 0.883, 1.086, 1.280],
    )
    prediction = fit.inverse_prediction(0.973, confidence=0.95)
    prediction_interval = prediction.confidence_interval
    print("\nCalibration curve")
    print(f"  slope: {fit.slope:.5f}, intercept: {fit.intercept:.5f}")
    print(f"  unknown x from response 0.973: {prediction.x_value:.2f} +/- {prediction_interval.half_width:.2f}")


def demonstrate_quality_assurance() -> None:
    """Show quality-assurance and standard-addition calculations."""
    blank_absorbances = [0.0002, 0.0002, 0.0005, 0.0001, 0.0008, 0.0001, 0.0007, 0.0001, 0.0001]
    blank_mean = sum(blank_absorbances) / len(blank_absorbances)
    detection_limit = detection_limit_concentration(
        slope=2.34e4,
        intercept=blank_mean,
        blank_values=blank_absorbances,
    )
    print("\nQuality assurance")
    print(f"  concentration detection limit: {detection_limit:.2e} M")

    standard_addition = standard_addition_from_added_concentrations(
        [0.0, 2.5, 5.0, 7.5, 10.0],
        [28.0, 34.3, 42.8, 51.5, 58.6],
    )
    print(f"  standard-addition unknown: {standard_addition.unknown_concentration:.2f}")

    nickel = single_standard_addition_concentration(2.36, 3.79, 25.00, 0.500, 0.0287)
    print(f"  one-point Ni standard addition: {nickel:.3e} M")

    factor = internal_standard_response_factor(10222, 8477, 3.47, 1.72)
    unknown_final = concentration_from_internal_standard(5428, 4431, 2.155, factor)
    print(f"  internal-standard unknown: {unknown_final:.2f}")

    control = control_chart_status(105.0, center=100.0, standard_deviation=2.0)
    print(f"  control-chart status: {control.status}")


if __name__ == "__main__":
    demonstrate_solution_concepts()
    demonstrate_stoichiometry()
    demonstrate_equilibrium()
    demonstrate_edta_complexometry()
    demonstrate_measurement_corrections()
    demonstrate_preparation_and_dilution()
    demonstrate_experimental_error()
    demonstrate_statistics_and_calibration()
    demonstrate_quality_assurance()
