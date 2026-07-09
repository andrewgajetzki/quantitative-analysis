"""Examples for exploring quantitative-analysis calculations.

Keep this file as a scratchpad: import the helpers you need, name each example
after the concept, and print the result with units.
"""

from formula import Formula
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
    demonstrate_measurement_corrections()
    demonstrate_preparation_and_dilution()
    demonstrate_experimental_error()
    demonstrate_statistics_and_calibration()
    demonstrate_quality_assurance()
