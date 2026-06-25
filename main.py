"""Examples for exploring quantitative-analysis calculations.

Keep this file as a scratchpad: import the helpers you need, name each example
after the concept, and print the result with units.
"""

from formula import Formula
from measurements import (
    BuretCalibration,
    aqueous_molarity_at_temperature,
    apparent_mass_from_true,
    true_mass_from_apparent,
    water_density_g_per_ml,
)
from solutions import Solution, dilution_volume, serial_dilution, solute_mass_for_molarity
from stoichiometry import Reaction


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


if __name__ == "__main__":
    demonstrate_solution_concepts()
    demonstrate_stoichiometry()
    demonstrate_measurement_corrections()
    demonstrate_preparation_and_dilution()
