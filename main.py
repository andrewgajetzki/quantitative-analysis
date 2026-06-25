"""Examples for exploring quantitative-analysis calculations.

Keep this file as a scratchpad for textbook problems: import the helpers you
need, name the example after the concept, and print the result with units.
"""

from formula import Formula
from solutions import Solution, dilution_volume
from stoichiometry import Reaction


def demonstrate_solution_concepts() -> None:
    """Run representative calculations from the Chemical Measurements chapter."""
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


if __name__ == "__main__":
    demonstrate_solution_concepts()
    demonstrate_stoichiometry()
