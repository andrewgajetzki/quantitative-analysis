"""Solution concentration, dilution, and preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from formula import Formula


def molarity_from_mass(solute_formula: str, solute_grams: float, solution_volume_l: float) -> float:
    """Return molarity from solute mass and final solution volume."""
    return Formula(solute_formula).moles_from_grams(solute_grams) / solution_volume_l


def dilution_volume(concentrated_molarity: float, dilute_molarity: float, dilute_volume_ml: float) -> float:
    """Return milliliters of stock solution needed using M1 V1 = M2 V2."""
    if concentrated_molarity <= 0:
        raise ValueError("concentrated_molarity must be positive.")
    if dilute_molarity > concentrated_molarity:
        raise ValueError("dilute_molarity cannot exceed concentrated_molarity for a dilution.")
    return dilute_molarity * dilute_volume_ml / concentrated_molarity


@dataclass(frozen=True)
class Solution:
    """A solution described by solute identity and molarity."""

    solute_formula: str
    molarity: float
    density_g_per_ml: float | None = None
    weight_percent: float | None = None

    @classmethod
    def from_weight_percent(
        cls,
        solute_formula: str,
        weight_percent: float,
        density_g_per_ml: float,
    ) -> "Solution":
        """Build a solution from wt% and density.

        Uses a 100 g solution basis. Density converts that mass into solution
        volume, then formula mass converts solute mass to moles.
        """
        if not 0 < weight_percent <= 100:
            raise ValueError("weight_percent must be between 0 and 100.")
        if density_g_per_ml <= 0:
            raise ValueError("density_g_per_ml must be positive.")

        solute_grams = weight_percent
        solution_volume_l = (100.0 / density_g_per_ml) / 1000.0
        molarity = molarity_from_mass(solute_formula, solute_grams, solution_volume_l)
        return cls(solute_formula, molarity, density_g_per_ml, weight_percent)

    @classmethod
    def from_solute_mass(
        cls,
        solute_formula: str,
        solute_grams: float,
        solution_volume_l: float,
    ) -> "Solution":
        return cls(solute_formula, molarity_from_mass(solute_formula, solute_grams, solution_volume_l))

    def moles_in_volume_ml(self, volume_ml: float) -> float:
        return self.molarity * volume_ml / 1000.0

    def grams_in_volume_ml(self, volume_ml: float) -> float:
        return Formula(self.solute_formula).grams_from_moles(self.moles_in_volume_ml(volume_ml))

    def volume_ml_for_moles(self, moles: float) -> float:
        return moles / self.molarity * 1000.0

    def volume_ml_for_grams(self, grams: float) -> float:
        return self.volume_ml_for_moles(Formula(self.solute_formula).moles_from_grams(grams))

    def solution_mass_for_solute(self, solute_grams: float) -> float:
        if self.weight_percent is None:
            raise ValueError("weight_percent is required for solution mass calculations.")
        return solute_grams / (self.weight_percent / 100.0)
