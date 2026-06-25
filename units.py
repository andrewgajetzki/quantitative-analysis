"""Small unit-conversion helper for quantitative-analysis calculations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitConverter:
    """Convert between units by registering factors to a base unit."""

    factors_to_base: dict[str, float]

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        try:
            return value * self.factors_to_base[from_unit] / self.factors_to_base[to_unit]
        except KeyError as error:
            raise ValueError(f"Unknown unit: {error.args[0]}") from error


mass = UnitConverter(
    {
        "kg": 1000.0,
        "g": 1.0,
        "mg": 1e-3,
        "ug": 1e-6,
        "lb": 453.59237,
        "oz": 28.349523125,
    }
)

volume = UnitConverter(
    {
        "L": 1.0,
        "mL": 1e-3,
        "uL": 1e-6,
        "m3": 1000.0,
        "cm3": 1e-3,
    }
)

energy = UnitConverter(
    {
        "J": 1.0,
        "kJ": 1000.0,
        "cal": 4.184,
        "kcal": 4184.0,
    }
)
