"""Balanced reaction helpers for limiting-reagent calculations."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Reaction:
    """A balanced reaction represented by stoichiometric coefficients."""

    reactants: dict[str, float]
    products: dict[str, float]

    @classmethod
    def from_equation(cls, equation: str) -> "Reaction":
        """Parse a simple balanced equation such as ``2H2 + O2 -> 2H2O``."""
        left, right = equation.replace("→", "->").split("->")
        return cls(_parse_side(left), _parse_side(right))

    def limiting_reagent(self, available_moles: dict[str, float]) -> "LimitingReagentResult":
        missing = set(self.reactants) - set(available_moles)
        if missing:
            raise ValueError(f"Missing available moles for: {', '.join(sorted(missing))}")

        reaction_extents = {
            species: available_moles[species] / coefficient
            for species, coefficient in self.reactants.items()
        }
        limiting_species = min(reaction_extents, key=reaction_extents.get)
        extent = reaction_extents[limiting_species]
        consumed = {species: coefficient * extent for species, coefficient in self.reactants.items()}
        products = {species: coefficient * extent for species, coefficient in self.products.items()}

        return LimitingReagentResult(limiting_species, extent, consumed, products)


@dataclass(frozen=True)
class LimitingReagentResult:
    limiting_species: str
    reaction_extent: float
    reactant_moles_consumed: dict[str, float]
    product_moles_produced: dict[str, float]

    def excess_moles(self, available_moles: dict[str, float], species: str) -> float:
        return available_moles[species] - self.reactant_moles_consumed.get(species, 0.0)

    def product_moles(self, species: str) -> float:
        return self.product_moles_produced[species]


def _parse_side(side: str) -> dict[str, float]:
    terms = {}
    for raw_term in re.split(r"\s+\+\s+", side.strip()):
        term = raw_term.strip()
        coefficient, species = _split_coefficient(term)
        terms[species] = coefficient
    return terms


def _split_coefficient(term: str) -> tuple[float, str]:
    pieces = term.split(maxsplit=1)
    if len(pieces) == 2 and _is_number(pieces[0]):
        return float(pieces[0]), pieces[1].strip()

    index = 0
    while index < len(term) and (term[index].isdigit() or term[index] == "."):
        index += 1
    if index and index < len(term) and term[index].isalpha():
        return float(term[:index]), term[index:].strip()
    return 1.0, term


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
