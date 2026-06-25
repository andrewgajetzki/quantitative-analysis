"""Chemical formula parsing and formula-mass calculations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .constants import ATOMIC_MASSES


@dataclass(frozen=True)
class Formula:
    """A parsed chemical formula.

    Supports common textbook formula syntax such as ``H2SO4``, ``Ca(NO3)2``,
    ``CuSO4.5H2O``, and charged species such as ``Ba2+`` or ``SO4^2-``.
    Charge annotations are ignored for mass and stoichiometry labels.
    """

    text: str

    @property
    def atoms(self) -> Counter[str]:
        return _parse_formula(_strip_charge(self.text))

    @property
    def mass(self) -> float:
        return sum(ATOMIC_MASSES[element] * count for element, count in self.atoms.items())

    def moles_from_grams(self, grams: float) -> float:
        return grams / self.mass

    def grams_from_moles(self, moles: float) -> float:
        return moles * self.mass


def _strip_charge(formula: str) -> str:
    cleaned = formula.strip().replace(" ", "")
    if "^" in cleaned:
        return cleaned.split("^", 1)[0]
    had_charge_sign = cleaned.endswith(("+", "-"))
    while cleaned and cleaned[-1] in "+-":
        cleaned = cleaned[:-1]
    if had_charge_sign and cleaned and cleaned[-1].isdigit():
        cleaned = cleaned[:-1]
    if len(cleaned) >= 2 and cleaned[-1].isdigit() and cleaned[-2].isalpha():
        return cleaned
    if len(cleaned) >= 2 and cleaned[-1].isdigit() and not cleaned[-2].isdigit():
        return cleaned
    return cleaned


def _parse_formula(formula: str) -> Counter[str]:
    if not formula:
        raise ValueError("Formula cannot be empty.")

    total = Counter()
    for hydrate_part in formula.replace("·", ".").split("."):
        multiplier, part = _leading_multiplier(hydrate_part)
        total.update({element: count * multiplier for element, count in _parse_group(part).items()})
    return total


def _leading_multiplier(text: str) -> tuple[int, str]:
    index = 0
    while index < len(text) and text[index].isdigit():
        index += 1
    if index == 0:
        return 1, text
    return int(text[:index]), text[index:]


def _parse_group(text: str, start: int = 0) -> Counter[str]:
    counts: Counter[str] = Counter()
    index = start

    while index < len(text):
        character = text[index]
        if character == "(":
            nested, index = _parse_group(text, index + 1)
            multiplier, index = _read_number(text, index)
            counts.update({element: count * multiplier for element, count in nested.items()})
        elif character == ")":
            return counts, index + 1
        elif character.isupper():
            element, index = _read_element(text, index)
            if element not in ATOMIC_MASSES:
                raise ValueError(f"Unknown element: {element}")
            multiplier, index = _read_number(text, index)
            counts[element] += multiplier
        else:
            raise ValueError(f"Unexpected character {character!r} in formula {text!r}")

    if start:
        raise ValueError(f"Unclosed parenthesis in formula {text!r}")
    return counts


def _read_element(text: str, start: int) -> tuple[str, int]:
    end = start + 1
    if end < len(text) and text[end].islower():
        end += 1
    return text[start:end], end


def _read_number(text: str, start: int) -> tuple[int, int]:
    end = start
    while end < len(text) and text[end].isdigit():
        end += 1
    if end == start:
        return 1, start
    return int(text[start:end]), end
