"""Redox reaction, titration, and iodometric calculation helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
import math
import re

from electrochemistry import (
    STANDARD_TEMPERATURE_K,
    nernst_log10_slope_v,
)


SATURATED_CALOMEL_ELECTRODE_V = 0.241
SATURATED_AG_AGCL_ELECTRODE_V = 0.197


COMMON_OXIDATION_NUMBERS = {
    "H": 1,
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Rb": 1,
    "Cs": 1,
    "Be": 2,
    "Mg": 2,
    "Ca": 2,
    "Sr": 2,
    "Ba": 2,
    "F": -1,
    "Cl": -1,
    "Br": -1,
    "I": -1,
    "O": -2,
}


@dataclass(frozen=True)
class Species:
    """A chemical species split into formula atoms and net charge."""

    formula: str
    charge: int
    atoms: Counter[str]


@dataclass(frozen=True)
class HalfReaction:
    """A balanced half-reaction with electrons on exactly one side."""

    reactants: dict[str, Fraction]
    products: dict[str, Fraction]

    @property
    def electrons_transferred(self) -> Fraction:
        return self.reactants.get("e-", Fraction(0)) or self.products.get("e-", Fraction(0))

    @property
    def is_reduction(self) -> bool:
        return "e-" in self.reactants

    @property
    def is_oxidation(self) -> bool:
        return "e-" in self.products

    def scaled(self, factor: int | Fraction) -> "HalfReaction":
        scale = Fraction(factor)
        return HalfReaction(
            {species: coefficient * scale for species, coefficient in self.reactants.items()},
            {species: coefficient * scale for species, coefficient in self.products.items()},
        )

    def equation(self) -> str:
        """Return a compact equation string."""
        return f"{_format_side(self.reactants)} -> {_format_side(self.products)}"


@dataclass(frozen=True)
class RedoxReaction:
    """A balanced redox reaction after canceling electrons."""

    reactants: dict[str, Fraction]
    products: dict[str, Fraction]

    def equation(self) -> str:
        """Return a compact equation string."""
        return f"{_format_side(self.reactants)} -> {_format_side(self.products)}"


@dataclass(frozen=True)
class RedoxCouple:
    """A reduction couple written as oxidized form plus electrons to reduced form."""

    oxidized: str
    reduced: str
    electrons: float
    standard_potential_v: float
    fixed_numerator_factors: tuple[tuple[float, float], ...] = ()
    fixed_denominator_factors: tuple[tuple[float, float], ...] = ()

    def formal_potential_v(self, temperature_k: float = STANDARD_TEMPERATURE_K) -> float:
        """Return the potential adjusted for fixed Nernst terms such as fixed acid."""
        return redox_formal_potential(
            self.standard_potential_v,
            self.electrons,
            self.fixed_numerator_factors,
            self.fixed_denominator_factors,
            temperature_k,
        )

    def potential(
        self,
        oxidized_activity: float,
        reduced_activity: float,
        temperature_k: float = STANDARD_TEMPERATURE_K,
    ) -> float:
        """Return the indicator-electrode potential for this redox couple."""
        return redox_couple_potential(
            self.standard_potential_v,
            self.electrons,
            oxidized_activity,
            reduced_activity,
            self.fixed_numerator_factors,
            self.fixed_denominator_factors,
            temperature_k,
        )


@dataclass(frozen=True)
class RedoxTitrationState:
    """Composition and potential at one point in a redox titration."""

    stage: str
    titrant_volume_ml: float
    total_volume_ml: float
    equivalence_volume_ml: float
    indicator_potential_v: float
    cell_voltage_v: float | None
    analyte_oxidized_moles: float
    analyte_reduced_moles: float
    titrant_oxidized_moles: float
    titrant_reduced_moles: float


@dataclass(frozen=True)
class IndicatorTransitionRange:
    """Transition range for an oxidation-reduction indicator."""

    lower_v: float
    midpoint_v: float
    upper_v: float

    def contains(self, potential_v: float) -> bool:
        """Return True when a potential falls inside the visible transition range."""
        return self.lower_v <= potential_v <= self.upper_v


@dataclass(frozen=True)
class GranPlotResult:
    """Linearized redox titration data and its x-intercept."""

    slope: float
    intercept: float
    equivalence_volume_ml: float
    points: tuple[tuple[float, float], ...]


def parse_species(species: str, charge: int | None = None) -> Species:
    """Parse a formula label and infer or override its net charge."""
    formula, inferred_charge = _split_formula_charge(species)
    resolved_charge = inferred_charge if charge is None else charge
    return Species(formula=formula, charge=resolved_charge, atoms=_parse_formula(formula))


def species_charge(species: str) -> int:
    """Return the inferred charge from a species label such as ``Fe2+`` or ``SO4^2-``."""
    return parse_species(species).charge


def oxidation_number(
    formula: str,
    element: str,
    charge: int | None = None,
    known_oxidation_numbers: Mapping[str, float] | None = None,
) -> float:
    """Return the oxidation number of one element in a formula or ion.

    The target element must be the only element whose oxidation number is not
    supplied or covered by the common textbook defaults.
    """
    species = parse_species(formula, charge)
    if element not in species.atoms:
        raise ValueError(f"{element} is not present in {formula!r}.")

    oxidation_numbers = dict(COMMON_OXIDATION_NUMBERS)
    if known_oxidation_numbers is not None:
        oxidation_numbers.update(known_oxidation_numbers)

    known_charge = 0.0
    unknown_count = 0
    for atom, count in species.atoms.items():
        if atom == element:
            unknown_count += count
        elif atom in oxidation_numbers:
            known_charge += oxidation_numbers[atom] * count
        else:
            raise ValueError(f"Oxidation number for {atom} is unknown.")

    return (species.charge - known_charge) / unknown_count


def balance_half_reaction(
    reactant: str,
    product: str,
    medium: str = "acidic",
    reactant_charge: int | None = None,
    product_charge: int | None = None,
) -> HalfReaction:
    """Balance a single-reactant/single-product redox half-reaction.

    ``medium`` may be ``"acidic"`` or ``"basic"``. The helper balances atoms
    using H2O and H+, then converts H+ to H2O/OH- for basic solution.
    """
    medium_key = medium.lower()
    if medium_key not in {"acidic", "basic"}:
        raise ValueError("medium must be 'acidic' or 'basic'.")

    reactant_species = parse_species(reactant, reactant_charge)
    product_species = parse_species(product, product_charge)
    left, right = _balance_non_hydrogen_oxygen_atoms(reactant_species, product_species)
    _balance_oxygen_with_water(left, right)
    _balance_hydrogen_with_protons(left, right)
    if medium_key == "basic":
        _convert_protons_to_basic(left, right)
    _cancel_common_species(left, right)
    _balance_charge_with_electrons(left, right)

    reaction = HalfReaction(_clean_side(left), _clean_side(right))
    if not (reaction.is_reduction or reaction.is_oxidation):
        raise ValueError("No electron transfer was required for this half-reaction.")
    return reaction


def combine_half_reactions(oxidation: HalfReaction, reduction: HalfReaction) -> RedoxReaction:
    """Combine oxidation and reduction half-reactions into one balanced reaction."""
    if oxidation.is_reduction and reduction.is_oxidation:
        oxidation, reduction = reduction, oxidation
    if not oxidation.is_oxidation or not reduction.is_reduction:
        raise ValueError("One oxidation half-reaction and one reduction half-reaction are required.")

    oxidation_electrons = oxidation.electrons_transferred
    reduction_electrons = reduction.electrons_transferred
    electron_lcm = _lcm_int(int(oxidation_electrons), int(reduction_electrons))
    oxidation_scaled = oxidation.scaled(electron_lcm / oxidation_electrons)
    reduction_scaled = reduction.scaled(electron_lcm / reduction_electrons)

    reactants = Counter(oxidation_scaled.reactants)
    reactants.update(reduction_scaled.reactants)
    products = Counter(oxidation_scaled.products)
    products.update(reduction_scaled.products)
    reactants.pop("e-", None)
    products.pop("e-", None)
    _cancel_common_species(reactants, products)
    return RedoxReaction(_clean_side(reactants), _clean_side(products))


def redox_equivalents(moles: float, electrons_per_mole: float) -> float:
    """Return electron equivalents from moles and electron stoichiometry."""
    _require_nonnegative(moles, "moles")
    _require_positive(electrons_per_mole, "electrons_per_mole")
    return moles * electrons_per_mole


def normality_from_molarity(molarity: float, electrons_per_mole: float) -> float:
    """Return redox normality from molarity and electrons transferred per mole."""
    _require_nonnegative(molarity, "molarity")
    _require_positive(electrons_per_mole, "electrons_per_mole")
    return molarity * electrons_per_mole


def molarity_from_normality(normality: float, electrons_per_mole: float) -> float:
    """Return molarity from redox normality."""
    _require_nonnegative(normality, "normality")
    _require_positive(electrons_per_mole, "electrons_per_mole")
    return normality / electrons_per_mole


def redox_equivalence_volume_ml(
    analyte_molarity: float,
    analyte_volume_ml: float,
    titrant_molarity: float,
    analyte_electrons: float,
    titrant_electrons: float,
) -> float:
    """Return titrant volume at the redox equivalence point."""
    _require_nonnegative(analyte_molarity, "analyte_molarity")
    _require_positive(analyte_volume_ml, "analyte_volume_ml")
    _require_positive(titrant_molarity, "titrant_molarity")
    _require_positive(analyte_electrons, "analyte_electrons")
    _require_positive(titrant_electrons, "titrant_electrons")
    analyte_moles = analyte_molarity * analyte_volume_ml / 1000.0
    titrant_moles = analyte_moles * analyte_electrons / titrant_electrons
    return titrant_moles / titrant_molarity * 1000.0


def analyte_molarity_from_redox_titration(
    titrant_molarity: float,
    titrant_volume_ml: float,
    sample_volume_ml: float,
    analyte_electrons: float,
    titrant_electrons: float,
) -> float:
    """Return analyte molarity from redox endpoint stoichiometry."""
    _require_nonnegative(titrant_molarity, "titrant_molarity")
    _require_nonnegative(titrant_volume_ml, "titrant_volume_ml")
    _require_positive(sample_volume_ml, "sample_volume_ml")
    _require_positive(analyte_electrons, "analyte_electrons")
    _require_positive(titrant_electrons, "titrant_electrons")
    titrant_moles = titrant_molarity * titrant_volume_ml / 1000.0
    analyte_moles = titrant_moles * titrant_electrons / analyte_electrons
    return analyte_moles / (sample_volume_ml / 1000.0)


def redox_formal_potential(
    standard_potential_v: float,
    electrons_transferred: float,
    fixed_numerator_factors: Iterable[tuple[float, float]] = (),
    fixed_denominator_factors: Iterable[tuple[float, float]] = (),
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return E' after applying fixed Nernst activity factors.

    Each fixed factor is ``(activity, exponent)``. For example, the acidic
    permanganate couple uses ``fixed_numerator_factors=((h_activity, 8),)``.
    """
    slope = nernst_log10_slope_v(electrons_transferred, temperature_k)
    fixed_ratio = _activity_product(fixed_numerator_factors) / _activity_product(
        fixed_denominator_factors
    )
    return standard_potential_v + slope * math.log10(fixed_ratio)


def redox_couple_potential(
    standard_potential_v: float,
    electrons_transferred: float,
    oxidized_activity: float,
    reduced_activity: float,
    fixed_numerator_factors: Iterable[tuple[float, float]] = (),
    fixed_denominator_factors: Iterable[tuple[float, float]] = (),
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return E for a reduction couple from oxidized and reduced activities."""
    _require_positive(oxidized_activity, "oxidized_activity")
    _require_positive(reduced_activity, "reduced_activity")
    formal_potential = redox_formal_potential(
        standard_potential_v,
        electrons_transferred,
        fixed_numerator_factors,
        fixed_denominator_factors,
        temperature_k,
    )
    slope = nernst_log10_slope_v(electrons_transferred, temperature_k)
    return formal_potential + slope * math.log10(oxidized_activity / reduced_activity)


def indicator_cell_voltage(
    indicator_potential_v: float,
    reference_electrode_potential_v: float,
) -> float:
    """Return measured cell voltage for an indicator electrode versus a reference."""
    return indicator_potential_v - reference_electrode_potential_v


def redox_titration_state(
    analyte_molarity: float,
    analyte_volume_ml: float,
    titrant_molarity: float,
    titrant_volume_ml: float,
    analyte_couple: RedoxCouple,
    titrant_couple: RedoxCouple,
    analyte_initial_form: str = "reduced",
    reference_electrode_potential_v: float | None = None,
    equivalence_tolerance_ml: float = 1e-9,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> RedoxTitrationState:
    """Return composition and potential at a redox titration volume.

    ``analyte_initial_form`` is ``"reduced"`` for reductants such as Fe2+ or
    Sn2+ titrated by an oxidant, and ``"oxidized"`` for oxidants such as Ce4+
    titrated by a reductant.
    """
    _require_nonnegative(analyte_molarity, "analyte_molarity")
    _require_positive(analyte_volume_ml, "analyte_volume_ml")
    _require_positive(titrant_molarity, "titrant_molarity")
    _require_nonnegative(titrant_volume_ml, "titrant_volume_ml")
    _require_positive(equivalence_tolerance_ml, "equivalence_tolerance_ml")

    initial_form = analyte_initial_form.lower()
    if initial_form not in {"reduced", "oxidized"}:
        raise ValueError("analyte_initial_form must be 'reduced' or 'oxidized'.")
    if analyte_couple.electrons <= 0 or titrant_couple.electrons <= 0:
        raise ValueError("couple electron counts must be positive.")

    analyte_initial_moles = analyte_molarity * analyte_volume_ml / 1000.0
    titrant_added_moles = titrant_molarity * titrant_volume_ml / 1000.0
    equivalence_volume = redox_equivalence_volume_ml(
        analyte_molarity,
        analyte_volume_ml,
        titrant_molarity,
        analyte_couple.electrons,
        titrant_couple.electrons,
    )
    equivalence_titrant_moles = titrant_molarity * equivalence_volume / 1000.0
    total_volume_l = (analyte_volume_ml + titrant_volume_ml) / 1000.0

    if titrant_volume_ml < equivalence_volume - equivalence_tolerance_ml:
        stage = "before equivalence"
        analyte_reacted = titrant_added_moles * titrant_couple.electrons / analyte_couple.electrons
        analyte_unreacted = analyte_initial_moles - analyte_reacted
        potential_source = analyte_couple
        if initial_form == "reduced":
            analyte_reduced = analyte_unreacted
            analyte_oxidized = analyte_reacted
        else:
            analyte_oxidized = analyte_unreacted
            analyte_reduced = analyte_reacted
        titrant_oxidized, titrant_reduced = _titrant_product_moles(
            titrant_added_moles,
            initial_form,
        )
        potential = potential_source.potential(
            analyte_oxidized / total_volume_l,
            analyte_reduced / total_volume_l,
            temperature_k,
        )
    elif titrant_volume_ml > equivalence_volume + equivalence_tolerance_ml:
        stage = "after equivalence"
        titrant_excess = titrant_added_moles - equivalence_titrant_moles
        if initial_form == "reduced":
            analyte_reduced = 0.0
            analyte_oxidized = analyte_initial_moles
            titrant_oxidized = titrant_excess
            titrant_reduced = equivalence_titrant_moles
        else:
            analyte_oxidized = 0.0
            analyte_reduced = analyte_initial_moles
            titrant_reduced = titrant_excess
            titrant_oxidized = equivalence_titrant_moles
        potential = titrant_couple.potential(
            titrant_oxidized / total_volume_l,
            titrant_reduced / total_volume_l,
            temperature_k,
        )
    else:
        stage = "at equivalence"
        if initial_form == "reduced":
            analyte_reduced = 0.0
            analyte_oxidized = analyte_initial_moles
            titrant_oxidized = 0.0
            titrant_reduced = equivalence_titrant_moles
        else:
            analyte_oxidized = 0.0
            analyte_reduced = analyte_initial_moles
            titrant_reduced = 0.0
            titrant_oxidized = equivalence_titrant_moles
        potential = _equivalence_potential(analyte_couple, titrant_couple, temperature_k)

    cell_voltage = None
    if reference_electrode_potential_v is not None:
        cell_voltage = indicator_cell_voltage(potential, reference_electrode_potential_v)

    return RedoxTitrationState(
        stage=stage,
        titrant_volume_ml=titrant_volume_ml,
        total_volume_ml=analyte_volume_ml + titrant_volume_ml,
        equivalence_volume_ml=equivalence_volume,
        indicator_potential_v=potential,
        cell_voltage_v=cell_voltage,
        analyte_oxidized_moles=analyte_oxidized,
        analyte_reduced_moles=analyte_reduced,
        titrant_oxidized_moles=titrant_oxidized,
        titrant_reduced_moles=titrant_reduced,
    )


def redox_indicator_transition_range(
    standard_potential_v: float,
    electrons_transferred: float = 1.0,
    color_ratio: float = 10.0,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> IndicatorTransitionRange:
    """Return the approximate visual transition range for a redox indicator."""
    _require_positive(color_ratio, "color_ratio")
    span = nernst_log10_slope_v(electrons_transferred, temperature_k) * math.log10(color_ratio)
    return IndicatorTransitionRange(
        lower_v=standard_potential_v - span,
        midpoint_v=standard_potential_v,
        upper_v=standard_potential_v + span,
    )


def redox_indicator_is_suitable(
    equivalence_potential_v: float,
    indicator_standard_potential_v: float,
    electrons_transferred: float = 1.0,
    color_ratio: float = 10.0,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> bool:
    """Return True when an indicator transition contains the equivalence potential."""
    transition = redox_indicator_transition_range(
        indicator_standard_potential_v,
        electrons_transferred,
        color_ratio,
        temperature_k,
    )
    return transition.contains(equivalence_potential_v)


def redox_gran_equivalence_volume(
    titrant_volumes_ml: Iterable[float],
    potentials_v: Iterable[float],
    formal_potential_v: float,
    electrons_transferred: float,
    branch: str = "after",
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> GranPlotResult:
    """Estimate redox equivalence volume from linearized potentiometric data.

    Use ``branch="after"`` for excess titrant data governed by
    ``E = E' + S log10((V - Veq) / Veq)``. Use ``branch="before"`` for
    reductant-analyte data governed by ``E = E' + S log10(V / (Veq - V))``.
    """
    volumes = tuple(titrant_volumes_ml)
    potentials = tuple(potentials_v)
    if len(volumes) != len(potentials):
        raise ValueError("titrant_volumes_ml and potentials_v must have the same length.")
    if len(volumes) < 2:
        raise ValueError("At least two Gran-plot points are required.")
    for volume in volumes:
        _require_positive(volume, "titrant volume")

    branch_key = branch.lower()
    slope_v = nernst_log10_slope_v(electrons_transferred, temperature_k)
    if branch_key == "after":
        points = tuple(
            (volume, 10.0 ** ((potential - formal_potential_v) / slope_v))
            for volume, potential in zip(volumes, potentials, strict=True)
        )
    elif branch_key == "before":
        points = tuple(
            (
                volume,
                volume * 10.0 ** (-(potential - formal_potential_v) / slope_v),
            )
            for volume, potential in zip(volumes, potentials, strict=True)
        )
    else:
        raise ValueError("branch must be 'before' or 'after'.")

    slope, intercept = _linear_fit(points)
    if slope == 0:
        raise ValueError("Gran plot slope cannot be zero.")
    return GranPlotResult(
        slope=slope,
        intercept=intercept,
        equivalence_volume_ml=-intercept / slope,
        points=points,
    )


def iodine_moles_from_thiosulfate(
    thiosulfate_molarity: float,
    thiosulfate_volume_ml: float,
) -> float:
    """Return moles I2 from thiosulfate titration stoichiometry."""
    _require_nonnegative(thiosulfate_molarity, "thiosulfate_molarity")
    _require_nonnegative(thiosulfate_volume_ml, "thiosulfate_volume_ml")
    return thiosulfate_molarity * thiosulfate_volume_ml / 1000.0 / 2.0


def winkler_oxygen_moles_from_thiosulfate(
    thiosulfate_molarity: float,
    thiosulfate_volume_ml: float,
) -> float:
    """Return moles dissolved O2 from a Winkler iodometric titration."""
    _require_nonnegative(thiosulfate_molarity, "thiosulfate_molarity")
    _require_nonnegative(thiosulfate_volume_ml, "thiosulfate_volume_ml")
    return thiosulfate_molarity * thiosulfate_volume_ml / 1000.0 / 4.0


def winkler_oxygen_mg_per_l(
    thiosulfate_molarity: float,
    thiosulfate_volume_ml: float,
    sample_volume_ml: float,
) -> float:
    """Return dissolved oxygen concentration in mg/L by the Winkler method."""
    _require_positive(sample_volume_ml, "sample_volume_ml")
    oxygen_moles = winkler_oxygen_moles_from_thiosulfate(
        thiosulfate_molarity,
        thiosulfate_volume_ml,
    )
    return oxygen_moles * 31.9988 * 1000.0 / (sample_volume_ml / 1000.0)


def _titrant_product_moles(
    titrant_added_moles: float,
    analyte_initial_form: str,
) -> tuple[float, float]:
    if analyte_initial_form == "reduced":
        return 0.0, titrant_added_moles
    return titrant_added_moles, 0.0


def _equivalence_potential(
    analyte_couple: RedoxCouple,
    titrant_couple: RedoxCouple,
    temperature_k: float,
) -> float:
    analyte_formal = analyte_couple.formal_potential_v(temperature_k)
    titrant_formal = titrant_couple.formal_potential_v(temperature_k)
    return (
        analyte_couple.electrons * analyte_formal
        + titrant_couple.electrons * titrant_formal
    ) / (analyte_couple.electrons + titrant_couple.electrons)


def _activity_product(factors: Iterable[tuple[float, float]]) -> float:
    product = 1.0
    for activity, exponent in factors:
        _require_positive(activity, "fixed activity")
        product *= activity**exponent
    return product


def _split_formula_charge(species: str) -> tuple[str, int]:
    text = species.strip().replace(" ", "")
    if not text:
        raise ValueError("Species cannot be empty.")
    if text == "e-":
        return "e", -1
    if text in {"H+", "OH-", "H2O"}:
        return text.rstrip("+-"), {"H+": 1, "OH-": -1, "H2O": 0}[text]

    text = re.sub(r"\([a-zA-Z]+\)$", "", text)
    if "^" in text:
        formula, raw_charge = text.split("^", 1)
        return formula, _parse_charge_suffix(raw_charge)
    if text.endswith(("+", "-")):
        sign = 1 if text[-1] == "+" else -1
        sign_start = len(text) - 1
        while sign_start > 0 and text[sign_start - 1] in "+-":
            sign_start -= 1
        sign_count = len(text) - sign_start
        base = text[:sign_start]
        digit_start = len(base)
        while digit_start > 0 and base[digit_start - 1].isdigit():
            digit_start -= 1
        if sign_count > 1:
            return base, sign * sign_count
        if sign > 0 and digit_start < len(base) and _looks_like_monatomic_charge(base[:digit_start]):
            return base[:digit_start], sign * int(base[digit_start:])
        return base, sign
    return text, 0


def _parse_charge_suffix(raw_charge: str) -> int:
    match = re.fullmatch(r"([+-]?)(\d*)([+-]?)", raw_charge)
    if not match:
        raise ValueError(f"Invalid charge suffix: {raw_charge!r}")
    leading_sign, digits, trailing_sign = match.groups()
    sign_text = trailing_sign or leading_sign
    if not sign_text:
        raise ValueError(f"Invalid charge suffix: {raw_charge!r}")
    magnitude = int(digits) if digits else 1
    return magnitude if sign_text == "+" else -magnitude


def _looks_like_monatomic_charge(formula: str) -> bool:
    return re.fullmatch(r"[A-Z][a-z]?", formula) is not None


def _parse_formula(formula: str) -> Counter[str]:
    total: Counter[str] = Counter()
    for hydrate_part in formula.replace("·", ".").split("."):
        multiplier, part = _leading_multiplier(hydrate_part)
        total.update(
            {
                element: count * multiplier
                for element, count in _parse_formula_group(part).items()
            }
        )
    return total


def _leading_multiplier(text: str) -> tuple[int, str]:
    index = 0
    while index < len(text) and text[index].isdigit():
        index += 1
    if index == 0:
        return 1, text
    return int(text[:index]), text[index:]


def _parse_formula_group(text: str, start: int = 0) -> tuple[Counter[str], int] | Counter[str]:
    counts: Counter[str] = Counter()
    index = start
    while index < len(text):
        character = text[index]
        if character == "(":
            nested, index = _parse_formula_group(text, index + 1)
            multiplier, index = _read_number(text, index)
            counts.update({element: count * multiplier for element, count in nested.items()})
        elif character == ")":
            return counts, index + 1
        elif character.isupper():
            element, index = _read_element(text, index)
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


def _balance_non_hydrogen_oxygen_atoms(
    reactant: Species,
    product: Species,
) -> tuple[Counter[str], Counter[str]]:
    elements = (set(reactant.atoms) | set(product.atoms)) - {"H", "O"}
    left_coefficient = Fraction(1)
    right_coefficient = Fraction(1)
    ratio: Fraction | None = None
    for element in elements:
        reactant_count = reactant.atoms.get(element, 0)
        product_count = product.atoms.get(element, 0)
        if reactant_count == 0 or product_count == 0:
            raise ValueError(f"Cannot balance element {element} between {reactant.formula} and {product.formula}.")
        candidate = Fraction(reactant_count, product_count)
        if ratio is None:
            ratio = candidate
        elif ratio != candidate:
            raise ValueError("Multiple non-H/O elements require incompatible coefficients.")

    if ratio is not None:
        left_coefficient = Fraction(ratio.denominator)
        right_coefficient = Fraction(ratio.numerator)

    return (
        Counter({_species_label(reactant.formula, reactant.charge): left_coefficient}),
        Counter({_species_label(product.formula, product.charge): right_coefficient}),
    )


def _balance_oxygen_with_water(left: Counter[str], right: Counter[str]) -> None:
    left_oxygen = _side_atom_count(left, "O")
    right_oxygen = _side_atom_count(right, "O")
    difference = left_oxygen - right_oxygen
    if difference > 0:
        right["H2O"] += difference
    elif difference < 0:
        left["H2O"] += -difference


def _balance_hydrogen_with_protons(left: Counter[str], right: Counter[str]) -> None:
    left_hydrogen = _side_atom_count(left, "H")
    right_hydrogen = _side_atom_count(right, "H")
    difference = left_hydrogen - right_hydrogen
    if difference > 0:
        right["H+"] += difference
    elif difference < 0:
        left["H+"] += -difference


def _convert_protons_to_basic(left: Counter[str], right: Counter[str]) -> None:
    if "H+" in left:
        coefficient = left.pop("H+")
        left["H2O"] += coefficient
        right["OH-"] += coefficient
    if "H+" in right:
        coefficient = right.pop("H+")
        right["H2O"] += coefficient
        left["OH-"] += coefficient
    _cancel_common_species(left, right)


def _balance_charge_with_electrons(left: Counter[str], right: Counter[str]) -> None:
    left_charge = _side_charge(left)
    right_charge = _side_charge(right)
    difference = left_charge - right_charge
    if difference > 0:
        left["e-"] += difference
    elif difference < 0:
        right["e-"] += -difference


def _side_atom_count(side: Mapping[str, Fraction], element: str) -> Fraction:
    total = Fraction(0)
    for species, coefficient in side.items():
        if species == "e-":
            continue
        total += coefficient * parse_species(species).atoms.get(element, 0)
    return total


def _side_charge(side: Mapping[str, Fraction]) -> Fraction:
    total = Fraction(0)
    for species, coefficient in side.items():
        if species == "e-":
            total -= coefficient
        else:
            total += coefficient * parse_species(species).charge
    return total


def _cancel_common_species(left: Counter[str], right: Counter[str]) -> None:
    for species in set(left) & set(right):
        amount = min(left[species], right[species])
        left[species] -= amount
        right[species] -= amount
        if left[species] == 0:
            del left[species]
        if right[species] == 0:
            del right[species]


def _clean_side(side: Mapping[str, Fraction]) -> dict[str, Fraction]:
    return {species: coefficient for species, coefficient in side.items() if coefficient != 0}


def _charge_label(charge: int) -> str:
    if charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    if magnitude == 1:
        return sign
    return f"{magnitude}{sign}"


def _species_label(formula: str, charge: int) -> str:
    if charge == 0:
        return formula
    if abs(charge) == 1 or _looks_like_monatomic_charge(formula):
        return formula + _charge_label(charge)
    sign = "+" if charge > 0 else "-"
    return f"{formula}^{abs(charge)}{sign}"


def _format_side(side: Mapping[str, Fraction]) -> str:
    if not side:
        return "0"
    return " + ".join(_format_term(species, coefficient) for species, coefficient in side.items())


def _format_term(species: str, coefficient: Fraction) -> str:
    if coefficient == 1:
        return species
    if coefficient.denominator == 1:
        return f"{coefficient.numerator} {species}"
    return f"{coefficient} {species}"


def _linear_fit(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    count = len(points)
    x_mean = sum(point[0] for point in points) / count
    y_mean = sum(point[1] for point in points) / count
    ss_xx = sum((point[0] - x_mean) ** 2 for point in points)
    if ss_xx == 0:
        raise ValueError("x values must not all be identical.")
    ss_xy = sum((point[0] - x_mean) * (point[1] - y_mean) for point in points)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _lcm_int(first: int, second: int) -> int:
    return abs(first * second) // math.gcd(first, second)


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
