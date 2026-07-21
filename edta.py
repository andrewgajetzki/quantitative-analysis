"""EDTA complexometric titration and assay helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math


EDTA_ACID_DISSOCIATION_CONSTANTS = (1.02e-2, 2.14e-3, 6.92e-7, 5.50e-11)
EDTA_SPECIES_NAMES = ("H4Y", "H3Y-", "H2Y2-", "HY3-", "Y4-")


@dataclass(frozen=True)
class EDTATitrationState:
    """Composition at one point in a 1:1 metal-EDTA titration."""

    metal_molarity: float
    metal_volume_ml: float
    edta_molarity: float
    edta_volume_ml: float
    conditional_formation_constant: float
    total_volume_ml: float
    total_metal_concentration: float
    total_edta_concentration: float
    complex_concentration: float
    free_metal_concentration: float
    free_edta_concentration: float
    p_metal: float
    stage: str


@dataclass(frozen=True)
class EDTAAssayResult:
    """Stoichiometric result from a direct EDTA titration."""

    edta_moles: float
    analyte_moles: float
    analyte_molarity: float | None = None


@dataclass(frozen=True)
class EDTABackTitrationResult:
    """Stoichiometric result from adding excess EDTA and back-titrating it."""

    edta_added_moles: float
    back_titrant_moles: float
    excess_edta_moles: float
    edta_consumed_moles: float
    analyte_moles: float
    analyte_molarity: float | None = None


def edta_species_fractions_from_ph(
    ph: float,
    acid_dissociation_constants: Iterable[float] = EDTA_ACID_DISSOCIATION_CONSTANTS,
) -> tuple[float, ...]:
    """Return EDTA acid-base fractions from fully protonated EDTA through Y4-."""
    hydrogen = 10.0 ** (-ph)
    constants = tuple(acid_dissociation_constants)
    if not constants:
        return (1.0,)

    terms = []
    cumulative_constant = 1.0
    proton_count = len(constants)
    for deprotonations in range(proton_count + 1):
        if deprotonations:
            constant = constants[deprotonations - 1]
            _require_positive(constant, "acid_dissociation_constant")
            cumulative_constant *= constant
        terms.append(cumulative_constant * hydrogen ** (proton_count - deprotonations))

    denominator = sum(terms)
    return tuple(term / denominator for term in terms)


def edta_y4_fraction_from_ph(
    ph: float,
    acid_dissociation_constants: Iterable[float] = EDTA_ACID_DISSOCIATION_CONSTANTS,
) -> float:
    """Return alpha_Y4-, the fraction of unbound EDTA present as Y4-."""
    return edta_species_fractions_from_ph(ph, acid_dissociation_constants)[-1]


def edta_species_concentrations_from_ph(
    total_unbound_edta_concentration: float,
    ph: float,
    acid_dissociation_constants: Iterable[float] = EDTA_ACID_DISSOCIATION_CONSTANTS,
    species_names: Iterable[str] = EDTA_SPECIES_NAMES,
) -> dict[str, float]:
    """Return concentrations of unbound EDTA acid-base forms at a fixed pH."""
    _require_nonnegative(total_unbound_edta_concentration, "total_unbound_edta_concentration")
    fractions = edta_species_fractions_from_ph(ph, acid_dissociation_constants)
    names = tuple(species_names)
    if len(names) != len(fractions):
        raise ValueError("species_names length must match the number of EDTA species.")
    return {
        name: total_unbound_edta_concentration * fraction
        for name, fraction in zip(names, fractions)
    }


def free_metal_fraction_with_complexing_agent(
    free_ligand_concentration: float,
    cumulative_formation_constants: Iterable[float],
) -> float:
    """Return alpha_M for metal side reactions M + nL <=> MLn."""
    _require_nonnegative(free_ligand_concentration, "free_ligand_concentration")
    denominator = 1.0
    for ligand_count, beta in enumerate(cumulative_formation_constants, start=1):
        _require_positive(beta, "cumulative_formation_constant")
        denominator += beta * free_ligand_concentration**ligand_count
    return 1.0 / denominator


def edta_conditional_formation_constant(
    formation_constant: float,
    ph: float | None = None,
    *,
    edta_y4_fraction: float | None = None,
    acid_dissociation_constants: Iterable[float] = EDTA_ACID_DISSOCIATION_CONSTANTS,
    free_metal_fraction: float = 1.0,
) -> float:
    """Return Kf' or Kf'' for EDTA complexation at fixed pH and side reactions."""
    _require_positive(formation_constant, "formation_constant")
    _require_fraction_greater_than_zero(free_metal_fraction, "free_metal_fraction")
    if edta_y4_fraction is None:
        if ph is None:
            raise ValueError("ph is required when edta_y4_fraction is not provided.")
        edta_y4_fraction = edta_y4_fraction_from_ph(ph, acid_dissociation_constants)
    _require_fraction_greater_than_zero(edta_y4_fraction, "edta_y4_fraction")
    return formation_constant * edta_y4_fraction * free_metal_fraction


def metal_buffer_free_metal_concentration(
    complex_concentration: float,
    unbound_edta_concentration: float,
    conditional_formation_constant: float,
) -> float:
    """Return [M] in an EDTA metal-ion buffer from [MY] and unbound EDTA."""
    _require_nonnegative(complex_concentration, "complex_concentration")
    _require_positive(unbound_edta_concentration, "unbound_edta_concentration")
    _require_positive(conditional_formation_constant, "conditional_formation_constant")
    return complex_concentration / (
        conditional_formation_constant * unbound_edta_concentration
    )


def p_metal_from_concentration(free_metal_concentration: float) -> float:
    """Return pM = -log10([M])."""
    _require_positive(free_metal_concentration, "free_metal_concentration")
    return -math.log10(free_metal_concentration)


def edta_equivalence_volume_ml(
    metal_molarity: float,
    metal_volume_ml: float,
    edta_molarity: float,
    edta_per_metal: float = 1.0,
) -> float:
    """Return EDTA volume needed to reach the complexometric equivalence point."""
    _require_nonnegative(metal_molarity, "metal_molarity")
    _require_nonnegative(metal_volume_ml, "metal_volume_ml")
    _require_positive(edta_molarity, "edta_molarity")
    _require_positive(edta_per_metal, "edta_per_metal")
    return metal_molarity * _ml_to_l(metal_volume_ml) * edta_per_metal / edta_molarity * 1000.0


def metal_molarity_from_edta_titration(
    edta_molarity: float,
    edta_volume_ml: float,
    sample_volume_ml: float,
    edta_per_metal: float = 1.0,
) -> float:
    """Return metal molarity from a direct EDTA titration endpoint."""
    _require_nonnegative(edta_molarity, "edta_molarity")
    _require_nonnegative(edta_volume_ml, "edta_volume_ml")
    _require_positive(sample_volume_ml, "sample_volume_ml")
    _require_positive(edta_per_metal, "edta_per_metal")
    return edta_molarity * _ml_to_l(edta_volume_ml) / (
        _ml_to_l(sample_volume_ml) * edta_per_metal
    )


def edta_titration_state(
    metal_molarity: float,
    metal_volume_ml: float,
    edta_molarity: float,
    edta_volume_ml: float,
    conditional_formation_constant: float,
) -> EDTATitrationState:
    """Return free metal, free EDTA, complex, and pM at a titration volume."""
    _require_nonnegative(metal_molarity, "metal_molarity")
    _require_nonnegative(metal_volume_ml, "metal_volume_ml")
    _require_nonnegative(edta_molarity, "edta_molarity")
    _require_nonnegative(edta_volume_ml, "edta_volume_ml")
    _require_positive(conditional_formation_constant, "conditional_formation_constant")
    total_volume_ml = metal_volume_ml + edta_volume_ml
    _require_positive(total_volume_ml, "total_volume_ml")

    total_metal = metal_molarity * _ml_to_l(metal_volume_ml) / _ml_to_l(total_volume_ml)
    total_edta = edta_molarity * _ml_to_l(edta_volume_ml) / _ml_to_l(total_volume_ml)
    free_metal = _free_metal_for_one_to_one_complex(
        total_metal,
        total_edta,
        conditional_formation_constant,
    )
    complex_concentration = total_metal - free_metal
    free_edta = max(total_edta - complex_concentration, 0.0)

    stage = "before_equivalence"
    if math.isclose(total_edta, total_metal, rel_tol=1e-12, abs_tol=1e-15):
        stage = "at_equivalence"
    elif total_edta > total_metal:
        stage = "after_equivalence"

    return EDTATitrationState(
        metal_molarity=metal_molarity,
        metal_volume_ml=metal_volume_ml,
        edta_molarity=edta_molarity,
        edta_volume_ml=edta_volume_ml,
        conditional_formation_constant=conditional_formation_constant,
        total_volume_ml=total_volume_ml,
        total_metal_concentration=total_metal,
        total_edta_concentration=total_edta,
        complex_concentration=complex_concentration,
        free_metal_concentration=free_metal,
        free_edta_concentration=free_edta,
        p_metal=p_metal_from_concentration(free_metal),
        stage=stage,
    )


def edta_titration_curve(
    metal_molarity: float,
    metal_volume_ml: float,
    edta_molarity: float,
    edta_volumes_ml: Iterable[float],
    conditional_formation_constant: float,
) -> tuple[EDTATitrationState, ...]:
    """Return titration states for a sequence of EDTA volumes."""
    return tuple(
        edta_titration_state(
            metal_molarity,
            metal_volume_ml,
            edta_molarity,
            volume_ml,
            conditional_formation_constant,
        )
        for volume_ml in edta_volumes_ml
    )


def metal_indicator_complex_fraction(
    free_metal_concentration: float,
    indicator_formation_constant: float,
) -> float:
    """Return the fraction of indicator bound as MIn at a fixed free-metal level."""
    _require_nonnegative(free_metal_concentration, "free_metal_concentration")
    _require_positive(indicator_formation_constant, "indicator_formation_constant")
    term = indicator_formation_constant * free_metal_concentration
    return term / (1.0 + term)


def metal_indicator_color(
    free_metal_concentration: float,
    indicator_formation_constant: float,
    free_indicator_color: str,
    metal_indicator_color_value: str,
    *,
    mixed_lower_fraction: float = 0.1,
    mixed_upper_fraction: float = 0.9,
) -> str:
    """Return the expected metal-ion indicator color at a fixed free-metal level."""
    _require_fraction_between_zero_and_one(mixed_lower_fraction, "mixed_lower_fraction")
    _require_fraction_between_zero_and_one(mixed_upper_fraction, "mixed_upper_fraction")
    if mixed_lower_fraction > mixed_upper_fraction:
        raise ValueError("mixed_lower_fraction cannot exceed mixed_upper_fraction.")

    fraction = metal_indicator_complex_fraction(
        free_metal_concentration,
        indicator_formation_constant,
    )
    if fraction <= mixed_lower_fraction:
        return free_indicator_color
    if fraction >= mixed_upper_fraction:
        return metal_indicator_color_value
    return "mixed"


def direct_edta_assay(
    edta_molarity: float,
    edta_volume_ml: float,
    *,
    sample_volume_ml: float | None = None,
    edta_per_analyte: float = 1.0,
) -> EDTAAssayResult:
    """Return analyte amount from a direct EDTA endpoint."""
    _require_nonnegative(edta_molarity, "edta_molarity")
    _require_nonnegative(edta_volume_ml, "edta_volume_ml")
    _require_positive(edta_per_analyte, "edta_per_analyte")
    edta_moles = edta_molarity * _ml_to_l(edta_volume_ml)
    analyte_moles = edta_moles / edta_per_analyte
    analyte_molarity = None
    if sample_volume_ml is not None:
        _require_positive(sample_volume_ml, "sample_volume_ml")
        analyte_molarity = analyte_moles / _ml_to_l(sample_volume_ml)
    return EDTAAssayResult(edta_moles, analyte_moles, analyte_molarity)


def back_edta_titration(
    edta_molarity: float,
    edta_volume_ml: float,
    back_titrant_molarity: float,
    back_titrant_volume_ml: float,
    *,
    sample_volume_ml: float | None = None,
    edta_per_analyte: float = 1.0,
    edta_per_back_titrant: float = 1.0,
) -> EDTABackTitrationResult:
    """Return analyte amount from excess-EDTA/back-titration data."""
    _require_nonnegative(edta_molarity, "edta_molarity")
    _require_nonnegative(edta_volume_ml, "edta_volume_ml")
    _require_nonnegative(back_titrant_molarity, "back_titrant_molarity")
    _require_nonnegative(back_titrant_volume_ml, "back_titrant_volume_ml")
    _require_positive(edta_per_analyte, "edta_per_analyte")
    _require_positive(edta_per_back_titrant, "edta_per_back_titrant")

    edta_added = edta_molarity * _ml_to_l(edta_volume_ml)
    back_titrant_moles = back_titrant_molarity * _ml_to_l(back_titrant_volume_ml)
    excess_edta = back_titrant_moles * edta_per_back_titrant
    edta_consumed = edta_added - excess_edta
    if edta_consumed < -1e-15:
        raise ValueError("back titration consumes more EDTA than was added.")
    edta_consumed = max(edta_consumed, 0.0)
    analyte_moles = edta_consumed / edta_per_analyte
    analyte_molarity = None
    if sample_volume_ml is not None:
        _require_positive(sample_volume_ml, "sample_volume_ml")
        analyte_molarity = analyte_moles / _ml_to_l(sample_volume_ml)

    return EDTABackTitrationResult(
        edta_added_moles=edta_added,
        back_titrant_moles=back_titrant_moles,
        excess_edta_moles=excess_edta,
        edta_consumed_moles=edta_consumed,
        analyte_moles=analyte_moles,
        analyte_molarity=analyte_molarity,
    )


def displacement_edta_assay(
    edta_molarity: float,
    edta_volume_ml: float,
    *,
    sample_volume_ml: float | None = None,
    displaced_metal_per_analyte: float = 1.0,
    edta_per_displaced_metal: float = 1.0,
) -> EDTAAssayResult:
    """Return analyte amount when displaced metal is titrated with EDTA."""
    _require_positive(displaced_metal_per_analyte, "displaced_metal_per_analyte")
    _require_positive(edta_per_displaced_metal, "edta_per_displaced_metal")
    assay = direct_edta_assay(
        edta_molarity,
        edta_volume_ml,
        sample_volume_ml=sample_volume_ml,
        edta_per_analyte=displaced_metal_per_analyte * edta_per_displaced_metal,
    )
    return assay


def _free_metal_for_one_to_one_complex(
    total_metal_concentration: float,
    total_edta_concentration: float,
    conditional_formation_constant: float,
) -> float:
    _require_nonnegative(total_metal_concentration, "total_metal_concentration")
    _require_nonnegative(total_edta_concentration, "total_edta_concentration")
    if total_metal_concentration == 0:
        return 0.0
    if total_edta_concentration == 0:
        return total_metal_concentration

    difference = total_edta_concentration - total_metal_concentration
    b = conditional_formation_constant * difference + 1.0
    discriminant = b * b + 4.0 * conditional_formation_constant * total_metal_concentration
    root = math.sqrt(discriminant)
    if b >= 0:
        return 2.0 * total_metal_concentration / (root + b)
    return (root - b) / (2.0 * conditional_formation_constant)


def _ml_to_l(volume_ml: float) -> float:
    _require_nonnegative(volume_ml, "volume_ml")
    return volume_ml / 1000.0


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _require_fraction_between_zero_and_one(value: float, name: str) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _require_fraction_greater_than_zero(value: float, name: str) -> None:
    if value <= 0 or value > 1:
        raise ValueError(f"{name} must be greater than 0 and no greater than 1.")
