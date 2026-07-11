"""Chemical-equilibrium, acid-base, solubility, and thermodynamics helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


KW_25C = 1.0e-14
R_J_PER_MOL_K = 8.314_462_618_153_24


@dataclass(frozen=True)
class EquilibriumResult:
    """Equilibrium composition for a single reaction extent."""

    concentrations: dict[str, float]
    extent: float
    reaction_quotient: float


@dataclass(frozen=True)
class PrecipitationResult:
    """Ion-product comparison against a solubility product."""

    ion_product: float
    ksp: float
    precipitates: bool


def reaction_quotient(
    concentrations: Mapping[str, float],
    stoichiometry: Mapping[str, float],
) -> float:
    """Return Q for species in the equilibrium expression.

    Use positive stoichiometric coefficients for products and negative
    coefficients for reactants. Pure solids and liquids should be omitted.
    """
    quotient = 1.0
    for species, coefficient in stoichiometry.items():
        if coefficient == 0:
            continue
        if species not in concentrations:
            raise ValueError(f"Missing concentration for {species}.")
        concentration = concentrations[species]
        _require_nonnegative(concentration, f"concentration for {species}")
        if concentration == 0:
            return 0.0 if coefficient > 0 else math.inf
        quotient *= concentration**coefficient
    return quotient


def equilibrium_direction(
    reaction_quotient_value: float,
    equilibrium_constant: float,
    tolerance: float = 1e-12,
) -> str:
    """Return whether a reaction shifts forward, reverse, or is at equilibrium."""
    _require_nonnegative(reaction_quotient_value, "reaction_quotient_value")
    _require_positive(equilibrium_constant, "equilibrium_constant")
    if math.isclose(
        reaction_quotient_value,
        equilibrium_constant,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        return "at_equilibrium"
    if reaction_quotient_value < equilibrium_constant:
        return "forward"
    return "reverse"


def combine_equilibrium_constants(*constants: tuple[float, float]) -> float:
    """Return the net K after reversing, scaling, or adding reactions.

    Each item is ``(K, multiplier)``. A multiplier of ``-1`` reverses a
    reaction, ``2`` doubles it, and added reactions are multiplied together.
    """
    combined = 1.0
    for equilibrium_constant, multiplier in constants:
        _require_positive(equilibrium_constant, "equilibrium_constant")
        combined *= equilibrium_constant**multiplier
    return combined


def solve_equilibrium(
    initial_concentrations: Mapping[str, float],
    stoichiometry: Mapping[str, float],
    equilibrium_constant: float,
    *,
    extent_bounds: tuple[float, float] | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> EquilibriumResult:
    """Solve a single-reaction ICE-table problem by bisection on extent.

    Stoichiometry follows the same sign convention as :func:`reaction_quotient`:
    products are positive and reactants are negative. Missing species from
    ``initial_concentrations`` are treated as zero.
    """
    _require_positive(equilibrium_constant, "equilibrium_constant")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if not stoichiometry:
        raise ValueError("stoichiometry must not be empty.")

    initial = _normalized_initial_concentrations(initial_concentrations, stoichiometry)
    lower, upper = extent_bounds or _extent_bounds(initial, stoichiometry)
    if lower > upper:
        raise ValueError("extent lower bound cannot exceed upper bound.")

    log_k = math.log(equilibrium_constant)
    initial_extent = 0.0
    if lower <= initial_extent <= upper:
        initial_residual = _log_reaction_quotient(initial, stoichiometry, initial_extent) - log_k
        if math.isfinite(initial_residual) and abs(initial_residual) <= tolerance:
            return _equilibrium_result(initial, stoichiometry, initial_extent)

    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=tolerance):
        quotient = reaction_quotient(_concentrations_at_extent(initial, stoichiometry, lower), stoichiometry)
        if math.isclose(quotient, equilibrium_constant, rel_tol=tolerance, abs_tol=tolerance):
            return _equilibrium_result(initial, stoichiometry, lower)
        raise ValueError("No equilibrium extent exists within the provided bounds.")

    span = upper - lower
    endpoint_offset = max(1e-15, abs(span) * 1e-12)
    low = lower + endpoint_offset
    high = upper - endpoint_offset
    if low > high:
        low = high = (lower + upper) / 2.0

    low_residual = _log_reaction_quotient(initial, stoichiometry, low) - log_k
    high_residual = _log_reaction_quotient(initial, stoichiometry, high) - log_k

    if abs(low_residual) <= tolerance:
        return _equilibrium_result(initial, stoichiometry, low)
    if abs(high_residual) <= tolerance:
        return _equilibrium_result(initial, stoichiometry, high)
    if _same_sign(low_residual, high_residual):
        raise ValueError("Equilibrium extent is not bracketed by the allowed concentration bounds.")

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        residual = _log_reaction_quotient(initial, stoichiometry, mid) - log_k
        if abs(residual) <= tolerance or math.isclose(low, high, rel_tol=tolerance, abs_tol=tolerance):
            return _equilibrium_result(initial, stoichiometry, mid)
        if _same_sign(low_residual, residual):
            low = mid
            low_residual = residual
        else:
            high = mid

    return _equilibrium_result(initial, stoichiometry, (low + high) / 2.0)


def hydrogen_from_ph(ph: float) -> float:
    """Return hydrogen-ion concentration from pH."""
    return 10.0**(-ph)


def hydroxide_from_poh(poh: float) -> float:
    """Return hydroxide-ion concentration from pOH."""
    return 10.0**(-poh)


def ph_from_hydrogen(hydrogen_concentration: float) -> float:
    """Return pH from hydrogen-ion concentration."""
    _require_positive(hydrogen_concentration, "hydrogen_concentration")
    return -math.log10(hydrogen_concentration)


def poh_from_hydroxide(hydroxide_concentration: float) -> float:
    """Return pOH from hydroxide-ion concentration."""
    _require_positive(hydroxide_concentration, "hydroxide_concentration")
    return -math.log10(hydroxide_concentration)


def pkw(kw: float = KW_25C) -> float:
    """Return pKw from Kw."""
    _require_positive(kw, "kw")
    return -math.log10(kw)


def poh_from_ph(ph: float, kw: float = KW_25C) -> float:
    """Return pOH from pH and Kw."""
    return pkw(kw) - ph


def ph_from_poh(poh: float, kw: float = KW_25C) -> float:
    """Return pH from pOH and Kw."""
    return pkw(kw) - poh


def hydroxide_from_hydrogen(hydrogen_concentration: float, kw: float = KW_25C) -> float:
    """Return hydroxide-ion concentration from [H+] and Kw."""
    _require_positive(hydrogen_concentration, "hydrogen_concentration")
    _require_positive(kw, "kw")
    return kw / hydrogen_concentration


def hydrogen_from_hydroxide(hydroxide_concentration: float, kw: float = KW_25C) -> float:
    """Return hydrogen-ion concentration from [OH-] and Kw."""
    _require_positive(hydroxide_concentration, "hydroxide_concentration")
    _require_positive(kw, "kw")
    return kw / hydroxide_concentration


def strong_acid_ph(concentration: float, acidic_protons: float = 1.0) -> float:
    """Return pH for a strong acid that fully dissociates."""
    _require_positive(concentration, "concentration")
    _require_positive(acidic_protons, "acidic_protons")
    return ph_from_hydrogen(concentration * acidic_protons)


def strong_base_ph(
    concentration: float,
    hydroxides: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return pH for a strong base that fully dissociates."""
    _require_positive(concentration, "concentration")
    _require_positive(hydroxides, "hydroxides")
    return ph_from_poh(poh_from_hydroxide(concentration * hydroxides), kw)


def ka_from_pka(pka: float) -> float:
    """Return Ka from pKa."""
    return 10.0**(-pka)


def pka_from_ka(ka: float) -> float:
    """Return pKa from Ka."""
    _require_positive(ka, "ka")
    return -math.log10(ka)


def kb_from_pkb(pkb: float) -> float:
    """Return Kb from pKb."""
    return 10.0**(-pkb)


def pkb_from_kb(kb: float) -> float:
    """Return pKb from Kb."""
    _require_positive(kb, "kb")
    return -math.log10(kb)


def conjugate_kb(ka: float, kw: float = KW_25C) -> float:
    """Return Kb for the conjugate base of an acid."""
    _require_positive(ka, "ka")
    _require_positive(kw, "kw")
    return kw / ka


def conjugate_ka(kb: float, kw: float = KW_25C) -> float:
    """Return Ka for the conjugate acid of a base."""
    _require_positive(kb, "kb")
    _require_positive(kw, "kw")
    return kw / kb


def weak_acid_hydrogen_concentration(formal_concentration: float, ka: float) -> float:
    """Return [H+] for HA <=> H+ + A- using the exact quadratic solution."""
    _require_positive(formal_concentration, "formal_concentration")
    _require_positive(ka, "ka")
    return (-ka + math.sqrt(ka**2 + 4.0 * ka * formal_concentration)) / 2.0


def weak_acid_ph(formal_concentration: float, ka: float) -> float:
    """Return pH for a monoprotic weak acid."""
    return ph_from_hydrogen(weak_acid_hydrogen_concentration(formal_concentration, ka))


def weak_base_hydroxide_concentration(formal_concentration: float, kb: float) -> float:
    """Return [OH-] for B + H2O <=> BH+ + OH- using the exact quadratic solution."""
    _require_positive(formal_concentration, "formal_concentration")
    _require_positive(kb, "kb")
    return (-kb + math.sqrt(kb**2 + 4.0 * kb * formal_concentration)) / 2.0


def weak_base_ph(formal_concentration: float, kb: float, kw: float = KW_25C) -> float:
    """Return pH for a weak base."""
    hydroxide = weak_base_hydroxide_concentration(formal_concentration, kb)
    return ph_from_poh(poh_from_hydroxide(hydroxide), kw)


def percent_ionization_weak_acid(formal_concentration: float, ka: float) -> float:
    """Return percent ionization for a monoprotic weak acid."""
    return weak_acid_hydrogen_concentration(formal_concentration, ka) / formal_concentration * 100.0


def percent_ionization_weak_base(formal_concentration: float, kb: float) -> float:
    """Return percent ionization for a weak base."""
    return weak_base_hydroxide_concentration(formal_concentration, kb) / formal_concentration * 100.0


def buffer_ph(
    acid_concentration: float,
    conjugate_base_concentration: float,
    pka: float,
) -> float:
    """Return buffer pH from the Henderson-Hasselbalch equation."""
    _require_positive(acid_concentration, "acid_concentration")
    _require_positive(conjugate_base_concentration, "conjugate_base_concentration")
    return pka + math.log10(conjugate_base_concentration / acid_concentration)


def ion_product(
    concentrations: Mapping[str, float],
    ion_stoichiometry: Mapping[str, float],
) -> float:
    """Return Qsp for a precipitation or dissolution expression."""
    product = 1.0
    for ion, coefficient in _positive_coefficients(ion_stoichiometry).items():
        if ion not in concentrations:
            raise ValueError(f"Missing concentration for {ion}.")
        concentration = concentrations[ion]
        _require_nonnegative(concentration, f"concentration for {ion}")
        if concentration == 0:
            return 0.0
        product *= concentration**coefficient
    return product


def ksp_from_molar_solubility(
    molar_solubility: float,
    ion_stoichiometry: Mapping[str, float],
) -> float:
    """Return Ksp from pure-water molar solubility and dissolution stoichiometry."""
    _require_nonnegative(molar_solubility, "molar_solubility")
    product = 1.0
    for coefficient in _positive_coefficients(ion_stoichiometry).values():
        product *= (coefficient * molar_solubility) ** coefficient
    return product


def molar_solubility_from_ksp(ksp: float, ion_stoichiometry: Mapping[str, float]) -> float:
    """Return pure-water molar solubility from Ksp."""
    _require_positive(ksp, "ksp")
    coefficients = _positive_coefficients(ion_stoichiometry)
    coefficient_product = 1.0
    coefficient_sum = 0.0
    for coefficient in coefficients.values():
        coefficient_product *= coefficient**coefficient
        coefficient_sum += coefficient
    return (ksp / coefficient_product) ** (1.0 / coefficient_sum)


def molar_solubility_with_common_ions(
    ksp: float,
    ion_stoichiometry: Mapping[str, float],
    initial_ion_concentrations: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Return molar solubility in a solution that already contains common ions."""
    _require_positive(ksp, "ksp")
    coefficients = _positive_coefficients(ion_stoichiometry)
    for ion, concentration in initial_ion_concentrations.items():
        if ion in coefficients:
            _require_nonnegative(concentration, f"concentration for {ion}")

    initial_product = ion_product(
        {ion: initial_ion_concentrations.get(ion, 0.0) for ion in coefficients},
        coefficients,
    )
    if initial_product >= ksp:
        return 0.0

    def qsp_at(solubility: float) -> float:
        return ion_product(
            {
                ion: initial_ion_concentrations.get(ion, 0.0) + coefficient * solubility
                for ion, coefficient in coefficients.items()
            },
            coefficients,
        )

    low = 0.0
    high = molar_solubility_from_ksp(ksp, coefficients)
    while qsp_at(high) < ksp:
        high *= 2.0

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        residual = qsp_at(mid) - ksp
        if abs(residual) <= max(abs(ksp) * tolerance, 1e-300):
            return mid
        if residual < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def will_precipitate(
    concentrations: Mapping[str, float],
    ksp: float,
    ion_stoichiometry: Mapping[str, float],
    tolerance: float = 1e-12,
) -> PrecipitationResult:
    """Return whether Qsp exceeds Ksp."""
    _require_positive(ksp, "ksp")
    qsp = ion_product(concentrations, ion_stoichiometry)
    return PrecipitationResult(qsp, ksp, qsp > ksp and not math.isclose(qsp, ksp, rel_tol=tolerance))


def precipitation_threshold_concentration(
    ksp: float,
    counter_ion_concentration: float,
    *,
    counter_ion_coefficient: float = 1.0,
    target_ion_coefficient: float = 1.0,
) -> float:
    """Return target-ion concentration at the start of precipitation."""
    _require_positive(ksp, "ksp")
    _require_positive(counter_ion_concentration, "counter_ion_concentration")
    _require_positive(counter_ion_coefficient, "counter_ion_coefficient")
    _require_positive(target_ion_coefficient, "target_ion_coefficient")
    denominator = counter_ion_concentration**counter_ion_coefficient
    return (ksp / denominator) ** (1.0 / target_ion_coefficient)


def formation_constant(
    free_metal_concentration: float,
    free_ligand_concentration: float,
    complex_concentration: float,
    ligand_coefficient: float = 1.0,
) -> float:
    """Return Kf for M + nL <=> MLn."""
    _require_positive(free_metal_concentration, "free_metal_concentration")
    _require_positive(free_ligand_concentration, "free_ligand_concentration")
    _require_nonnegative(complex_concentration, "complex_concentration")
    _require_positive(ligand_coefficient, "ligand_coefficient")
    return complex_concentration / (
        free_metal_concentration * free_ligand_concentration**ligand_coefficient
    )


def free_metal_concentration(
    complex_concentration: float,
    free_ligand_concentration: float,
    formation_constant_value: float,
    ligand_coefficient: float = 1.0,
) -> float:
    """Return free metal concentration from Kf, complex concentration, and ligand."""
    _require_nonnegative(complex_concentration, "complex_concentration")
    _require_positive(free_ligand_concentration, "free_ligand_concentration")
    _require_positive(formation_constant_value, "formation_constant_value")
    _require_positive(ligand_coefficient, "ligand_coefficient")
    return complex_concentration / (
        formation_constant_value * free_ligand_concentration**ligand_coefficient
    )


def complex_concentration_from_free(
    free_metal_concentration_value: float,
    free_ligand_concentration: float,
    formation_constant_value: float,
    ligand_coefficient: float = 1.0,
) -> float:
    """Return complex concentration from free metal, free ligand, and Kf."""
    _require_nonnegative(free_metal_concentration_value, "free_metal_concentration_value")
    _require_positive(free_ligand_concentration, "free_ligand_concentration")
    _require_positive(formation_constant_value, "formation_constant_value")
    _require_positive(ligand_coefficient, "ligand_coefficient")
    return (
        formation_constant_value
        * free_metal_concentration_value
        * free_ligand_concentration**ligand_coefficient
    )


def delta_g(
    delta_g_standard_j_per_mol: float,
    reaction_quotient_value: float,
    temperature_k: float,
) -> float:
    """Return Delta G in J/mol from Delta G standard, Q, and temperature."""
    _require_positive(reaction_quotient_value, "reaction_quotient_value")
    _require_positive(temperature_k, "temperature_k")
    return delta_g_standard_j_per_mol + R_J_PER_MOL_K * temperature_k * math.log(reaction_quotient_value)


def equilibrium_constant_from_delta_g_standard(
    delta_g_standard_j_per_mol: float,
    temperature_k: float,
) -> float:
    """Return K from Delta G standard in J/mol."""
    _require_positive(temperature_k, "temperature_k")
    return math.exp(-delta_g_standard_j_per_mol / (R_J_PER_MOL_K * temperature_k))


def delta_g_standard_from_equilibrium_constant(
    equilibrium_constant: float,
    temperature_k: float,
) -> float:
    """Return Delta G standard in J/mol from K."""
    _require_positive(equilibrium_constant, "equilibrium_constant")
    _require_positive(temperature_k, "temperature_k")
    return -R_J_PER_MOL_K * temperature_k * math.log(equilibrium_constant)


def henry_law_concentration(henry_constant: float, gas_pressure: float) -> float:
    """Return dissolved gas concentration from C = kH * P."""
    _require_nonnegative(henry_constant, "henry_constant")
    _require_nonnegative(gas_pressure, "gas_pressure")
    return henry_constant * gas_pressure


def henry_law_pressure(concentration: float, henry_constant: float) -> float:
    """Return gas pressure from C = kH * P."""
    _require_nonnegative(concentration, "concentration")
    _require_positive(henry_constant, "henry_constant")
    return concentration / henry_constant


def _normalized_initial_concentrations(
    initial_concentrations: Mapping[str, float],
    stoichiometry: Mapping[str, float],
) -> dict[str, float]:
    initial = {species: float(concentration) for species, concentration in initial_concentrations.items()}
    for species, concentration in initial.items():
        _require_nonnegative(concentration, f"initial concentration for {species}")
    for species in stoichiometry:
        initial.setdefault(species, 0.0)
    return initial


def _extent_bounds(
    initial_concentrations: Mapping[str, float],
    stoichiometry: Mapping[str, float],
) -> tuple[float, float]:
    lower = -math.inf
    upper = math.inf
    for species, coefficient in stoichiometry.items():
        if coefficient == 0:
            continue
        concentration = initial_concentrations.get(species, 0.0)
        if coefficient > 0:
            lower = max(lower, -concentration / coefficient)
        else:
            upper = min(upper, concentration / -coefficient)

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("extent_bounds are required when concentration bounds are unbounded.")
    return lower, upper


def _log_reaction_quotient(
    initial_concentrations: Mapping[str, float],
    stoichiometry: Mapping[str, float],
    extent: float,
) -> float:
    log_quotient = 0.0
    for species, coefficient in stoichiometry.items():
        if coefficient == 0:
            continue
        concentration = initial_concentrations.get(species, 0.0) + coefficient * extent
        if concentration < -1e-14:
            raise ValueError("Extent produces a negative concentration.")
        if concentration <= 0:
            return -math.inf if coefficient > 0 else math.inf
        log_quotient += coefficient * math.log(concentration)
    return log_quotient


def _concentrations_at_extent(
    initial_concentrations: Mapping[str, float],
    stoichiometry: Mapping[str, float],
    extent: float,
) -> dict[str, float]:
    concentrations = {}
    for species in set(initial_concentrations) | set(stoichiometry):
        concentration = initial_concentrations.get(species, 0.0) + stoichiometry.get(species, 0.0) * extent
        if concentration < 0:
            if concentration < -1e-12:
                raise ValueError("Extent produces a negative concentration.")
            concentration = 0.0
        concentrations[species] = concentration
    return concentrations


def _equilibrium_result(
    initial_concentrations: Mapping[str, float],
    stoichiometry: Mapping[str, float],
    extent: float,
) -> EquilibriumResult:
    concentrations = _concentrations_at_extent(initial_concentrations, stoichiometry, extent)
    return EquilibriumResult(concentrations, extent, reaction_quotient(concentrations, stoichiometry))


def _positive_coefficients(stoichiometry: Mapping[str, float]) -> dict[str, float]:
    if not stoichiometry:
        raise ValueError("stoichiometry must not be empty.")
    coefficients = {}
    for species, coefficient in stoichiometry.items():
        _require_positive(coefficient, f"coefficient for {species}")
        coefficients[species] = coefficient
    return coefficients


def _same_sign(left: float, right: float) -> bool:
    return (left > 0 and right > 0) or (left < 0 and right < 0)


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
