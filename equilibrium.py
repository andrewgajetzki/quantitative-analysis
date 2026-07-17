"""Chemical-equilibrium, acid-base, solubility, and thermodynamics helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


KW_25C = 1.0e-14
R_J_PER_MOL_K = 8.314_462_618_153_24
R_L_ATM_PER_MOL_K = 0.082_057


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


@dataclass(frozen=True)
class SaltHydrolysisResult:
    """Predicted acidity/basicity of a salt from hydrolyzing ions."""

    character: str
    cation_ka: float | None = None
    anion_kb: float | None = None


@dataclass(frozen=True)
class BufferAdjustment:
    """Strong reagent needed to move a buffer to a target pH."""

    reagent: str
    amount: float
    acid_amount: float
    conjugate_base_amount: float
    ph: float


@dataclass(frozen=True)
class ChargeBalanceResult:
    """Charge-balance accounting in equivalents per liter."""

    positive_charge: float
    negative_charge: float
    residual: float
    balanced: bool


@dataclass(frozen=True)
class MassBalanceResult:
    """Mass-balance accounting for an analytical concentration."""

    total_concentration: float
    accounted_concentration: float
    residual: float
    balanced: bool


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


def activity(concentration: float, activity_coefficient: float) -> float:
    """Return activity from concentration and activity coefficient."""
    _require_nonnegative(concentration, "concentration")
    _require_positive(activity_coefficient, "activity_coefficient")
    return concentration * activity_coefficient


def activities(
    concentrations: Mapping[str, float],
    activity_coefficients: Mapping[str, float],
) -> dict[str, float]:
    """Return activities for a set of concentrations."""
    return {
        species: activity(concentration, activity_coefficients.get(species, 1.0))
        for species, concentration in concentrations.items()
    }


def activity_reaction_quotient(
    concentrations: Mapping[str, float],
    stoichiometry: Mapping[str, float],
    activity_coefficients: Mapping[str, float] | None = None,
) -> float:
    """Return Q using activities instead of bare concentrations."""
    coefficients = activity_coefficients or {}
    return reaction_quotient(activities(concentrations, coefficients), stoichiometry)


def activity_coefficient_product(
    stoichiometry: Mapping[str, float],
    activity_coefficients: Mapping[str, float],
) -> float:
    """Return the gamma term that converts concentration quotient to activity quotient."""
    product = 1.0
    for species, coefficient in stoichiometry.items():
        if coefficient == 0:
            continue
        gamma = activity_coefficients.get(species, 1.0)
        _require_positive(gamma, f"activity coefficient for {species}")
        product *= gamma**coefficient
    return product


def concentration_equilibrium_constant(
    thermodynamic_equilibrium_constant: float,
    stoichiometry: Mapping[str, float],
    activity_coefficients: Mapping[str, float],
) -> float:
    """Return concentration-based K from thermodynamic K and activity coefficients."""
    _require_positive(thermodynamic_equilibrium_constant, "thermodynamic_equilibrium_constant")
    return thermodynamic_equilibrium_constant / activity_coefficient_product(
        stoichiometry,
        activity_coefficients,
    )


def thermodynamic_equilibrium_constant(
    concentration_equilibrium_constant_value: float,
    stoichiometry: Mapping[str, float],
    activity_coefficients: Mapping[str, float],
) -> float:
    """Return thermodynamic K from concentration-based K and activity coefficients."""
    _require_positive(concentration_equilibrium_constant_value, "concentration_equilibrium_constant_value")
    return concentration_equilibrium_constant_value * activity_coefficient_product(
        stoichiometry,
        activity_coefficients,
    )


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


def equilibrium_expression(stoichiometry: Mapping[str, float]) -> str:
    """Return a compact concentration-expression string for K or Q.

    Positive coefficients are products, negative coefficients are reactants.
    Omit pure solids and liquids from ``stoichiometry`` before calling.
    """
    numerator = []
    denominator = []
    for species, coefficient in stoichiometry.items():
        if coefficient > 0:
            numerator.append(_expression_factor(species, coefficient))
        elif coefficient < 0:
            denominator.append(_expression_factor(species, -coefficient))
    if not numerator:
        numerator = ["1"]
    if not denominator:
        return " ".join(numerator)
    return f"{' '.join(numerator)} / {' '.join(denominator)}"


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


def delta_gas_moles(stoichiometry: Mapping[str, float]) -> float:
    """Return change in moles of gas from gas-phase stoichiometry."""
    return sum(stoichiometry.values())


def kp_from_kc(
    kc: float,
    delta_moles_gas: float,
    temperature_k: float,
    gas_constant_l_atm: float = R_L_ATM_PER_MOL_K,
) -> float:
    """Return Kp from Kc using Kp = Kc(RT)^delta_n."""
    _require_positive(kc, "kc")
    _require_positive(temperature_k, "temperature_k")
    _require_positive(gas_constant_l_atm, "gas_constant_l_atm")
    return kc * (gas_constant_l_atm * temperature_k) ** delta_moles_gas


def kc_from_kp(
    kp: float,
    delta_moles_gas: float,
    temperature_k: float,
    gas_constant_l_atm: float = R_L_ATM_PER_MOL_K,
) -> float:
    """Return Kc from Kp using Kc = Kp/(RT)^delta_n."""
    _require_positive(kp, "kp")
    _require_positive(temperature_k, "temperature_k")
    _require_positive(gas_constant_l_atm, "gas_constant_l_atm")
    return kp / (gas_constant_l_atm * temperature_k) ** delta_moles_gas


def partial_pressures_from_moles(moles: Mapping[str, float], total_pressure: float) -> dict[str, float]:
    """Return gas partial pressures from mole amounts and total pressure."""
    _require_positive(total_pressure, "total_pressure")
    total_moles = 0.0
    for species, amount in moles.items():
        _require_nonnegative(amount, f"moles for {species}")
        total_moles += amount
    _require_positive(total_moles, "total gas moles")
    return {species: amount / total_moles * total_pressure for species, amount in moles.items()}


def pressure_reaction_quotient(
    partial_pressures: Mapping[str, float],
    stoichiometry: Mapping[str, float],
) -> float:
    """Return Qp from gas partial pressures."""
    return reaction_quotient(partial_pressures, stoichiometry)


def pressure_change_shift(stoichiometry: Mapping[str, float], pressure_increases: bool) -> str:
    """Return Le Chatelier shift for a pressure change in a gas reaction."""
    product_moles = sum(coefficient for coefficient in stoichiometry.values() if coefficient > 0)
    reactant_moles = sum(-coefficient for coefficient in stoichiometry.values() if coefficient < 0)
    if math.isclose(product_moles, reactant_moles):
        return "no_shift"
    if pressure_increases:
        return "forward" if product_moles < reactant_moles else "reverse"
    return "forward" if product_moles > reactant_moles else "reverse"


def temperature_change_shift(delta_h_j_per_mol: float, temperature_increases: bool) -> str:
    """Return Le Chatelier shift for heating or cooling a reaction."""
    if math.isclose(delta_h_j_per_mol, 0.0):
        return "no_shift"
    if temperature_increases:
        return "forward" if delta_h_j_per_mol > 0 else "reverse"
    return "forward" if delta_h_j_per_mol < 0 else "reverse"


def ionic_strength(
    concentrations: Mapping[str, float],
    charges: Mapping[str, float],
) -> float:
    """Return ionic strength, I = 1/2 sum(c_i z_i^2)."""
    strength = 0.0
    for species, concentration in concentrations.items():
        if species not in charges:
            raise ValueError(f"Missing charge for {species}.")
        _require_nonnegative(concentration, f"concentration for {species}")
        strength += concentration * charges[species] ** 2
    return 0.5 * strength


def debye_huckel_a_parameter(
    temperature_k: float = 298.15,
    dielectric_constant: float = 78.54,
) -> float:
    """Return the Debye-Huckel A parameter for water-like solvents."""
    _require_positive(temperature_k, "temperature_k")
    _require_positive(dielectric_constant, "dielectric_constant")
    return 1.825e6 / (dielectric_constant * temperature_k) ** 1.5


def debye_huckel_b_parameter_pm(
    temperature_k: float = 298.15,
    dielectric_constant: float = 78.54,
) -> float:
    """Return the Debye-Huckel B parameter for ion sizes in picometers."""
    _require_positive(temperature_k, "temperature_k")
    _require_positive(dielectric_constant, "dielectric_constant")
    return 0.5029 / math.sqrt(dielectric_constant * temperature_k)


def debye_huckel_log10_activity_coefficient(
    charge: float,
    ionic_strength_value: float,
    ion_size_pm: float | None = None,
    *,
    temperature_k: float = 298.15,
    dielectric_constant: float = 78.54,
) -> float:
    """Return log10(gamma) from the extended Debye-Huckel equation."""
    _require_nonnegative(ionic_strength_value, "ionic_strength_value")
    if math.isclose(charge, 0.0) or math.isclose(ionic_strength_value, 0.0):
        return 0.0
    sqrt_strength = math.sqrt(ionic_strength_value)
    denominator = 1.0
    if ion_size_pm is not None:
        _require_nonnegative(ion_size_pm, "ion_size_pm")
        denominator += (
            debye_huckel_b_parameter_pm(temperature_k, dielectric_constant)
            * ion_size_pm
            * sqrt_strength
        )
    return -(
        debye_huckel_a_parameter(temperature_k, dielectric_constant)
        * charge**2
        * sqrt_strength
        / denominator
    )


def debye_huckel_activity_coefficient(
    charge: float,
    ionic_strength_value: float,
    ion_size_pm: float | None = None,
    *,
    temperature_k: float = 298.15,
    dielectric_constant: float = 78.54,
) -> float:
    """Return gamma from the extended Debye-Huckel equation."""
    return 10.0 ** debye_huckel_log10_activity_coefficient(
        charge,
        ionic_strength_value,
        ion_size_pm,
        temperature_k=temperature_k,
        dielectric_constant=dielectric_constant,
    )


def debye_huckel_activity_coefficients(
    concentrations: Mapping[str, float],
    charges: Mapping[str, float],
    ion_sizes_pm: Mapping[str, float] | None = None,
    *,
    temperature_k: float = 298.15,
    dielectric_constant: float = 78.54,
) -> dict[str, float]:
    """Return activity coefficients for all species from solution ionic strength."""
    strength = ionic_strength(concentrations, charges)
    sizes = ion_sizes_pm or {}
    return {
        species: debye_huckel_activity_coefficient(
            charge,
            strength,
            sizes.get(species),
            temperature_k=temperature_k,
            dielectric_constant=dielectric_constant,
        )
        for species, charge in charges.items()
    }


def interpolate_activity_coefficient(
    ionic_strength_value: float,
    table_values: Mapping[float, float],
) -> float:
    """Linearly interpolate gamma from tabulated ionic-strength values."""
    _require_nonnegative(ionic_strength_value, "ionic_strength_value")
    if not table_values:
        raise ValueError("table_values must not be empty.")
    points = sorted(table_values.items())
    for strength, coefficient in points:
        _require_nonnegative(strength, "tabulated ionic strength")
        _require_positive(coefficient, "tabulated activity coefficient")
    if ionic_strength_value <= points[0][0]:
        return points[0][1]
    if ionic_strength_value >= points[-1][0]:
        return points[-1][1]
    for (left_strength, left_gamma), (right_strength, right_gamma) in zip(points, points[1:]):
        if left_strength <= ionic_strength_value <= right_strength:
            fraction = (ionic_strength_value - left_strength) / (right_strength - left_strength)
            return left_gamma + fraction * (right_gamma - left_gamma)
    raise RuntimeError("Interpolation failed.")


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


def hydrogen_activity_from_ph(ph: float) -> float:
    """Return hydrogen-ion activity from pH."""
    return hydrogen_from_ph(ph)


def ph_from_hydrogen_activity(hydrogen_activity: float) -> float:
    """Return pH from hydrogen-ion activity."""
    _require_positive(hydrogen_activity, "hydrogen_activity")
    return -math.log10(hydrogen_activity)


def ph_from_hydrogen_concentration_activity(
    hydrogen_concentration: float,
    hydrogen_activity_coefficient: float,
) -> float:
    """Return pH from [H+] and gamma_H."""
    return ph_from_hydrogen_activity(
        activity(hydrogen_concentration, hydrogen_activity_coefficient)
    )


def hydroxide_from_poh(poh: float) -> float:
    """Return hydroxide-ion concentration from pOH."""
    return 10.0**(-poh)


def hydroxide_activity_from_poh(poh: float) -> float:
    """Return hydroxide-ion activity from pOH."""
    return hydroxide_from_poh(poh)


def poh_from_hydroxide_activity(hydroxide_activity: float) -> float:
    """Return pOH from hydroxide-ion activity."""
    _require_positive(hydroxide_activity, "hydroxide_activity")
    return -math.log10(hydroxide_activity)


def ph_from_hydroxide_activity(hydroxide_activity: float, kw: float = KW_25C) -> float:
    """Return pH from hydroxide activity and Kw."""
    _require_positive(hydroxide_activity, "hydroxide_activity")
    _require_positive(kw, "kw")
    return pkw(kw) + math.log10(hydroxide_activity)


def ph_from_hydroxide_concentration_activity(
    hydroxide_concentration: float,
    hydroxide_activity_coefficient: float,
    kw: float = KW_25C,
) -> float:
    """Return pH from [OH-], gamma_OH, and Kw."""
    return ph_from_hydroxide_activity(
        activity(hydroxide_concentration, hydroxide_activity_coefficient),
        kw,
    )


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


def neutral_ph(kw: float = KW_25C) -> float:
    """Return the neutral pH for a given Kw."""
    return pkw(kw) / 2.0


def neutral_poh(kw: float = KW_25C) -> float:
    """Return the neutral pOH for a given Kw."""
    return pkw(kw) / 2.0


def classify_ph(ph: float, kw: float = KW_25C, tolerance: float = 1e-12) -> str:
    """Classify pH as acidic, basic, or neutral for a given Kw."""
    neutral = neutral_ph(kw)
    if math.isclose(ph, neutral, rel_tol=tolerance, abs_tol=tolerance):
        return "neutral"
    if ph < neutral:
        return "acidic"
    return "basic"


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


def hydrogen_from_strong_acid(
    concentration: float,
    acidic_protons: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return [H+] for a fully dissociated acid, including water autoionization."""
    _require_nonnegative(concentration, "concentration")
    _require_positive(acidic_protons, "acidic_protons")
    _require_positive(kw, "kw")
    strong_acid_concentration = concentration * acidic_protons
    return (
        strong_acid_concentration
        + math.sqrt(strong_acid_concentration**2 + 4.0 * kw)
    ) / 2.0


def hydroxide_from_strong_base(
    concentration: float,
    hydroxides: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return [OH-] for a fully dissociated base, including water autoionization."""
    _require_nonnegative(concentration, "concentration")
    _require_positive(hydroxides, "hydroxides")
    _require_positive(kw, "kw")
    strong_base_concentration = concentration * hydroxides
    return (
        strong_base_concentration
        + math.sqrt(strong_base_concentration**2 + 4.0 * kw)
    ) / 2.0


def strong_acid_ph(
    concentration: float,
    acidic_protons: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return pH for a strong acid that fully dissociates."""
    return ph_from_hydrogen(hydrogen_from_strong_acid(concentration, acidic_protons, kw))


def strong_base_ph(
    concentration: float,
    hydroxides: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return pH for a strong base that fully dissociates."""
    hydroxide = hydroxide_from_strong_base(concentration, hydroxides, kw)
    return ph_from_poh(poh_from_hydroxide(hydroxide), kw)


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


def proton_transfer_equilibrium_constant(
    reactant_acid_pka: float,
    product_acid_pka: float,
) -> float:
    """Return K for HA + B <=> A- + HB+ from the two acid pKa values."""
    return 10.0 ** (product_acid_pka - reactant_acid_pka)


def proton_transfer_direction(
    reactant_acid_pka: float,
    product_acid_pka: float,
    tolerance: float = 1e-12,
) -> str:
    """Return favored direction for HA + B <=> A- + HB+ from pKa values."""
    equilibrium_constant = proton_transfer_equilibrium_constant(
        reactant_acid_pka,
        product_acid_pka,
    )
    return equilibrium_direction(1.0, equilibrium_constant, tolerance)


def conjugate_base_hydrolysis_constants(
    acid_dissociation_constants: list[float] | tuple[float, ...],
    kw: float = KW_25C,
) -> tuple[float, ...]:
    """Return Kb values for a polyprotic acid's fully deprotonated conjugate base."""
    _require_positive(kw, "kw")
    constants = []
    for ka in reversed(acid_dissociation_constants):
        _require_positive(ka, "acid_dissociation_constant")
        constants.append(kw / ka)
    return tuple(constants)


def salt_solution_character(
    cation_ka: float | None = None,
    anion_kb: float | None = None,
    *,
    tolerance: float = 1e-12,
) -> SaltHydrolysisResult:
    """Classify a salt solution from acidic cation and basic anion constants.

    Pass ``None`` for ions that do not hydrolyze appreciably, such as alkali
    cations or conjugate bases of strong acids.
    """
    if cation_ka is not None:
        _require_nonnegative(cation_ka, "cation_ka")
    if anion_kb is not None:
        _require_nonnegative(anion_kb, "anion_kb")

    acid_strength = cation_ka or 0.0
    base_strength = anion_kb or 0.0
    if math.isclose(acid_strength, base_strength, rel_tol=tolerance, abs_tol=tolerance):
        character = "neutral"
    elif acid_strength > base_strength:
        character = "acidic"
    else:
        character = "basic"
    return SaltHydrolysisResult(character, cation_ka, anion_kb)


def weak_acid_hydrogen_concentration(formal_concentration: float, ka: float) -> float:
    """Return [H+] for HA <=> H+ + A- using the exact quadratic solution."""
    _require_positive(formal_concentration, "formal_concentration")
    _require_positive(ka, "ka")
    return (-ka + math.sqrt(ka**2 + 4.0 * ka * formal_concentration)) / 2.0


def weak_acid_ph(formal_concentration: float, ka: float) -> float:
    """Return pH for a monoprotic weak acid."""
    return ph_from_hydrogen(weak_acid_hydrogen_concentration(formal_concentration, ka))


def weak_acid_fraction_dissociated(formal_concentration: float, ka: float) -> float:
    """Return the dissociated fraction alpha for a monoprotic weak acid."""
    return weak_acid_hydrogen_concentration(formal_concentration, ka) / formal_concentration


def weak_acid_ka_from_fraction_dissociated(
    formal_concentration: float,
    fraction_dissociated: float,
    hydrogen_activity_coefficient: float = 1.0,
    conjugate_base_activity_coefficient: float = 1.0,
    acid_activity_coefficient: float = 1.0,
) -> float:
    """Return thermodynamic Ka from formal concentration and acid dissociation fraction."""
    _require_positive(formal_concentration, "formal_concentration")
    _require_fraction_between_zero_and_one(fraction_dissociated, "fraction_dissociated")
    coefficients = {
        "H+": hydrogen_activity_coefficient,
        "A-": conjugate_base_activity_coefficient,
        "HA": acid_activity_coefficient,
    }
    concentration_ka = formal_concentration * fraction_dissociated**2 / (
        1.0 - fraction_dissociated
    )
    return thermodynamic_equilibrium_constant(
        concentration_ka,
        {"H+": 1, "A-": 1, "HA": -1},
        coefficients,
    )


def weak_acid_ka_from_ph(
    formal_concentration: float,
    ph: float,
    hydrogen_activity_coefficient: float = 1.0,
    conjugate_base_activity_coefficient: float = 1.0,
    acid_activity_coefficient: float = 1.0,
) -> float:
    """Return thermodynamic Ka for HA from formal concentration and measured pH."""
    _require_positive(formal_concentration, "formal_concentration")
    _require_positive(hydrogen_activity_coefficient, "hydrogen_activity_coefficient")
    _require_positive(conjugate_base_activity_coefficient, "conjugate_base_activity_coefficient")
    _require_positive(acid_activity_coefficient, "acid_activity_coefficient")
    hydrogen_activity = hydrogen_activity_from_ph(ph)
    hydrogen_concentration = hydrogen_activity / hydrogen_activity_coefficient
    if hydrogen_concentration >= formal_concentration:
        raise ValueError("hydrogen concentration from pH must be less than formal_concentration.")
    conjugate_base_activity = conjugate_base_activity_coefficient * hydrogen_concentration
    acid_activity = acid_activity_coefficient * (formal_concentration - hydrogen_concentration)
    return hydrogen_activity * conjugate_base_activity / acid_activity


def weak_acid_pka_from_ph(
    formal_concentration: float,
    ph: float,
    hydrogen_activity_coefficient: float = 1.0,
    conjugate_base_activity_coefficient: float = 1.0,
    acid_activity_coefficient: float = 1.0,
) -> float:
    """Return pKa for HA from formal concentration and measured pH."""
    return pka_from_ka(
        weak_acid_ka_from_ph(
            formal_concentration,
            ph,
            hydrogen_activity_coefficient,
            conjugate_base_activity_coefficient,
            acid_activity_coefficient,
        )
    )


def weak_acid_hydrogen_concentration_with_activity(
    formal_concentration: float,
    ka: float,
    hydrogen_activity_coefficient: float,
    conjugate_base_activity_coefficient: float,
    acid_activity_coefficient: float = 1.0,
) -> float:
    """Return [H+] for HA using activity-corrected Ka."""
    concentration_ka = concentration_equilibrium_constant(
        ka,
        {"H+": 1, "A-": 1, "HA": -1},
        {
            "H+": hydrogen_activity_coefficient,
            "A-": conjugate_base_activity_coefficient,
            "HA": acid_activity_coefficient,
        },
    )
    return weak_acid_hydrogen_concentration(formal_concentration, concentration_ka)


def weak_acid_ph_with_activity(
    formal_concentration: float,
    ka: float,
    hydrogen_activity_coefficient: float,
    conjugate_base_activity_coefficient: float,
    acid_activity_coefficient: float = 1.0,
) -> float:
    """Return pH for a weak acid using activity-corrected Ka."""
    hydrogen_concentration = weak_acid_hydrogen_concentration_with_activity(
        formal_concentration,
        ka,
        hydrogen_activity_coefficient,
        conjugate_base_activity_coefficient,
        acid_activity_coefficient,
    )
    return ph_from_hydrogen_concentration_activity(
        hydrogen_concentration,
        hydrogen_activity_coefficient,
    )


def weak_base_hydroxide_concentration(formal_concentration: float, kb: float) -> float:
    """Return [OH-] for B + H2O <=> BH+ + OH- using the exact quadratic solution."""
    _require_positive(formal_concentration, "formal_concentration")
    _require_positive(kb, "kb")
    return (-kb + math.sqrt(kb**2 + 4.0 * kb * formal_concentration)) / 2.0


def weak_base_ph(formal_concentration: float, kb: float, kw: float = KW_25C) -> float:
    """Return pH for a weak base."""
    hydroxide = weak_base_hydroxide_concentration(formal_concentration, kb)
    return ph_from_poh(poh_from_hydroxide(hydroxide), kw)


def weak_base_fraction_protonated(formal_concentration: float, kb: float) -> float:
    """Return the fraction of a weak base converted to conjugate acid."""
    return weak_base_hydroxide_concentration(formal_concentration, kb) / formal_concentration


def weak_base_kb_from_fraction_protonated(
    formal_concentration: float,
    fraction_protonated: float,
    conjugate_acid_activity_coefficient: float = 1.0,
    hydroxide_activity_coefficient: float = 1.0,
    base_activity_coefficient: float = 1.0,
) -> float:
    """Return thermodynamic Kb from formal concentration and base protonation fraction."""
    _require_positive(formal_concentration, "formal_concentration")
    _require_fraction_between_zero_and_one(fraction_protonated, "fraction_protonated")
    coefficients = {
        "BH+": conjugate_acid_activity_coefficient,
        "OH-": hydroxide_activity_coefficient,
        "B": base_activity_coefficient,
    }
    concentration_kb = formal_concentration * fraction_protonated**2 / (
        1.0 - fraction_protonated
    )
    return thermodynamic_equilibrium_constant(
        concentration_kb,
        {"BH+": 1, "OH-": 1, "B": -1},
        coefficients,
    )


def weak_base_kb_from_ph(
    formal_concentration: float,
    ph: float,
    conjugate_acid_activity_coefficient: float = 1.0,
    hydroxide_activity_coefficient: float = 1.0,
    base_activity_coefficient: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return thermodynamic Kb for B from formal concentration and measured pH."""
    _require_positive(formal_concentration, "formal_concentration")
    _require_positive(conjugate_acid_activity_coefficient, "conjugate_acid_activity_coefficient")
    _require_positive(hydroxide_activity_coefficient, "hydroxide_activity_coefficient")
    _require_positive(base_activity_coefficient, "base_activity_coefficient")
    hydroxide_activity = hydroxide_activity_from_poh(poh_from_ph(ph, kw))
    hydroxide_concentration = hydroxide_activity / hydroxide_activity_coefficient
    if hydroxide_concentration >= formal_concentration:
        raise ValueError("hydroxide concentration from pH must be less than formal_concentration.")
    conjugate_acid_activity = conjugate_acid_activity_coefficient * hydroxide_concentration
    base_activity = base_activity_coefficient * (formal_concentration - hydroxide_concentration)
    return conjugate_acid_activity * hydroxide_activity / base_activity


def weak_base_pkb_from_ph(
    formal_concentration: float,
    ph: float,
    conjugate_acid_activity_coefficient: float = 1.0,
    hydroxide_activity_coefficient: float = 1.0,
    base_activity_coefficient: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return pKb for B from formal concentration and measured pH."""
    return pkb_from_kb(
        weak_base_kb_from_ph(
            formal_concentration,
            ph,
            conjugate_acid_activity_coefficient,
            hydroxide_activity_coefficient,
            base_activity_coefficient,
            kw,
        )
    )


def weak_base_hydroxide_concentration_with_activity(
    formal_concentration: float,
    kb: float,
    conjugate_acid_activity_coefficient: float,
    hydroxide_activity_coefficient: float,
    base_activity_coefficient: float = 1.0,
) -> float:
    """Return [OH-] for B using activity-corrected Kb."""
    concentration_kb = concentration_equilibrium_constant(
        kb,
        {"BH+": 1, "OH-": 1, "B": -1},
        {
            "BH+": conjugate_acid_activity_coefficient,
            "OH-": hydroxide_activity_coefficient,
            "B": base_activity_coefficient,
        },
    )
    return weak_base_hydroxide_concentration(formal_concentration, concentration_kb)


def weak_base_ph_with_activity(
    formal_concentration: float,
    kb: float,
    conjugate_acid_activity_coefficient: float,
    hydroxide_activity_coefficient: float,
    base_activity_coefficient: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return pH for a weak base using activity-corrected Kb."""
    hydroxide_concentration = weak_base_hydroxide_concentration_with_activity(
        formal_concentration,
        kb,
        conjugate_acid_activity_coefficient,
        hydroxide_activity_coefficient,
        base_activity_coefficient,
    )
    return ph_from_hydroxide_concentration_activity(
        hydroxide_concentration,
        hydroxide_activity_coefficient,
        kw,
    )


def percent_ionization_weak_acid(formal_concentration: float, ka: float) -> float:
    """Return percent ionization for a monoprotic weak acid."""
    return weak_acid_fraction_dissociated(formal_concentration, ka) * 100.0


def percent_ionization_weak_base(formal_concentration: float, kb: float) -> float:
    """Return percent ionization for a weak base."""
    return weak_base_fraction_protonated(formal_concentration, kb) * 100.0


def activity_coefficient_from_ph(concentration: float, ph: float) -> float:
    """Return gamma_H from an analytical [H+] and measured pH."""
    _require_positive(concentration, "concentration")
    return hydrogen_activity_from_ph(ph) / concentration


def conjugate_base_ph(formal_concentration: float, ka: float, kw: float = KW_25C) -> float:
    """Return pH for a salt containing A-, the conjugate base of HA."""
    return weak_base_ph(formal_concentration, conjugate_kb(ka, kw), kw)


def conjugate_acid_ph(formal_concentration: float, kb: float, kw: float = KW_25C) -> float:
    """Return pH for a salt containing BH+, the conjugate acid of B."""
    return weak_acid_ph(formal_concentration, conjugate_ka(kb, kw))


def buffer_ph(
    acid_concentration: float,
    conjugate_base_concentration: float,
    pka: float,
    acid_activity_coefficient: float = 1.0,
    conjugate_base_activity_coefficient: float = 1.0,
) -> float:
    """Return buffer pH from the Henderson-Hasselbalch equation."""
    _require_positive(acid_concentration, "acid_concentration")
    _require_positive(conjugate_base_concentration, "conjugate_base_concentration")
    _require_positive(acid_activity_coefficient, "acid_activity_coefficient")
    _require_positive(conjugate_base_activity_coefficient, "conjugate_base_activity_coefficient")
    return pka + math.log10(
        conjugate_base_activity_coefficient
        * conjugate_base_concentration
        / (acid_activity_coefficient * acid_concentration)
    )


def buffer_ph_from_amounts(
    acid_amount: float,
    conjugate_base_amount: float,
    pka: float,
    acid_activity_coefficient: float = 1.0,
    conjugate_base_activity_coefficient: float = 1.0,
) -> float:
    """Return Henderson-Hasselbalch pH from acid/base amounts in the same volume."""
    return buffer_ph(
        acid_amount,
        conjugate_base_amount,
        pka,
        acid_activity_coefficient,
        conjugate_base_activity_coefficient,
    )


def target_buffer_base_acid_ratio(
    target_ph: float,
    pka: float,
    acid_activity_coefficient: float = 1.0,
    conjugate_base_activity_coefficient: float = 1.0,
) -> float:
    """Return [base]/[acid] needed for a target Henderson-Hasselbalch pH."""
    _require_positive(acid_activity_coefficient, "acid_activity_coefficient")
    _require_positive(conjugate_base_activity_coefficient, "conjugate_base_activity_coefficient")
    return (
        10.0 ** (target_ph - pka)
        * acid_activity_coefficient
        / conjugate_base_activity_coefficient
    )


def strong_reagent_for_target_buffer_ph(
    acid_amount: float,
    conjugate_base_amount: float,
    pka: float,
    target_ph: float,
    acid_activity_coefficient: float = 1.0,
    conjugate_base_activity_coefficient: float = 1.0,
) -> BufferAdjustment:
    """Return strong acid/base amount needed to adjust an HA/A- buffer to target pH."""
    _require_nonnegative(acid_amount, "acid_amount")
    _require_nonnegative(conjugate_base_amount, "conjugate_base_amount")
    if acid_amount == 0 and conjugate_base_amount == 0:
        raise ValueError("At least one buffer component amount must be positive.")

    target_ratio = target_buffer_base_acid_ratio(
        target_ph,
        pka,
        acid_activity_coefficient,
        conjugate_base_activity_coefficient,
    )
    if acid_amount == 0:
        current_ratio = math.inf
    else:
        current_ratio = conjugate_base_amount / acid_amount

    if math.isclose(current_ratio, target_ratio, rel_tol=1e-12, abs_tol=1e-15):
        return BufferAdjustment(
            "none",
            0.0,
            acid_amount,
            conjugate_base_amount,
            buffer_ph_from_amounts(
                acid_amount,
                conjugate_base_amount,
                pka,
                acid_activity_coefficient,
                conjugate_base_activity_coefficient,
            ),
        )

    if target_ratio < current_ratio:
        amount = (conjugate_base_amount - target_ratio * acid_amount) / (1.0 + target_ratio)
        acid_after = acid_amount + amount
        base_after = conjugate_base_amount - amount
        reagent = "strong_acid"
    else:
        amount = (target_ratio * acid_amount - conjugate_base_amount) / (1.0 + target_ratio)
        acid_after = acid_amount - amount
        base_after = conjugate_base_amount + amount
        reagent = "strong_base"

    return BufferAdjustment(
        reagent,
        max(amount, 0.0),
        acid_after,
        base_after,
        buffer_ph_from_amounts(
            acid_after,
            base_after,
            pka,
            acid_activity_coefficient,
            conjugate_base_activity_coefficient,
        ),
    )


def buffer_ph_after_strong_acid(
    acid_amount: float,
    conjugate_base_amount: float,
    pka: float,
    strong_acid_amount: float,
    final_volume_l: float | None = None,
    kw: float = KW_25C,
) -> float:
    """Return pH after adding strong acid to an HA/A- buffer."""
    _require_nonnegative(acid_amount, "acid_amount")
    _require_nonnegative(conjugate_base_amount, "conjugate_base_amount")
    _require_nonnegative(strong_acid_amount, "strong_acid_amount")
    ka = ka_from_pka(pka)
    base_after = conjugate_base_amount - strong_acid_amount
    acid_after = acid_amount + min(strong_acid_amount, conjugate_base_amount)
    if base_after > 0:
        return buffer_ph_from_amounts(acid_after, base_after, pka)
    if final_volume_l is None:
        raise ValueError("final_volume_l is required when strong acid exhausts the buffer base.")
    _require_positive(final_volume_l, "final_volume_l")
    if math.isclose(base_after, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return weak_acid_ph(acid_after / final_volume_l, ka)
    return strong_acid_ph(-base_after / final_volume_l, kw=kw)


def buffer_ph_after_strong_base(
    acid_amount: float,
    conjugate_base_amount: float,
    pka: float,
    strong_base_amount: float,
    final_volume_l: float | None = None,
    kw: float = KW_25C,
) -> float:
    """Return pH after adding strong base to an HA/A- buffer."""
    _require_nonnegative(acid_amount, "acid_amount")
    _require_nonnegative(conjugate_base_amount, "conjugate_base_amount")
    _require_nonnegative(strong_base_amount, "strong_base_amount")
    ka = ka_from_pka(pka)
    acid_after = acid_amount - strong_base_amount
    base_after = conjugate_base_amount + min(strong_base_amount, acid_amount)
    if acid_after > 0:
        return buffer_ph_from_amounts(acid_after, base_after, pka)
    if final_volume_l is None:
        raise ValueError("final_volume_l is required when strong base exhausts the buffer acid.")
    _require_positive(final_volume_l, "final_volume_l")
    if math.isclose(acid_after, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return conjugate_base_ph(base_after / final_volume_l, ka, kw)
    return strong_base_ph(-acid_after / final_volume_l, kw=kw)


def strong_acid_base_mixture_ph(
    strong_acid_amount: float,
    strong_base_amount: float,
    final_volume_l: float,
    acidic_protons: float = 1.0,
    hydroxides: float = 1.0,
    kw: float = KW_25C,
) -> float:
    """Return pH after mixing fully dissociated strong acid and strong base."""
    _require_nonnegative(strong_acid_amount, "strong_acid_amount")
    _require_nonnegative(strong_base_amount, "strong_base_amount")
    _require_positive(final_volume_l, "final_volume_l")
    _require_positive(acidic_protons, "acidic_protons")
    _require_positive(hydroxides, "hydroxides")
    acid_equivalents = strong_acid_amount * acidic_protons
    base_equivalents = strong_base_amount * hydroxides
    if math.isclose(acid_equivalents, base_equivalents, rel_tol=1e-12, abs_tol=1e-15):
        return neutral_ph(kw)
    if acid_equivalents > base_equivalents:
        return strong_acid_ph((acid_equivalents - base_equivalents) / final_volume_l, kw=kw)
    return strong_base_ph((base_equivalents - acid_equivalents) / final_volume_l, kw=kw)


def weak_acid_strong_base_mixture_ph(
    weak_acid_amount: float,
    strong_base_amount: float,
    final_volume_l: float,
    pka: float,
    kw: float = KW_25C,
) -> float:
    """Return pH after mixing HA with strong base."""
    _require_nonnegative(weak_acid_amount, "weak_acid_amount")
    _require_nonnegative(strong_base_amount, "strong_base_amount")
    _require_positive(final_volume_l, "final_volume_l")
    ka = ka_from_pka(pka)
    if weak_acid_amount == 0:
        return strong_base_ph(strong_base_amount / final_volume_l, kw=kw)
    if strong_base_amount == 0:
        return weak_acid_ph(weak_acid_amount / final_volume_l, ka)
    if math.isclose(weak_acid_amount, strong_base_amount, rel_tol=1e-12, abs_tol=1e-15):
        return conjugate_base_ph(weak_acid_amount / final_volume_l, ka, kw)
    if strong_base_amount < weak_acid_amount:
        return buffer_ph_from_amounts(
            weak_acid_amount - strong_base_amount,
            strong_base_amount,
            pka,
        )
    return strong_base_ph((strong_base_amount - weak_acid_amount) / final_volume_l, kw=kw)


def weak_base_strong_acid_mixture_ph(
    weak_base_amount: float,
    strong_acid_amount: float,
    final_volume_l: float,
    kb: float,
    kw: float = KW_25C,
) -> float:
    """Return pH after mixing B with strong acid."""
    _require_nonnegative(weak_base_amount, "weak_base_amount")
    _require_nonnegative(strong_acid_amount, "strong_acid_amount")
    _require_positive(final_volume_l, "final_volume_l")
    _require_positive(kb, "kb")
    ka = conjugate_ka(kb, kw)
    pka = pka_from_ka(ka)
    if weak_base_amount == 0:
        return strong_acid_ph(strong_acid_amount / final_volume_l, kw=kw)
    if strong_acid_amount == 0:
        return weak_base_ph(weak_base_amount / final_volume_l, kb, kw)
    if math.isclose(weak_base_amount, strong_acid_amount, rel_tol=1e-12, abs_tol=1e-15):
        return weak_acid_ph(weak_base_amount / final_volume_l, ka)
    if strong_acid_amount < weak_base_amount:
        return buffer_ph_from_amounts(
            strong_acid_amount,
            weak_base_amount - strong_acid_amount,
            pka,
        )
    return strong_acid_ph((strong_acid_amount - weak_base_amount) / final_volume_l, kw=kw)


def monoprotic_acid_charge_balance_ph(
    total_acid_concentration: float,
    ka: float,
    strong_cation_concentration: float = 0.0,
    strong_anion_concentration: float = 0.0,
    kw: float = KW_25C,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Return exact pH for an HA/A- system from mass and charge balance."""
    _require_nonnegative(total_acid_concentration, "total_acid_concentration")
    _require_positive(ka, "ka")
    _require_nonnegative(strong_cation_concentration, "strong_cation_concentration")
    _require_nonnegative(strong_anion_concentration, "strong_anion_concentration")
    _require_positive(kw, "kw")

    def residual(hydrogen: float) -> float:
        hydroxide = kw / hydrogen
        conjugate_base = total_acid_concentration * ka / (hydrogen + ka)
        return (
            hydrogen
            + strong_cation_concentration
            - hydroxide
            - conjugate_base
            - strong_anion_concentration
        )

    low = 1.0e-16
    high = max(1.0, total_acid_concentration + strong_anion_concentration + 1.0)
    while residual(low) > 0:
        low /= 10.0
    while residual(high) < 0:
        high *= 10.0

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        value = residual(mid)
        if abs(value) <= tolerance or math.isclose(low, high, rel_tol=tolerance, abs_tol=tolerance):
            return ph_from_hydrogen(mid)
        if value < 0:
            low = mid
        else:
            high = mid
    return ph_from_hydrogen((low + high) / 2.0)


def monoprotic_acid_mixture_ph(
    weak_acid_amount: float,
    conjugate_base_amount: float,
    final_volume_l: float,
    ka: float,
    strong_acid_amount: float = 0.0,
    strong_base_amount: float = 0.0,
    kw: float = KW_25C,
) -> float:
    """Return exact pH for HA/A- mixtures with optional strong acid/base additions."""
    _require_nonnegative(weak_acid_amount, "weak_acid_amount")
    _require_nonnegative(conjugate_base_amount, "conjugate_base_amount")
    _require_nonnegative(strong_acid_amount, "strong_acid_amount")
    _require_nonnegative(strong_base_amount, "strong_base_amount")
    _require_positive(final_volume_l, "final_volume_l")
    total_acid = (weak_acid_amount + conjugate_base_amount) / final_volume_l
    strong_cation = (conjugate_base_amount + strong_base_amount) / final_volume_l
    strong_anion = strong_acid_amount / final_volume_l
    return monoprotic_acid_charge_balance_ph(
        total_acid,
        ka,
        strong_cation,
        strong_anion,
        kw,
    )


def buffer_capacity(
    total_buffer_concentration: float,
    ka: float,
    ph: float,
    kw: float = KW_25C,
    include_water: bool = True,
) -> float:
    """Return buffer capacity, moles of strong base per liter per pH unit."""
    _require_nonnegative(total_buffer_concentration, "total_buffer_concentration")
    _require_positive(ka, "ka")
    _require_positive(kw, "kw")
    hydrogen = hydrogen_from_ph(ph)
    capacity = (
        math.log(10.0)
        * total_buffer_concentration
        * ka
        * hydrogen
        / (ka + hydrogen) ** 2
    )
    if include_water:
        capacity += math.log(10.0) * (hydrogen + kw / hydrogen)
    return capacity


def maximum_buffer_capacity(total_buffer_concentration: float) -> float:
    """Return the maximum analytical HA/A- buffer capacity, reached near pH = pKa."""
    _require_nonnegative(total_buffer_concentration, "total_buffer_concentration")
    return math.log(10.0) * total_buffer_concentration / 4.0


def polyprotic_acid_distribution_fractions(
    hydrogen_concentration: float,
    acid_dissociation_constants: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    """Return alpha fractions for HnA, H(n-1)A, ..., A forms."""
    _require_positive(hydrogen_concentration, "hydrogen_concentration")
    if not acid_dissociation_constants:
        return (1.0,)
    n_constants = len(acid_dissociation_constants)
    terms = []
    cumulative_ka = 1.0
    for deprotonations in range(n_constants + 1):
        if deprotonations:
            ka = acid_dissociation_constants[deprotonations - 1]
            _require_positive(ka, "acid_dissociation_constant")
            cumulative_ka *= ka
        terms.append(cumulative_ka * hydrogen_concentration ** (n_constants - deprotonations))
    denominator = sum(terms)
    return tuple(term / denominator for term in terms)


def polyprotic_species_concentrations(
    total_concentration: float,
    hydrogen_concentration: float,
    acid_dissociation_constants: list[float] | tuple[float, ...],
    species_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return concentrations for a polyprotic acid distribution."""
    _require_nonnegative(total_concentration, "total_concentration")
    fractions = polyprotic_acid_distribution_fractions(
        hydrogen_concentration,
        acid_dissociation_constants,
    )
    names = species_names or tuple(f"alpha{index}" for index in range(len(fractions)))
    if len(names) != len(fractions):
        raise ValueError("species_names length must match the number of acid species.")
    return {name: total_concentration * fraction for name, fraction in zip(names, fractions)}


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


def activity_product(
    concentrations: Mapping[str, float],
    ion_stoichiometry: Mapping[str, float],
    activity_coefficients: Mapping[str, float] | None = None,
) -> float:
    """Return an ion product using ion activities."""
    return activity_reaction_quotient(
        concentrations,
        _positive_coefficients(ion_stoichiometry),
        activity_coefficients,
    )


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


def ksp_from_molar_solubility_with_activity(
    molar_solubility: float,
    ion_stoichiometry: Mapping[str, float],
    activity_coefficients: Mapping[str, float],
) -> float:
    """Return thermodynamic Ksp from solubility and activity coefficients."""
    concentration_ksp = ksp_from_molar_solubility(molar_solubility, ion_stoichiometry)
    return thermodynamic_equilibrium_constant(
        concentration_ksp,
        _positive_coefficients(ion_stoichiometry),
        activity_coefficients,
    )


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


def molar_solubility_from_ksp_with_activity(
    ksp: float,
    ion_stoichiometry: Mapping[str, float],
    activity_coefficients: Mapping[str, float],
) -> float:
    """Return molar solubility from thermodynamic Ksp and fixed activity coefficients."""
    concentration_ksp = concentration_equilibrium_constant(
        ksp,
        _positive_coefficients(ion_stoichiometry),
        activity_coefficients,
    )
    return molar_solubility_from_ksp(concentration_ksp, ion_stoichiometry)


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


def molar_solubility_with_common_ions_and_activity(
    ksp: float,
    ion_stoichiometry: Mapping[str, float],
    initial_ion_concentrations: Mapping[str, float],
    activity_coefficients: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Return molar solubility with common ions and fixed activity coefficients."""
    _require_positive(ksp, "ksp")
    coefficients = _positive_coefficients(ion_stoichiometry)
    for ion, concentration in initial_ion_concentrations.items():
        if ion in coefficients:
            _require_nonnegative(concentration, f"concentration for {ion}")

    initial_product = activity_product(
        {ion: initial_ion_concentrations.get(ion, 0.0) for ion in coefficients},
        coefficients,
        activity_coefficients,
    )
    if initial_product >= ksp:
        return 0.0

    def qsp_at(solubility: float) -> float:
        return activity_product(
            {
                ion: initial_ion_concentrations.get(ion, 0.0) + coefficient * solubility
                for ion, coefficient in coefficients.items()
            },
            coefficients,
            activity_coefficients,
        )

    low = 0.0
    high = molar_solubility_from_ksp_with_activity(ksp, coefficients, activity_coefficients)
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


def will_precipitate_with_activity(
    concentrations: Mapping[str, float],
    ksp: float,
    ion_stoichiometry: Mapping[str, float],
    activity_coefficients: Mapping[str, float],
    tolerance: float = 1e-12,
) -> PrecipitationResult:
    """Return whether the activity product exceeds Ksp."""
    _require_positive(ksp, "ksp")
    qsp = activity_product(concentrations, ion_stoichiometry, activity_coefficients)
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


def successive_to_cumulative_constants(stepwise_constants: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Return cumulative formation constants from stepwise constants."""
    cumulative = []
    running_product = 1.0
    for constant in stepwise_constants:
        _require_positive(constant, "stepwise formation constant")
        running_product *= constant
        cumulative.append(running_product)
    return tuple(cumulative)


def cumulative_to_stepwise_constants(cumulative_constants: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Return stepwise formation constants from cumulative constants."""
    stepwise = []
    previous = 1.0
    for constant in cumulative_constants:
        _require_positive(constant, "cumulative formation constant")
        stepwise.append(constant / previous)
        previous = constant
    return tuple(stepwise)


def complex_distribution_fractions(
    free_ligand_concentration: float,
    cumulative_formation_constants: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    """Return fractions for M, ML, ML2, ... from cumulative beta constants."""
    _require_nonnegative(free_ligand_concentration, "free_ligand_concentration")
    terms = [1.0]
    for ligand_count, beta in enumerate(cumulative_formation_constants, start=1):
        _require_positive(beta, "cumulative formation constant")
        terms.append(beta * free_ligand_concentration**ligand_count)
    denominator = sum(terms)
    return tuple(term / denominator for term in terms)


def complex_species_concentrations(
    total_metal_concentration: float,
    free_ligand_concentration: float,
    cumulative_formation_constants: list[float] | tuple[float, ...],
    species_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return concentrations for M, ML, ML2, ... complex species."""
    _require_nonnegative(total_metal_concentration, "total_metal_concentration")
    fractions = complex_distribution_fractions(
        free_ligand_concentration,
        cumulative_formation_constants,
    )
    names = species_names or tuple(f"ML{index}" if index else "M" for index in range(len(fractions)))
    if len(names) != len(fractions):
        raise ValueError("species_names length must match the number of complex species.")
    return {name: total_metal_concentration * fraction for name, fraction in zip(names, fractions)}


def free_metal_from_total_metal(
    total_metal_concentration: float,
    free_ligand_concentration: float,
    cumulative_formation_constants: list[float] | tuple[float, ...],
) -> float:
    """Return free metal concentration from total metal and cumulative beta constants."""
    return complex_species_concentrations(
        total_metal_concentration,
        free_ligand_concentration,
        cumulative_formation_constants,
    )["M"]


def delta_g(
    delta_g_standard_j_per_mol: float,
    reaction_quotient_value: float,
    temperature_k: float,
) -> float:
    """Return Delta G in J/mol from Delta G standard, Q, and temperature."""
    _require_positive(reaction_quotient_value, "reaction_quotient_value")
    _require_positive(temperature_k, "temperature_k")
    return delta_g_standard_j_per_mol + R_J_PER_MOL_K * temperature_k * math.log(reaction_quotient_value)


def delta_g_from_enthalpy_entropy(
    delta_h_j_per_mol: float,
    delta_s_j_per_mol_k: float,
    temperature_k: float,
) -> float:
    """Return Delta G = Delta H - T Delta S."""
    _require_positive(temperature_k, "temperature_k")
    return delta_h_j_per_mol - temperature_k * delta_s_j_per_mol_k


def spontaneity_from_delta_g(delta_g_j_per_mol: float, tolerance: float = 1e-12) -> str:
    """Return spontaneous direction from Delta G."""
    if math.isclose(delta_g_j_per_mol, 0.0, rel_tol=tolerance, abs_tol=tolerance):
        return "at_equilibrium"
    if delta_g_j_per_mol < 0:
        return "forward"
    return "reverse"


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


def equilibrium_constant_from_delta_h_delta_s(
    delta_h_j_per_mol: float,
    delta_s_j_per_mol_k: float,
    temperature_k: float,
) -> float:
    """Return K from Delta H and Delta S at a temperature."""
    delta_g_standard = delta_g_from_enthalpy_entropy(
        delta_h_j_per_mol,
        delta_s_j_per_mol_k,
        temperature_k,
    )
    return equilibrium_constant_from_delta_g_standard(delta_g_standard, temperature_k)


def equilibrium_constant_at_temperature(
    reference_equilibrium_constant: float,
    reference_temperature_k: float,
    target_temperature_k: float,
    delta_h_j_per_mol: float,
) -> float:
    """Return K at a new temperature from the integrated van't Hoff equation."""
    _require_positive(reference_equilibrium_constant, "reference_equilibrium_constant")
    _require_positive(reference_temperature_k, "reference_temperature_k")
    _require_positive(target_temperature_k, "target_temperature_k")
    exponent = -delta_h_j_per_mol / R_J_PER_MOL_K * (
        1.0 / target_temperature_k - 1.0 / reference_temperature_k
    )
    return reference_equilibrium_constant * math.exp(exponent)


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


def charge_balance(
    concentrations: Mapping[str, float],
    charges: Mapping[str, float],
    tolerance: float = 1e-12,
) -> ChargeBalanceResult:
    """Return positive and negative charge totals for a charge-balance check."""
    positive_charge = 0.0
    negative_charge = 0.0
    for species, concentration in concentrations.items():
        if species not in charges:
            raise ValueError(f"Missing charge for {species}.")
        _require_nonnegative(concentration, f"concentration for {species}")
        charge = charges[species]
        if charge > 0:
            positive_charge += charge * concentration
        elif charge < 0:
            negative_charge += -charge * concentration
    residual = positive_charge - negative_charge
    return ChargeBalanceResult(
        positive_charge,
        negative_charge,
        residual,
        math.isclose(residual, 0.0, rel_tol=tolerance, abs_tol=tolerance),
    )


def charge_balance_residual(
    concentrations: Mapping[str, float],
    charges: Mapping[str, float],
) -> float:
    """Return sum(z_i c_i), which is zero when charge is balanced."""
    return charge_balance(concentrations, charges).residual


def mass_balance(
    total_concentration: float,
    species_concentrations: Mapping[str, float],
    species_coefficients: Mapping[str, float] | None = None,
    tolerance: float = 1e-12,
) -> MassBalanceResult:
    """Return analytical mass-balance accounting for related species."""
    _require_nonnegative(total_concentration, "total_concentration")
    coefficients = species_coefficients or {}
    accounted = 0.0
    for species, concentration in species_concentrations.items():
        _require_nonnegative(concentration, f"concentration for {species}")
        coefficient = coefficients.get(species, 1.0)
        _require_nonnegative(coefficient, f"coefficient for {species}")
        accounted += coefficient * concentration
    residual = total_concentration - accounted
    return MassBalanceResult(
        total_concentration,
        accounted,
        residual,
        math.isclose(residual, 0.0, rel_tol=tolerance, abs_tol=tolerance),
    )


def mass_balance_residual(
    total_concentration: float,
    species_concentrations: Mapping[str, float],
    species_coefficients: Mapping[str, float] | None = None,
) -> float:
    """Return total minus accounted concentration for a mass balance."""
    return mass_balance(
        total_concentration,
        species_concentrations,
        species_coefficients,
    ).residual


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


def _expression_factor(species: str, coefficient: float) -> str:
    factor = f"[{species}]"
    if math.isclose(coefficient, 1.0):
        return factor
    if float(coefficient).is_integer():
        coefficient_text = str(int(coefficient))
    else:
        coefficient_text = str(coefficient)
    return f"{factor}^{coefficient_text}"


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _require_fraction_between_zero_and_one(value: float, name: str) -> None:
    if value <= 0 or value >= 1:
        raise ValueError(f"{name} must be greater than 0 and less than 1.")
