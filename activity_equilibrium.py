"""Activity-corrected acid-base and coupled-equilibrium helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math

from equilibrium import KW_25C


DAVIES_A_25C = 0.509


@dataclass(frozen=True)
class AcidBaseComponent:
    """One analytical acid-base component in a charge-balance calculation."""

    total_concentration: float
    acid_dissociation_constants: tuple[float, ...]
    species_charges: tuple[float, ...]
    species_names: tuple[str, ...]

    @classmethod
    def from_pkas(
        cls,
        total_concentration: float,
        pkas: Iterable[float],
        species_charges: Iterable[float],
        species_names: Iterable[str] | None = None,
    ) -> "AcidBaseComponent":
        """Build a component from pKa values for HnA, H(n-1)A, ..., A."""
        constants = tuple(10.0 ** (-pka) for pka in pkas)
        charges = tuple(species_charges)
        names = tuple(species_names or (f"species_{index}" for index in range(len(charges))))
        return cls(total_concentration, constants, charges, names)

    def __post_init__(self) -> None:
        _require_nonnegative(self.total_concentration, "total_concentration")
        if len(self.species_charges) != len(self.acid_dissociation_constants) + 1:
            raise ValueError("species_charges length must be one more than acid_dissociation_constants.")
        if len(self.species_names) != len(self.species_charges):
            raise ValueError("species_names length must match species_charges.")
        for constant in self.acid_dissociation_constants:
            _require_positive(constant, "acid_dissociation_constant")


@dataclass(frozen=True)
class ActivityAcidBaseResult:
    """Result of an activity-corrected acid-base charge-balance solve."""

    ph: float
    ionic_strength: float
    hydrogen_concentration: float
    hydroxide_concentration: float
    species_concentrations: dict[str, float]
    activity_coefficients: dict[str, float]
    charge_balance_residual: float
    iterations: int


@dataclass(frozen=True)
class PkaFitResult:
    """Least-squares fit of pKa values to measured mean-charge data."""

    pkas: tuple[float, ...]
    residuals: tuple[float, ...]
    sum_squares: float
    iterations: int


def davies_log10_activity_coefficient(
    charge: float,
    ionic_strength: float,
    a_parameter: float = DAVIES_A_25C,
) -> float:
    """Return log10(gamma) from the Davies equation."""
    _require_nonnegative(ionic_strength, "ionic_strength")
    _require_positive(a_parameter, "a_parameter")
    if math.isclose(charge, 0.0) or math.isclose(ionic_strength, 0.0):
        return 0.0
    sqrt_strength = math.sqrt(ionic_strength)
    return -a_parameter * charge**2 * (
        sqrt_strength / (1.0 + sqrt_strength) - 0.3 * ionic_strength
    )


def davies_activity_coefficient(
    charge: float,
    ionic_strength: float,
    a_parameter: float = DAVIES_A_25C,
) -> float:
    """Return gamma from the Davies equation."""
    return 10.0 ** davies_log10_activity_coefficient(charge, ionic_strength, a_parameter)


def davies_activity_coefficients(
    charges: Mapping[str, float],
    ionic_strength: float,
    a_parameter: float = DAVIES_A_25C,
) -> dict[str, float]:
    """Return Davies activity coefficients for named species."""
    return {
        species: davies_activity_coefficient(charge, ionic_strength, a_parameter)
        for species, charge in charges.items()
    }


def concentration_equilibrium_constant_with_davies(
    thermodynamic_equilibrium_constant: float,
    stoichiometry: Mapping[str, float],
    charges: Mapping[str, float],
    ionic_strength: float,
    a_parameter: float = DAVIES_A_25C,
) -> float:
    """Return concentration-based K' from thermodynamic K and Davies gammas."""
    _require_positive(thermodynamic_equilibrium_constant, "thermodynamic_equilibrium_constant")
    gamma_product = 1.0
    for species, coefficient in stoichiometry.items():
        if species not in charges:
            raise ValueError(f"Missing charge for {species}.")
        gamma = davies_activity_coefficient(charges[species], ionic_strength, a_parameter)
        gamma_product *= gamma**coefficient
    return thermodynamic_equilibrium_constant / gamma_product


def davies_concentration_pka(
    thermodynamic_pka: float,
    acid_charge: float,
    conjugate_base_charge: float,
    ionic_strength: float,
    hydrogen_charge: float = 1.0,
    a_parameter: float = DAVIES_A_25C,
) -> float:
    """Return apparent concentration pKa at an ionic strength using Davies gammas."""
    gamma_acid = davies_activity_coefficient(acid_charge, ionic_strength, a_parameter)
    gamma_base = davies_activity_coefficient(conjugate_base_charge, ionic_strength, a_parameter)
    gamma_hydrogen = davies_activity_coefficient(hydrogen_charge, ionic_strength, a_parameter)
    return thermodynamic_pka + math.log10(gamma_hydrogen * gamma_base / gamma_acid)


def polyprotic_activity_distribution_fractions(
    ph: float,
    acid_dissociation_constants: Iterable[float],
    species_activity_coefficients: Iterable[float],
    hydrogen_activity_coefficient: float = 1.0,
) -> tuple[float, ...]:
    """Return HnA, H(n-1)A, ..., A fractions using thermodynamic Ka values."""
    constants = tuple(acid_dissociation_constants)
    gammas = tuple(species_activity_coefficients)
    if len(gammas) != len(constants) + 1:
        raise ValueError("species_activity_coefficients length must be one more than acid_dissociation_constants.")
    _require_positive(hydrogen_activity_coefficient, "hydrogen_activity_coefficient")

    hydrogen_activity = 10.0 ** (-ph)
    terms = [1.0]
    cumulative_constant = 1.0
    for index, constant in enumerate(constants, start=1):
        _require_positive(constant, "acid_dissociation_constant")
        cumulative_constant *= constant
        terms.append(
            cumulative_constant
            * gammas[0]
            / (hydrogen_activity**index * gammas[index])
        )
    denominator = sum(terms)
    return tuple(term / denominator for term in terms)


def polyprotic_activity_species_concentrations(
    total_concentration: float,
    ph: float,
    acid_dissociation_constants: Iterable[float],
    species_activity_coefficients: Iterable[float],
    species_names: Iterable[str],
    hydrogen_activity_coefficient: float = 1.0,
) -> dict[str, float]:
    """Return activity-corrected acid-base species concentrations at a pH."""
    _require_nonnegative(total_concentration, "total_concentration")
    fractions = polyprotic_activity_distribution_fractions(
        ph,
        acid_dissociation_constants,
        species_activity_coefficients,
        hydrogen_activity_coefficient,
    )
    names = tuple(species_names)
    if len(names) != len(fractions):
        raise ValueError("species_names length must match species fractions.")
    return {name: total_concentration * fraction for name, fraction in zip(names, fractions)}


def polyprotic_activity_average_charge(
    ph: float,
    acid_dissociation_constants: Iterable[float],
    species_charges: Iterable[float],
    species_activity_coefficients: Iterable[float],
    hydrogen_activity_coefficient: float = 1.0,
) -> float:
    """Return concentration-weighted average charge with activity corrections."""
    charges = tuple(species_charges)
    fractions = polyprotic_activity_distribution_fractions(
        ph,
        acid_dissociation_constants,
        species_activity_coefficients,
        hydrogen_activity_coefficient,
    )
    if len(charges) != len(fractions):
        raise ValueError("species_charges length must match species fractions.")
    return sum(fraction * charge for fraction, charge in zip(fractions, charges))


def solve_acid_base_mixture_activity(
    components: Iterable[AcidBaseComponent],
    strong_ion_concentrations: Mapping[str, float] | None = None,
    strong_ion_charges: Mapping[str, float] | None = None,
    *,
    ionic_strength: float | None = None,
    initial_ionic_strength: float | None = None,
    kw: float = KW_25C,
    ph_bounds: tuple[float, float] = (-2.0, 16.0),
    tolerance: float = 1e-12,
    max_iterations: int = 100,
    a_parameter: float = DAVIES_A_25C,
) -> ActivityAcidBaseResult:
    """Solve pH and composition for acid-base systems using Davies activities.

    Pass ``ionic_strength`` for a fixed-background-electrolyte calculation.
    Leave it as ``None`` to iterate ionic strength from the calculated species.
    """
    component_list = tuple(components)
    ions = dict(strong_ion_concentrations or {})
    ion_charges = dict(strong_ion_charges or {})
    _require_positive(kw, "kw")
    if not component_list and not ions:
        raise ValueError("At least one component or strong ion is required.")
    for species, concentration in ions.items():
        _require_nonnegative(concentration, f"concentration for {species}")
        if species not in ion_charges:
            raise ValueError(f"Missing charge for {species}.")

    if ionic_strength is not None:
        _require_nonnegative(ionic_strength, "ionic_strength")
        return _solve_fixed_ionic_strength(
            component_list,
            ions,
            ion_charges,
            ionic_strength,
            kw,
            ph_bounds,
            tolerance,
            max_iterations,
            a_parameter,
            outer_iterations=1,
        )

    if initial_ionic_strength is None:
        current_strength = max(_strong_ion_ionic_strength(ions, ion_charges), 1e-8)
    else:
        _require_nonnegative(initial_ionic_strength, "initial_ionic_strength")
        current_strength = max(initial_ionic_strength, 1e-12)

    result = None
    for outer_iteration in range(1, max_iterations + 1):
        result = _solve_fixed_ionic_strength(
            component_list,
            ions,
            ion_charges,
            current_strength,
            kw,
            ph_bounds,
            tolerance,
            max_iterations,
            a_parameter,
            outer_iterations=outer_iteration,
        )
        next_strength = ionic_strength_from_concentrations(
            result.species_concentrations,
            _result_species_charges(component_list, ion_charges),
        )
        if math.isclose(next_strength, current_strength, rel_tol=1e-8, abs_tol=1e-12):
            return result
        current_strength = 0.5 * current_strength + 0.5 * max(next_strength, 1e-12)

    if result is None:
        raise RuntimeError("Activity-corrected acid-base solve failed.")
    return result


def ionic_strength_from_concentrations(
    concentrations: Mapping[str, float],
    charges: Mapping[str, float],
) -> float:
    """Return ionic strength, I = 1/2 sum(c_i z_i^2), for named species."""
    strength = 0.0
    for species, concentration in concentrations.items():
        if species not in charges:
            raise ValueError(f"Missing charge for {species}.")
        _require_nonnegative(concentration, f"concentration for {species}")
        strength += concentration * charges[species] ** 2
    return 0.5 * strength


def fit_pkas_from_mean_charge(
    ph_values: Sequence[float],
    measured_mean_charges: Sequence[float],
    species_charges: Sequence[float],
    initial_pkas: Sequence[float],
    *,
    ionic_strength: float = 0.0,
    initial_step: float = 0.5,
    tolerance: float = 1e-7,
    max_iterations: int = 200,
    a_parameter: float = DAVIES_A_25C,
) -> PkaFitResult:
    """Fit pKa values by minimizing squared mean-charge residuals."""
    if len(ph_values) != len(measured_mean_charges):
        raise ValueError("ph_values and measured_mean_charges must have the same length.")
    if len(species_charges) != len(initial_pkas) + 1:
        raise ValueError("species_charges length must be one more than initial_pkas.")
    _require_nonnegative(ionic_strength, "ionic_strength")
    _require_positive(initial_step, "initial_step")
    _require_positive(tolerance, "tolerance")

    pkas = tuple(float(value) for value in initial_pkas)
    if sorted(pkas) != list(pkas):
        raise ValueError("initial_pkas must be in ascending order.")

    species_gammas = tuple(
        davies_activity_coefficient(charge, ionic_strength, a_parameter)
        for charge in species_charges
    )
    hydrogen_gamma = davies_activity_coefficient(1.0, ionic_strength, a_parameter)

    def objective(candidate_pkas: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        if sorted(candidate_pkas) != list(candidate_pkas):
            return math.inf, tuple(math.inf for _ in ph_values)
        constants = tuple(10.0 ** (-pka) for pka in candidate_pkas)
        residuals = tuple(
            measured
            - polyprotic_activity_average_charge(
                ph,
                constants,
                species_charges,
                species_gammas,
                hydrogen_gamma,
            )
            for ph, measured in zip(ph_values, measured_mean_charges)
        )
        return sum(residual * residual for residual in residuals), residuals

    best = list(pkas)
    best_sum, best_residuals = objective(best)
    step = initial_step
    iterations = 0
    while step > tolerance and iterations < max_iterations:
        iterations += 1
        improved = False
        for index in range(len(best)):
            for direction in (1.0, -1.0):
                candidate = best.copy()
                candidate[index] += direction * step
                candidate_sum, candidate_residuals = objective(candidate)
                if candidate_sum < best_sum:
                    best = candidate
                    best_sum = candidate_sum
                    best_residuals = candidate_residuals
                    improved = True
        if not improved:
            step *= 0.5

    return PkaFitResult(tuple(best), best_residuals, best_sum, iterations)


def _solve_fixed_ionic_strength(
    components: tuple[AcidBaseComponent, ...],
    strong_ion_concentrations: Mapping[str, float],
    strong_ion_charges: Mapping[str, float],
    ionic_strength: float,
    kw: float,
    ph_bounds: tuple[float, float],
    tolerance: float,
    max_iterations: int,
    a_parameter: float,
    outer_iterations: int,
) -> ActivityAcidBaseResult:
    charge_gamma = {
        charge: davies_activity_coefficient(charge, ionic_strength, a_parameter)
        for charge in _all_charges(components, strong_ion_charges)
    }
    charge_gamma[1.0] = davies_activity_coefficient(1.0, ionic_strength, a_parameter)
    charge_gamma[-1.0] = davies_activity_coefficient(-1.0, ionic_strength, a_parameter)

    strong_charge = sum(
        strong_ion_concentrations[species] * strong_ion_charges[species]
        for species in strong_ion_concentrations
    )

    def residual(ph: float) -> float:
        hydrogen_activity = 10.0 ** (-ph)
        hydrogen = hydrogen_activity / charge_gamma[1.0]
        hydroxide = kw / hydrogen_activity / charge_gamma[-1.0]
        balance = hydrogen - hydroxide + strong_charge
        for component in components:
            gammas = tuple(charge_gamma[charge] for charge in component.species_charges)
            balance += component.total_concentration * polyprotic_activity_average_charge(
                ph,
                component.acid_dissociation_constants,
                component.species_charges,
                gammas,
                charge_gamma[1.0],
            )
        return balance

    ph = _bisect_ph(residual, ph_bounds, tolerance, max_iterations)
    species = dict(strong_ion_concentrations)
    activity_coefficients = {
        species_name: davies_activity_coefficient(charge, ionic_strength, a_parameter)
        for species_name, charge in strong_ion_charges.items()
    }
    for component in components:
        gammas = tuple(charge_gamma[charge] for charge in component.species_charges)
        concentrations = polyprotic_activity_species_concentrations(
            component.total_concentration,
            ph,
            component.acid_dissociation_constants,
            gammas,
            component.species_names,
            charge_gamma[1.0],
        )
        for species_name, concentration in concentrations.items():
            species[species_name] = species.get(species_name, 0.0) + concentration
        for species_name, charge in zip(component.species_names, component.species_charges):
            activity_coefficients[species_name] = charge_gamma[charge]

    hydrogen_activity = 10.0 ** (-ph)
    hydrogen = hydrogen_activity / charge_gamma[1.0]
    hydroxide = kw / hydrogen_activity / charge_gamma[-1.0]
    species["H+"] = hydrogen
    species["OH-"] = hydroxide
    activity_coefficients["H+"] = charge_gamma[1.0]
    activity_coefficients["OH-"] = charge_gamma[-1.0]

    return ActivityAcidBaseResult(
        ph=ph,
        ionic_strength=ionic_strength,
        hydrogen_concentration=hydrogen,
        hydroxide_concentration=hydroxide,
        species_concentrations=species,
        activity_coefficients=activity_coefficients,
        charge_balance_residual=residual(ph),
        iterations=outer_iterations,
    )


def _bisect_ph(
    residual,
    ph_bounds: tuple[float, float],
    tolerance: float,
    max_iterations: int,
) -> float:
    low, high = ph_bounds
    if low >= high:
        raise ValueError("ph_bounds must be increasing.")
    low_value = residual(low)
    high_value = residual(high)
    expansion = 0
    while low_value * high_value > 0 and expansion < 6:
        low -= 2.0
        high += 2.0
        low_value = residual(low)
        high_value = residual(high)
        expansion += 1
    if low_value * high_value > 0:
        raise ValueError("Charge balance root is not bracketed by ph_bounds.")

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        mid_value = residual(mid)
        if abs(mid_value) <= tolerance or math.isclose(low, high, rel_tol=tolerance, abs_tol=tolerance):
            return mid
        if low_value * mid_value > 0:
            low = mid
            low_value = mid_value
        else:
            high = mid
    return (low + high) / 2.0


def _strong_ion_ionic_strength(
    strong_ion_concentrations: Mapping[str, float],
    strong_ion_charges: Mapping[str, float],
) -> float:
    return ionic_strength_from_concentrations(strong_ion_concentrations, strong_ion_charges)


def _result_species_charges(
    components: tuple[AcidBaseComponent, ...],
    strong_ion_charges: Mapping[str, float],
) -> dict[str, float]:
    charges = dict(strong_ion_charges)
    for component in components:
        for species, charge in zip(component.species_names, component.species_charges):
            charges[species] = charge
    charges["H+"] = 1.0
    charges["OH-"] = -1.0
    return charges


def _all_charges(
    components: tuple[AcidBaseComponent, ...],
    strong_ion_charges: Mapping[str, float],
) -> set[float]:
    charges = set(strong_ion_charges.values())
    for component in components:
        charges.update(component.species_charges)
    charges.update((1.0, -1.0))
    return charges


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
