"""Electrochemistry helpers for cells, Nernst calculations, and electrolysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from constants import FARADAY_CONSTANT_C_PER_MOL


R_J_PER_MOL_K = 8.314_462_618_153_24
STANDARD_TEMPERATURE_K = 298.15
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class GalvanicCell:
    """A two-electrode cell described by tabulated reduction potentials."""

    cathode: str
    anode: str
    cathode_reduction_potential_v: float
    anode_reduction_potential_v: float
    electrons_transferred: float | None = None

    @property
    def standard_cell_potential_v(self) -> float:
        return standard_cell_potential(
            self.cathode_reduction_potential_v,
            self.anode_reduction_potential_v,
        )

    @property
    def spontaneous(self) -> bool:
        return self.standard_cell_potential_v > 0.0

    @property
    def delta_g_standard_j_per_mol(self) -> float | None:
        if self.electrons_transferred is None:
            return None
        return delta_g_from_cell_potential(
            self.electrons_transferred,
            self.standard_cell_potential_v,
        )


def standard_cell_potential(
    cathode_reduction_potential_v: float,
    anode_reduction_potential_v: float,
) -> float:
    """Return E_cell from standard reduction potentials."""
    return cathode_reduction_potential_v - anode_reduction_potential_v


def spontaneous_galvanic_cell(
    reduction_potentials_v: Mapping[str, float],
    electrons_transferred: float | None = None,
) -> GalvanicCell:
    """Choose cathode and anode from reduction potentials for a galvanic cell."""
    if len(reduction_potentials_v) < 2:
        raise ValueError("At least two reduction potentials are required.")
    if electrons_transferred is not None:
        _require_positive(electrons_transferred, "electrons_transferred")

    cathode = max(reduction_potentials_v, key=reduction_potentials_v.get)
    anode = min(reduction_potentials_v, key=reduction_potentials_v.get)
    return GalvanicCell(
        cathode=cathode,
        anode=anode,
        cathode_reduction_potential_v=reduction_potentials_v[cathode],
        anode_reduction_potential_v=reduction_potentials_v[anode],
        electrons_transferred=electrons_transferred,
    )


def nernst_log10_slope_v(
    electrons_transferred: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return the base-10 Nernst slope, 2.303 RT / nF, in volts."""
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(temperature_k, "temperature_k")
    return math.log(10.0) * R_J_PER_MOL_K * temperature_k / (
        electrons_transferred * FARADAY_CONSTANT_C_PER_MOL
    )


def nernst_potential(
    standard_potential_v: float,
    electrons_transferred: float,
    reaction_quotient: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return E = E_standard - RT/(nF) ln(Q)."""
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(reaction_quotient, "reaction_quotient")
    _require_positive(temperature_k, "temperature_k")
    return standard_potential_v - (
        R_J_PER_MOL_K
        * temperature_k
        / (electrons_transferred * FARADAY_CONSTANT_C_PER_MOL)
        * math.log(reaction_quotient)
    )


def cell_potential(
    cathode_reduction_potential_v: float,
    anode_reduction_potential_v: float,
    electrons_transferred: float,
    reaction_quotient: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return nonstandard cell potential from reduction potentials and Q."""
    return nernst_potential(
        standard_cell_potential(cathode_reduction_potential_v, anode_reduction_potential_v),
        electrons_transferred,
        reaction_quotient,
        temperature_k,
    )


def concentration_cell_potential(
    higher_activity: float,
    lower_activity: float,
    electrons_transferred: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return the potential for a concentration cell with identical electrodes."""
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(higher_activity, "higher_activity")
    _require_positive(lower_activity, "lower_activity")
    if higher_activity < lower_activity:
        raise ValueError("higher_activity must be greater than or equal to lower_activity.")
    return (
        R_J_PER_MOL_K
        * temperature_k
        / (electrons_transferred * FARADAY_CONSTANT_C_PER_MOL)
        * math.log(higher_activity / lower_activity)
    )


def delta_g_from_cell_potential(
    electrons_transferred: float,
    cell_potential_v: float,
) -> float:
    """Return Delta G = -nFE in J/mol of reaction."""
    _require_positive(electrons_transferred, "electrons_transferred")
    return -electrons_transferred * FARADAY_CONSTANT_C_PER_MOL * cell_potential_v


def cell_potential_from_delta_g(
    electrons_transferred: float,
    delta_g_j_per_mol: float,
) -> float:
    """Return cell potential from Delta G in J/mol of reaction."""
    _require_positive(electrons_transferred, "electrons_transferred")
    return -delta_g_j_per_mol / (electrons_transferred * FARADAY_CONSTANT_C_PER_MOL)


def equilibrium_constant_from_cell_potential(
    electrons_transferred: float,
    standard_cell_potential_v: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return K from E_standard for a redox reaction."""
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(temperature_k, "temperature_k")
    exponent = (
        electrons_transferred
        * FARADAY_CONSTANT_C_PER_MOL
        * standard_cell_potential_v
        / (R_J_PER_MOL_K * temperature_k)
    )
    return math.exp(exponent)


def cell_potential_from_equilibrium_constant(
    electrons_transferred: float,
    equilibrium_constant: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return E_standard from K for a redox reaction."""
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(equilibrium_constant, "equilibrium_constant")
    _require_positive(temperature_k, "temperature_k")
    return (
        R_J_PER_MOL_K
        * temperature_k
        / (electrons_transferred * FARADAY_CONSTANT_C_PER_MOL)
        * math.log(equilibrium_constant)
    )


def charge_from_current_time(current_a: float, time_s: float) -> float:
    """Return charge in coulombs from current and time."""
    _require_nonnegative(current_a, "current_a")
    _require_nonnegative(time_s, "time_s")
    return current_a * time_s


def current_from_charge_time(charge_c: float, time_s: float) -> float:
    """Return current in amperes from charge and time."""
    _require_nonnegative(charge_c, "charge_c")
    _require_positive(time_s, "time_s")
    return charge_c / time_s


def time_from_charge_current(charge_c: float, current_a: float) -> float:
    """Return time in seconds from charge and current."""
    _require_nonnegative(charge_c, "charge_c")
    _require_positive(current_a, "current_a")
    return charge_c / current_a


def moles_electrons_from_charge(charge_c: float) -> float:
    """Return moles of electrons from charge in coulombs."""
    _require_nonnegative(charge_c, "charge_c")
    return charge_c / FARADAY_CONSTANT_C_PER_MOL


def charge_from_moles_electrons(moles_electrons: float) -> float:
    """Return charge in coulombs from moles of electrons."""
    _require_nonnegative(moles_electrons, "moles_electrons")
    return moles_electrons * FARADAY_CONSTANT_C_PER_MOL


def moles_product_from_charge(
    charge_c: float,
    electrons_per_mole_product: float,
) -> float:
    """Return moles of product or reactant made/consumed by electrolysis."""
    _require_nonnegative(charge_c, "charge_c")
    _require_positive(electrons_per_mole_product, "electrons_per_mole_product")
    return moles_electrons_from_charge(charge_c) / electrons_per_mole_product


def charge_for_moles_product(
    moles_product: float,
    electrons_per_mole_product: float,
) -> float:
    """Return charge needed for a target amount of electrolysis product."""
    _require_nonnegative(moles_product, "moles_product")
    _require_positive(electrons_per_mole_product, "electrons_per_mole_product")
    return charge_from_moles_electrons(moles_product * electrons_per_mole_product)


def mass_from_charge(
    charge_c: float,
    molar_mass_g_per_mol: float,
    electrons_per_mole_product: float,
) -> float:
    """Return product mass from charge, molar mass, and electron stoichiometry."""
    _require_positive(molar_mass_g_per_mol, "molar_mass_g_per_mol")
    return moles_product_from_charge(charge_c, electrons_per_mole_product) * molar_mass_g_per_mol


def mass_from_current_time(
    current_a: float,
    time_s: float,
    molar_mass_g_per_mol: float,
    electrons_per_mole_product: float,
) -> float:
    """Return product mass from electrolysis current and time."""
    return mass_from_charge(
        charge_from_current_time(current_a, time_s),
        molar_mass_g_per_mol,
        electrons_per_mole_product,
    )


def charge_for_mass(
    mass_g: float,
    molar_mass_g_per_mol: float,
    electrons_per_mole_product: float,
) -> float:
    """Return charge needed to produce or consume a target mass."""
    _require_nonnegative(mass_g, "mass_g")
    _require_positive(molar_mass_g_per_mol, "molar_mass_g_per_mol")
    return charge_for_moles_product(mass_g / molar_mass_g_per_mol, electrons_per_mole_product)


def time_for_mass(
    mass_g: float,
    current_a: float,
    molar_mass_g_per_mol: float,
    electrons_per_mole_product: float,
) -> float:
    """Return electrolysis time in seconds for a target mass at fixed current."""
    return time_from_charge_current(
        charge_for_mass(mass_g, molar_mass_g_per_mol, electrons_per_mole_product),
        current_a,
    )


def charge_from_amp_hours(amp_hours: float) -> float:
    """Return charge in coulombs from amp-hours."""
    _require_nonnegative(amp_hours, "amp_hours")
    return amp_hours * SECONDS_PER_HOUR


def amp_hours_from_charge(charge_c: float) -> float:
    """Return amp-hours from charge in coulombs."""
    _require_nonnegative(charge_c, "charge_c")
    return charge_c / SECONDS_PER_HOUR


def amp_hours_from_moles_electrons(moles_electrons: float) -> float:
    """Return battery capacity in amp-hours from moles of electrons."""
    return amp_hours_from_charge(charge_from_moles_electrons(moles_electrons))


def moles_electrons_from_amp_hours(amp_hours: float) -> float:
    """Return moles of electrons from battery capacity in amp-hours."""
    return moles_electrons_from_charge(charge_from_amp_hours(amp_hours))


def electrical_energy_j(charge_c: float, voltage_v: float) -> float:
    """Return electrical energy in joules from charge and voltage."""
    _require_nonnegative(charge_c, "charge_c")
    _require_nonnegative(voltage_v, "voltage_v")
    return charge_c * voltage_v


def power_from_current_voltage(current_a: float, voltage_v: float) -> float:
    """Return electrical power in watts."""
    _require_nonnegative(current_a, "current_a")
    _require_nonnegative(voltage_v, "voltage_v")
    return current_a * voltage_v


def energy_from_current_voltage_time(
    current_a: float,
    voltage_v: float,
    time_s: float,
) -> float:
    """Return electrical energy in joules from current, voltage, and time."""
    return electrical_energy_j(charge_from_current_time(current_a, time_s), voltage_v)


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
