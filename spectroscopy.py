"""Spectroscopy and photometric-analysis helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from constants import AVOGADRO, PLANCK_CONSTANT_J_S, SPEED_OF_LIGHT_M_PER_S
from measurements import LinearFitResult, linear_least_squares


_MASS_CONCENTRATION_FACTORS_G_PER_L = {
    "g/L": 1.0,
    "mg/mL": 1.0,
    "mg/L": 1.0e-3,
    "ug/mL": 1.0e-3,
    "microgram/mL": 1.0e-3,
    "ug/L": 1.0e-6,
    "ng/mL": 1.0e-6,
}


@dataclass(frozen=True)
class StandardAdditionAmountResult:
    """Standard-addition result where standards are added as amounts."""

    fit: LinearFitResult
    x_intercept: float
    unknown_amount: float
    original_sample_amount: float | None = None
    amount_per_sample_mass: float | None = None


@dataclass(frozen=True)
class TwoLineEndpoint:
    """Endpoint found from the intersection of two fitted signal branches."""

    x_value: float
    y_value: float
    first_slope: float
    first_intercept: float
    second_slope: float
    second_intercept: float


def frequency_from_wavelength_nm(wavelength_nm: float) -> float:
    """Return radiation frequency in Hz from wavelength in nm."""
    _require_positive(wavelength_nm, "wavelength_nm")
    return SPEED_OF_LIGHT_M_PER_S / (wavelength_nm * 1.0e-9)


def wavelength_nm_from_frequency_hz(frequency_hz: float) -> float:
    """Return wavelength in nm from frequency in Hz."""
    _require_positive(frequency_hz, "frequency_hz")
    return SPEED_OF_LIGHT_M_PER_S / frequency_hz * 1.0e9


def wavenumber_from_wavelength_nm(wavelength_nm: float) -> float:
    """Return wavenumber in cm^-1 from wavelength in nm."""
    _require_positive(wavelength_nm, "wavelength_nm")
    return 1.0 / (wavelength_nm * 1.0e-7)


def wavelength_nm_from_wavenumber_cm(wavenumber_cm: float) -> float:
    """Return wavelength in nm from wavenumber in cm^-1."""
    _require_positive(wavenumber_cm, "wavenumber_cm")
    return 1.0e7 / wavenumber_cm


def photon_energy_j_from_wavelength_nm(wavelength_nm: float) -> float:
    """Return photon energy in joules from wavelength in nm."""
    _require_positive(wavelength_nm, "wavelength_nm")
    return PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S / (wavelength_nm * 1.0e-9)


def photon_energy_j_from_frequency_hz(frequency_hz: float) -> float:
    """Return photon energy in joules from frequency in Hz."""
    _require_positive(frequency_hz, "frequency_hz")
    return PLANCK_CONSTANT_J_S * frequency_hz


def photon_energy_kj_per_mol_from_wavelength_nm(wavelength_nm: float) -> float:
    """Return photon energy in kJ/mol from wavelength in nm."""
    return photon_energy_j_from_wavelength_nm(wavelength_nm) * AVOGADRO / 1000.0


def absorbance_from_transmittance(transmittance: float) -> float:
    """Return absorbance from fractional transmittance."""
    _require_fraction(transmittance, "transmittance")
    if transmittance == 0:
        raise ValueError("transmittance must be greater than zero.")
    return -math.log10(transmittance)


def absorbance_from_percent_transmittance(percent_transmittance: float) -> float:
    """Return absorbance from percent transmittance."""
    _require_range(percent_transmittance, "percent_transmittance", 0.0, 100.0)
    if percent_transmittance == 0:
        raise ValueError("percent_transmittance must be greater than zero.")
    return absorbance_from_transmittance(percent_transmittance / 100.0)


def transmittance_from_absorbance(absorbance: float) -> float:
    """Return fractional transmittance from absorbance."""
    return 10.0 ** (-absorbance)


def percent_transmittance_from_absorbance(absorbance: float) -> float:
    """Return percent transmittance from absorbance."""
    return transmittance_from_absorbance(absorbance) * 100.0


def absorbance_from_intensities(transmitted_intensity: float, incident_intensity: float) -> float:
    """Return absorbance from transmitted and incident radiant intensities."""
    _require_positive(transmitted_intensity, "transmitted_intensity")
    _require_positive(incident_intensity, "incident_intensity")
    return absorbance_from_transmittance(transmitted_intensity / incident_intensity)


def corrected_absorbance(sample_absorbance: float, blank_absorbance: float = 0.0) -> float:
    """Return blank-corrected absorbance."""
    return sample_absorbance - blank_absorbance


def beer_lambert_absorbance(
    molar_absorptivity: float,
    path_length_cm: float,
    concentration_mol_l: float,
) -> float:
    """Return absorbance from Beer's law, ``A = epsilon b c``."""
    _require_positive(molar_absorptivity, "molar_absorptivity")
    _require_positive(path_length_cm, "path_length_cm")
    if concentration_mol_l < 0:
        raise ValueError("concentration_mol_l cannot be negative.")
    return molar_absorptivity * path_length_cm * concentration_mol_l


def beer_lambert_concentration(
    absorbance: float,
    molar_absorptivity: float,
    path_length_cm: float,
) -> float:
    """Return concentration in mol/L from absorbance and Beer's law."""
    _require_positive(molar_absorptivity, "molar_absorptivity")
    _require_positive(path_length_cm, "path_length_cm")
    return absorbance / (molar_absorptivity * path_length_cm)


def beer_lambert_molar_absorptivity(
    absorbance: float,
    concentration_mol_l: float,
    path_length_cm: float,
) -> float:
    """Return molar absorptivity from absorbance, concentration, and path length."""
    _require_positive(concentration_mol_l, "concentration_mol_l")
    _require_positive(path_length_cm, "path_length_cm")
    return absorbance / (path_length_cm * concentration_mol_l)


def beer_lambert_path_length(
    absorbance: float,
    molar_absorptivity: float,
    concentration_mol_l: float,
) -> float:
    """Return path length in cm from absorbance and Beer's law."""
    _require_positive(molar_absorptivity, "molar_absorptivity")
    _require_positive(concentration_mol_l, "concentration_mol_l")
    return absorbance / (molar_absorptivity * concentration_mol_l)


def molar_concentration_from_mass_concentration(
    mass_concentration: float,
    molar_mass_g_per_mol: float,
    unit: str = "ug/mL",
) -> float:
    """Return molar concentration from a mass concentration."""
    _require_positive(molar_mass_g_per_mol, "molar_mass_g_per_mol")
    return mass_concentration * _mass_concentration_factor(unit) / molar_mass_g_per_mol


def mass_concentration_from_molar_concentration(
    concentration_mol_l: float,
    molar_mass_g_per_mol: float,
    unit: str = "ug/mL",
) -> float:
    """Return mass concentration from molar concentration."""
    _require_positive(molar_mass_g_per_mol, "molar_mass_g_per_mol")
    factor = _mass_concentration_factor(unit)
    if factor == 0:
        raise ValueError("unit conversion factor cannot be zero.")
    return concentration_mol_l * molar_mass_g_per_mol / factor


def molar_absorptivity_from_mass_calibration_slope(
    slope_absorbance_per_mass_concentration: float,
    molar_mass_g_per_mol: float,
    path_length_cm: float = 1.0,
    mass_concentration_unit: str = "ug/mL",
) -> float:
    """Convert a mass-concentration calibration slope to molar absorptivity."""
    _require_positive(molar_mass_g_per_mol, "molar_mass_g_per_mol")
    _require_positive(path_length_cm, "path_length_cm")
    factor = _mass_concentration_factor(mass_concentration_unit)
    return slope_absorbance_per_mass_concentration * molar_mass_g_per_mol / (path_length_cm * factor)


def protein_molar_absorptivity_a280(
    tryptophan_count: float,
    tyrosine_count: float,
    disulfide_count: float = 0.0,
) -> float:
    """Estimate protein molar absorptivity at 280 nm."""
    _require_nonnegative(tryptophan_count, "tryptophan_count")
    _require_nonnegative(tyrosine_count, "tyrosine_count")
    _require_nonnegative(disulfide_count, "disulfide_count")
    return 5500.0 * tryptophan_count + 1490.0 * tyrosine_count + 125.0 * disulfide_count


def calibration_concentrations_from_stock(
    stock_concentration: float,
    aliquot_volumes: Iterable[float],
    final_volume: float,
) -> tuple[float, ...]:
    """Return standard concentrations prepared by diluting stock aliquots."""
    _require_positive(final_volume, "final_volume")
    aliquots = _as_tuple(aliquot_volumes, "aliquot_volumes")
    concentrations = []
    for aliquot in aliquots:
        _require_nonnegative(aliquot, "aliquot volume")
        if aliquot > final_volume:
            raise ValueError("aliquot volumes cannot exceed final_volume.")
        concentrations.append(stock_concentration * aliquot / final_volume)
    return tuple(concentrations)


def concentration_from_calibration_signal(
    fit: LinearFitResult,
    signal: float,
    sample_aliquot_volume: float | None = None,
    final_volume: float | None = None,
) -> float:
    """Return concentration from a linear calibration, optionally undoing dilution."""
    measured_concentration = fit.x_at(signal)
    if sample_aliquot_volume is None and final_volume is None:
        return measured_concentration
    if sample_aliquot_volume is None or final_volume is None:
        raise ValueError("sample_aliquot_volume and final_volume must be provided together.")
    _require_positive(sample_aliquot_volume, "sample_aliquot_volume")
    _require_positive(final_volume, "final_volume")
    return measured_concentration * final_volume / sample_aliquot_volume


def dilution_corrected_signals(
    signals: Iterable[float],
    initial_volume: float,
    *,
    added_volumes: Iterable[float] | None = None,
    final_volumes: Iterable[float] | None = None,
) -> tuple[float, ...]:
    """Correct signals to the concentration scale of the initial volume."""
    _require_positive(initial_volume, "initial_volume")
    signal_data = _as_tuple(signals, "signals")
    if (added_volumes is None) == (final_volumes is None):
        raise ValueError("Provide exactly one of added_volumes or final_volumes.")

    if added_volumes is not None:
        added_data = _as_tuple(added_volumes, "added_volumes")
        _require_same_length(signal_data, added_data, "signals", "added_volumes")
        volumes = tuple(initial_volume + added_volume for added_volume in added_data)
    else:
        volumes = _as_tuple(final_volumes or (), "final_volumes")
        _require_same_length(signal_data, volumes, "signals", "final_volumes")

    corrected = []
    for signal, final_volume in zip(signal_data, volumes):
        _require_positive(final_volume, "final volume")
        corrected.append(signal * final_volume / initial_volume)
    return tuple(corrected)


def standard_addition_from_added_amounts(
    added_amounts: Iterable[float],
    signals: Iterable[float],
    *,
    blank_signal: float = 0.0,
    sample_aliquot_volume: float | None = None,
    total_sample_volume: float | None = None,
    sample_mass: float | None = None,
) -> StandardAdditionAmountResult:
    """Return unknown amount from a standard-addition line."""
    added_data = _as_tuple(added_amounts, "added_amounts")
    signal_data = _as_tuple(signals, "signals")
    _require_same_length(added_data, signal_data, "added_amounts", "signals")
    corrected_signals = tuple(signal - blank_signal for signal in signal_data)

    fit = linear_least_squares(added_data, corrected_signals)
    if fit.slope == 0:
        raise ValueError("standard-addition slope cannot be zero.")
    x_intercept = -fit.intercept / fit.slope
    unknown_amount = -x_intercept

    if (sample_aliquot_volume is None) != (total_sample_volume is None):
        raise ValueError("sample_aliquot_volume and total_sample_volume must be provided together.")

    original_sample_amount = None
    if sample_aliquot_volume is not None and total_sample_volume is not None:
        _require_positive(sample_aliquot_volume, "sample_aliquot_volume")
        _require_positive(total_sample_volume, "total_sample_volume")
        original_sample_amount = unknown_amount * total_sample_volume / sample_aliquot_volume

    amount_per_sample_mass = None
    if sample_mass is not None:
        _require_positive(sample_mass, "sample_mass")
        amount_for_mass = original_sample_amount if original_sample_amount is not None else unknown_amount
        amount_per_sample_mass = amount_for_mass / sample_mass

    return StandardAdditionAmountResult(
        fit=fit,
        x_intercept=x_intercept,
        unknown_amount=unknown_amount,
        original_sample_amount=original_sample_amount,
        amount_per_sample_mass=amount_per_sample_mass,
    )


def two_line_endpoint(
    first_branch_points: Iterable[tuple[float, float]],
    second_branch_points: Iterable[tuple[float, float]],
) -> TwoLineEndpoint:
    """Return the intersection of two linear branches."""
    first_x, first_y = _split_points(first_branch_points, "first_branch_points")
    second_x, second_y = _split_points(second_branch_points, "second_branch_points")
    first_fit = linear_least_squares(first_x, first_y)
    second_fit = linear_least_squares(second_x, second_y)
    if first_fit.slope == second_fit.slope:
        raise ValueError("Endpoint is undefined for parallel fitted lines.")
    x_value = (second_fit.intercept - first_fit.intercept) / (first_fit.slope - second_fit.slope)
    return TwoLineEndpoint(
        x_value=x_value,
        y_value=first_fit.y_at(x_value),
        first_slope=first_fit.slope,
        first_intercept=first_fit.intercept,
        second_slope=second_fit.slope,
        second_intercept=second_fit.intercept,
    )


def _as_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    data = tuple(values)
    if not data:
        raise ValueError(f"{name} must contain at least one value.")
    return data


def _split_points(
    points: Iterable[tuple[float, float]],
    name: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    point_data = tuple(points)
    if len(point_data) < 3:
        raise ValueError(f"{name} must contain at least three points.")
    return tuple(point[0] for point in point_data), tuple(point[1] for point in point_data)


def _require_same_length(
    first: tuple[float, ...],
    second: tuple[float, ...],
    first_name: str,
    second_name: str,
) -> None:
    if len(first) != len(second):
        raise ValueError(f"{first_name} and {second_name} must have the same length.")


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


def _require_fraction(value: float, name: str) -> None:
    _require_range(value, name, 0.0, 1.0)


def _require_range(value: float, name: str, lower: float, upper: float) -> None:
    if value < lower or value > upper:
        raise ValueError(f"{name} must be between {lower} and {upper}.")


def _mass_concentration_factor(unit: str) -> float:
    try:
        return _MASS_CONCENTRATION_FACTORS_G_PER_L[unit]
    except KeyError as error:
        units = ", ".join(sorted(_MASS_CONCENTRATION_FACTORS_G_PER_L))
        raise ValueError(f"Unsupported mass concentration unit {unit!r}. Use one of: {units}.") from error
