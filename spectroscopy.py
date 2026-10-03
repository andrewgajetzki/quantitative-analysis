"""Spectroscopy and photometric-analysis helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from constants import (
    AVOGADRO,
    BOLTZMANN_CONSTANT_J_PER_K,
    PLANCK_CONSTANT_J_S,
    SPEED_OF_LIGHT_M_PER_S,
)
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


@dataclass(frozen=True)
class StrayLightResult:
    """Apparent absorbance result when stray light reaches the detector."""

    true_transmittance: float
    apparent_transmittance: float
    apparent_absorbance: float
    concentration_relative_error_percent: float


@dataclass(frozen=True)
class FTIRSampling:
    """Sampling limits for an evenly sampled interferogram."""

    max_wavenumber_cm_inverse: float
    max_retardation_cm: float
    resolution_cm_inverse: float
    sample_interval_s: float | None = None
    acquisition_time_s: float | None = None


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


def blackbody_spectral_exitance(
    wavelength_um: float,
    temperature_k: float,
    *,
    per_um: bool = True,
) -> float:
    """Return blackbody spectral exitance at one wavelength.

    The result is in W m^-2 um^-1 by default. Set ``per_um=False`` to obtain
    W m^-3, the SI form per meter of wavelength.
    """
    _require_positive(wavelength_um, "wavelength_um")
    _require_positive(temperature_k, "temperature_k")
    wavelength_m = wavelength_um * 1.0e-6
    exponent = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S / (
        wavelength_m * BOLTZMANN_CONSTANT_J_PER_K * temperature_k
    )
    spectral_exitance_per_m = (
        2.0
        * math.pi
        * PLANCK_CONSTANT_J_S
        * SPEED_OF_LIGHT_M_PER_S**2
        / wavelength_m**5
        / math.expm1(exponent)
    )
    return spectral_exitance_per_m * 1.0e-6 if per_um else spectral_exitance_per_m


def integrated_blackbody_exitance(
    start_wavelength_um: float,
    end_wavelength_um: float,
    temperature_k: float,
    *,
    intervals: int = 1000,
) -> float:
    """Integrate blackbody spectral exitance over a wavelength interval."""
    _require_positive(start_wavelength_um, "start_wavelength_um")
    _require_positive(end_wavelength_um, "end_wavelength_um")
    _require_positive(temperature_k, "temperature_k")
    if end_wavelength_um <= start_wavelength_um:
        raise ValueError("end_wavelength_um must exceed start_wavelength_um.")
    if intervals < 1:
        raise ValueError("intervals must be at least 1.")

    step = (end_wavelength_um - start_wavelength_um) / intervals
    total = 0.0
    for index in range(intervals + 1):
        wavelength = start_wavelength_um + index * step
        weight = 0.5 if index in (0, intervals) else 1.0
        total += weight * blackbody_spectral_exitance(wavelength, temperature_k)
    return total * step


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


def apparent_transmittance_with_stray_light(
    true_absorbance: float,
    stray_fraction: float,
    *,
    stray_in_blank: bool = True,
) -> float:
    """Return apparent transmittance when a fraction of stray light is detected."""
    _require_nonnegative(true_absorbance, "true_absorbance")
    _require_nonnegative(stray_fraction, "stray_fraction")
    true_transmittance = transmittance_from_absorbance(true_absorbance)
    denominator = 1.0 + stray_fraction if stray_in_blank else 1.0
    return (true_transmittance + stray_fraction) / denominator


def apparent_absorbance_with_stray_light(
    true_absorbance: float,
    stray_fraction: float,
    *,
    stray_in_blank: bool = True,
) -> float:
    """Return apparent absorbance when a fraction of stray light is detected."""
    return absorbance_from_transmittance(
        apparent_transmittance_with_stray_light(true_absorbance, stray_fraction, stray_in_blank=stray_in_blank)
    )


def stray_light_error(
    true_absorbance: float,
    stray_fraction: float,
    *,
    stray_in_blank: bool = True,
) -> StrayLightResult:
    """Return apparent transmittance, absorbance, and Beer-law concentration error."""
    true_transmittance = transmittance_from_absorbance(true_absorbance)
    apparent_transmittance = apparent_transmittance_with_stray_light(
        true_absorbance,
        stray_fraction,
        stray_in_blank=stray_in_blank,
    )
    apparent_absorbance = absorbance_from_transmittance(apparent_transmittance)
    if true_absorbance == 0:
        concentration_error = 0.0
    else:
        concentration_error = (apparent_absorbance / true_absorbance - 1.0) * 100.0
    return StrayLightResult(
        true_transmittance=true_transmittance,
        apparent_transmittance=apparent_transmittance,
        apparent_absorbance=apparent_absorbance,
        concentration_relative_error_percent=concentration_error,
    )


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


def grating_line_density_from_angles(
    wavelength_nm: float,
    order: int,
    incident_angle_deg: float,
    diffraction_angle_deg: float,
    *,
    output: str = "lines/cm",
) -> float:
    """Return grating line density from the signed grating equation."""
    _require_positive(wavelength_nm, "wavelength_nm")
    _require_positive(order, "order")
    spacing_m = order * wavelength_nm * 1.0e-9 / (
        math.sin(math.radians(incident_angle_deg)) + math.sin(math.radians(diffraction_angle_deg))
    )
    if spacing_m <= 0:
        raise ValueError("Angles and order produce a nonpositive groove spacing.")
    if output == "lines/cm":
        return 1.0 / (spacing_m * 100.0)
    if output == "lines/mm":
        return 1.0 / (spacing_m * 1000.0)
    if output == "spacing_m":
        return spacing_m
    raise ValueError("output must be 'lines/cm', 'lines/mm', or 'spacing_m'.")


def diffraction_angle_deg(
    wavelength_nm: float,
    order: int,
    line_density: float,
    *,
    line_density_unit: str = "lines/mm",
    incident_angle_deg: float = 0.0,
) -> float:
    """Return the signed diffraction angle from the grating equation."""
    _require_positive(wavelength_nm, "wavelength_nm")
    _require_positive(order, "order")
    spacing_m = grating_spacing_m(line_density, line_density_unit)
    sine_value = order * wavelength_nm * 1.0e-9 / spacing_m - math.sin(math.radians(incident_angle_deg))
    if sine_value < -1.0 or sine_value > 1.0:
        raise ValueError("No real diffraction angle exists for these values.")
    return math.degrees(math.asin(sine_value))


def grating_spacing_m(line_density: float, unit: str = "lines/mm") -> float:
    """Return grating spacing in meters from a line density."""
    _require_positive(line_density, "line_density")
    if unit == "lines/mm":
        return 1.0 / (line_density * 1000.0)
    if unit == "lines/cm":
        return 1.0 / (line_density * 100.0)
    if unit == "lines/m":
        return 1.0 / line_density
    raise ValueError("unit must be 'lines/mm', 'lines/cm', or 'lines/m'.")


def grating_angular_dispersion_deg_per_um(
    order: int,
    line_density: float,
    diffraction_angle_deg_value: float,
    *,
    line_density_unit: str = "lines/mm",
) -> float:
    """Return grating angular dispersion in degrees per micrometer."""
    _require_positive(order, "order")
    spacing_um = grating_spacing_m(line_density, line_density_unit) * 1.0e6
    cosine = math.cos(math.radians(diffraction_angle_deg_value))
    if cosine == 0:
        raise ValueError("Angular dispersion is undefined at 90 degrees.")
    return order / (spacing_um * cosine) * 180.0 / math.pi


def grating_angular_separation_deg(
    wavelength_1_nm: float,
    wavelength_2_nm: float,
    order: int,
    line_density: float,
    diffraction_angle_deg_value: float,
    *,
    line_density_unit: str = "lines/mm",
) -> float:
    """Approximate angular separation between nearby wavelengths."""
    dispersion = grating_angular_dispersion_deg_per_um(
        order,
        line_density,
        diffraction_angle_deg_value,
        line_density_unit=line_density_unit,
    )
    return dispersion * abs(wavelength_2_nm - wavelength_1_nm) / 1000.0


def required_resolving_power(value_1: float, value_2: float) -> float:
    """Return resolving power needed to separate two nearby spectral values."""
    delta = abs(value_2 - value_1)
    if delta == 0:
        raise ValueError("Spectral values must be distinct.")
    return ((value_1 + value_2) / 2.0) / delta


def resolvable_delta(spectral_value: float, resolving_power: float) -> float:
    """Return smallest resolvable spacing at a given resolving power."""
    _require_positive(spectral_value, "spectral_value")
    _require_positive(resolving_power, "resolving_power")
    return spectral_value / resolving_power


def grating_resolving_power(order: int, illuminated_grooves: float) -> float:
    """Return grating resolving power, ``R = m N``."""
    _require_positive(order, "order")
    _require_positive(illuminated_grooves, "illuminated_grooves")
    return order * illuminated_grooves


def illuminated_grooves(grating_width: float, line_density: float, *, width_unit: str = "cm") -> float:
    """Return number of grooves illuminated over a grating width."""
    _require_positive(grating_width, "grating_width")
    _require_positive(line_density, "line_density")
    if width_unit == "cm":
        return grating_width * line_density
    if width_unit == "mm":
        return grating_width * line_density
    raise ValueError("width_unit must be 'cm' or 'mm', matching line_density units.")


def fringe_pathlength_cm_from_wavenumbers(
    fringe_count: float,
    wavenumber_1_cm: float,
    wavenumber_2_cm: float,
    *,
    refractive_index: float = 1.0,
) -> float:
    """Return cell pathlength from interference fringe count and wavenumbers."""
    _require_positive(fringe_count, "fringe_count")
    _require_positive(wavenumber_1_cm, "wavenumber_1_cm")
    _require_positive(wavenumber_2_cm, "wavenumber_2_cm")
    _require_positive(refractive_index, "refractive_index")
    delta_wavenumber = abs(wavenumber_2_cm - wavenumber_1_cm)
    if delta_wavenumber == 0:
        raise ValueError("wavenumbers must be distinct.")
    return fringe_count / (2.0 * refractive_index * delta_wavenumber)


def fringe_count_between_wavenumbers(
    pathlength_cm: float,
    wavenumber_1_cm: float,
    wavenumber_2_cm: float,
    *,
    refractive_index: float = 1.0,
) -> float:
    """Return number of interference fringes across a wavenumber interval."""
    _require_positive(pathlength_cm, "pathlength_cm")
    _require_positive(refractive_index, "refractive_index")
    return 2.0 * refractive_index * pathlength_cm * abs(wavenumber_2_cm - wavenumber_1_cm)


def ftir_sampling_limits(
    sampling_interval_cm: float,
    sample_count: int,
    *,
    symmetric: bool = True,
    mirror_velocity_cm_s: float | None = None,
) -> FTIRSampling:
    """Return FTIR wavenumber range, retardation, and resolution estimates."""
    _require_positive(sampling_interval_cm, "sampling_interval_cm")
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    max_wavenumber = 1.0 / (2.0 * sampling_interval_cm)
    if symmetric:
        max_retardation = (sample_count - 1) * sampling_interval_cm / 2.0
        interval_count = sample_count - 1
    else:
        max_retardation = (sample_count - 1) * sampling_interval_cm
        interval_count = sample_count - 1
    resolution = 1.0 / max_retardation

    sample_interval = None
    acquisition_time = None
    if mirror_velocity_cm_s is not None:
        _require_positive(mirror_velocity_cm_s, "mirror_velocity_cm_s")
        sample_interval = sampling_interval_cm / (2.0 * mirror_velocity_cm_s)
        acquisition_time = interval_count * sample_interval

    return FTIRSampling(
        max_wavenumber_cm_inverse=max_wavenumber,
        max_retardation_cm=max_retardation,
        resolution_cm_inverse=resolution,
        sample_interval_s=sample_interval,
        acquisition_time_s=acquisition_time,
    )


def signal_to_noise_after_averaging(single_scan_signal_to_noise: float, scan_count: float) -> float:
    """Return signal-to-noise ratio after averaging independent scans."""
    _require_positive(single_scan_signal_to_noise, "single_scan_signal_to_noise")
    _require_positive(scan_count, "scan_count")
    return single_scan_signal_to_noise * math.sqrt(scan_count)


def scans_required_for_signal_to_noise(
    current_signal_to_noise: float,
    target_signal_to_noise: float,
    *,
    current_scan_count: float = 1.0,
) -> float:
    """Return total scans needed to reach a target signal-to-noise ratio."""
    _require_positive(current_signal_to_noise, "current_signal_to_noise")
    _require_positive(target_signal_to_noise, "target_signal_to_noise")
    _require_positive(current_scan_count, "current_scan_count")
    return current_scan_count * (target_signal_to_noise / current_signal_to_noise) ** 2


def moving_average(values: Iterable[float], window_size: int) -> tuple[float, ...]:
    """Return a trailing moving average for smoothing noisy instrumental data."""
    data = _as_tuple(values, "values")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if window_size > len(data):
        raise ValueError("window_size cannot exceed number of values.")
    return tuple(
        sum(data[index : index + window_size]) / window_size
        for index in range(len(data) - window_size + 1)
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
