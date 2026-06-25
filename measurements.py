"""Analytical measurement corrections and calibration helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from constants import (
    DEFAULT_AIR_DENSITY_G_PER_ML,
    DEFAULT_BALANCE_WEIGHT_DENSITY_G_PER_ML,
    GAS_CONSTANT_L_BAR,
)


def celsius_to_kelvin(temp_c: float) -> float:
    return temp_c + 273.15


def water_density_g_per_ml(temp_c: float) -> float:
    """Return pure water density in g/mL near atmospheric pressure.

    The polynomial is commonly used for 0-40 C calibration work and returns
    density in kg/m^3 before converting to g/mL.
    """
    density_kg_per_m3 = (
        999.842594
        + 6.793952e-2 * temp_c
        - 9.095290e-3 * temp_c**2
        + 1.001685e-4 * temp_c**3
        - 1.120083e-6 * temp_c**4
        + 6.536332e-9 * temp_c**5
    )
    return density_kg_per_m3 / 1000.0


def ideal_gas_density_g_per_ml(
    molar_mass_g_per_mol: float,
    temp_c: float,
    pressure_bar: float = 1.0,
) -> float:
    """Return ideal-gas density in g/mL."""
    density_g_per_l = pressure_bar * molar_mass_g_per_mol / (
        GAS_CONSTANT_L_BAR * celsius_to_kelvin(temp_c)
    )
    return density_g_per_l / 1000.0


def buoyancy_correction_factor(
    object_density_g_per_ml: float,
    air_density_g_per_ml: float = DEFAULT_AIR_DENSITY_G_PER_ML,
    weight_density_g_per_ml: float = DEFAULT_BALANCE_WEIGHT_DENSITY_G_PER_ML,
) -> float:
    """Return factor that converts apparent mass in air to true mass in vacuum."""
    if object_density_g_per_ml <= 0:
        raise ValueError("object_density_g_per_ml must be positive.")
    return (1.0 - air_density_g_per_ml / weight_density_g_per_ml) / (
        1.0 - air_density_g_per_ml / object_density_g_per_ml
    )


def true_mass_from_apparent(
    apparent_mass_g: float,
    object_density_g_per_ml: float,
    air_density_g_per_ml: float = DEFAULT_AIR_DENSITY_G_PER_ML,
    weight_density_g_per_ml: float = DEFAULT_BALANCE_WEIGHT_DENSITY_G_PER_ML,
) -> float:
    """Return true mass in vacuum from a balance reading in air."""
    return apparent_mass_g * buoyancy_correction_factor(
        object_density_g_per_ml,
        air_density_g_per_ml,
        weight_density_g_per_ml,
    )


def apparent_mass_from_true(
    true_mass_g: float,
    object_density_g_per_ml: float,
    air_density_g_per_ml: float = DEFAULT_AIR_DENSITY_G_PER_ML,
    weight_density_g_per_ml: float = DEFAULT_BALANCE_WEIGHT_DENSITY_G_PER_ML,
) -> float:
    """Return the balance reading needed for a desired true mass."""
    return true_mass_g / buoyancy_correction_factor(
        object_density_g_per_ml,
        air_density_g_per_ml,
        weight_density_g_per_ml,
    )


def apparent_mass_bias_percent(
    object_density_g_per_ml: float,
    air_density_g_per_ml: float = DEFAULT_AIR_DENSITY_G_PER_ML,
    weight_density_g_per_ml: float = DEFAULT_BALANCE_WEIGHT_DENSITY_G_PER_ML,
) -> float:
    """Return percent bias if apparent mass is used as though it were true mass."""
    factor = buoyancy_correction_factor(
        object_density_g_per_ml,
        air_density_g_per_ml,
        weight_density_g_per_ml,
    )
    return (1.0 / factor - 1.0) * 100.0


def water_mass_to_volume_ml(
    apparent_water_mass_g: float,
    temp_c: float,
    air_density_g_per_ml: float = DEFAULT_AIR_DENSITY_G_PER_ML,
    weight_density_g_per_ml: float = DEFAULT_BALANCE_WEIGHT_DENSITY_G_PER_ML,
) -> float:
    """Convert apparent mass of water weighed in air to true water volume."""
    water_density = water_density_g_per_ml(temp_c)
    true_water_mass = true_mass_from_apparent(
        apparent_water_mass_g,
        water_density,
        air_density_g_per_ml,
        weight_density_g_per_ml,
    )
    return true_water_mass / water_density


def volume_correction_from_water_mass(
    nominal_volume_ml: float,
    apparent_water_mass_g: float,
    temp_c: float,
) -> float:
    """Return actual minus nominal volume for a water-calibrated vessel."""
    return water_mass_to_volume_ml(apparent_water_mass_g, temp_c) - nominal_volume_ml


@dataclass(frozen=True)
class BuretCalibrationInterval:
    """Calibration result for one buret delivery interval."""

    initial_reading_ml: float
    final_reading_ml: float
    apparent_water_mass_g: float
    temp_c: float

    @property
    def nominal_volume_ml(self) -> float:
        return self.final_reading_ml - self.initial_reading_ml

    @property
    def actual_volume_ml(self) -> float:
        return water_mass_to_volume_ml(self.apparent_water_mass_g, self.temp_c)

    @property
    def correction_ml(self) -> float:
        return self.actual_volume_ml - self.nominal_volume_ml


@dataclass(frozen=True)
class BuretCalibration:
    """Collection of buret delivery corrections."""

    intervals: tuple[BuretCalibrationInterval, ...]

    @classmethod
    def from_water_masses(
        cls,
        readings_ml: Iterable[tuple[float, float]],
        apparent_water_masses_g: Iterable[float],
        temp_c: float,
    ) -> "BuretCalibration":
        intervals = tuple(
            BuretCalibrationInterval(start, end, mass, temp_c)
            for (start, end), mass in zip(readings_ml, apparent_water_masses_g)
        )
        return cls(intervals)

    def correction_points(self) -> tuple[tuple[float, float], ...]:
        return tuple((interval.final_reading_ml, interval.correction_ml) for interval in self.intervals)

    def correction_at(self, reading_ml: float) -> float:
        return linear_interpolate(reading_ml, self.correction_points())


def linear_interpolate(x_value: float, points: Iterable[tuple[float, float]]) -> float:
    """Linearly interpolate y for x from sorted or unsorted ``(x, y)`` points."""
    ordered = sorted(points)
    if len(ordered) < 2:
        raise ValueError("At least two points are required.")
    if x_value < ordered[0][0] or x_value > ordered[-1][0]:
        raise ValueError("x_value is outside the calibration range.")

    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= x_value <= x1:
            if x1 == x0:
                raise ValueError("Calibration x-values must be distinct.")
            fraction = (x_value - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    raise ValueError("x_value is outside the calibration range.")


def volume_at_temperature(
    volume_ml: float,
    from_temp_c: float,
    to_temp_c: float,
    expansion_coefficient_per_c: float,
) -> float:
    """Return expanded or contracted volume using a volumetric coefficient."""
    return volume_ml * (1.0 + expansion_coefficient_per_c * (to_temp_c - from_temp_c))


def aqueous_volume_at_temperature(
    volume_ml: float,
    from_temp_c: float,
    to_temp_c: float,
) -> float:
    """Approximate dilute aqueous solution volume change using water density."""
    return volume_ml * water_density_g_per_ml(from_temp_c) / water_density_g_per_ml(to_temp_c)


def aqueous_molarity_at_temperature(
    molarity: float,
    from_temp_c: float,
    to_temp_c: float,
) -> float:
    """Approximate dilute aqueous solution molarity after a temperature change."""
    return molarity * water_density_g_per_ml(to_temp_c) / water_density_g_per_ml(from_temp_c)


def concentration_after_evaporation(
    initial_concentration: float,
    initial_volume_ml: float,
    evaporated_volume_ml: float,
) -> float:
    """Return concentration after solvent evaporation with solute amount fixed."""
    final_volume_ml = initial_volume_ml - evaporated_volume_ml
    if final_volume_ml <= 0:
        raise ValueError("evaporated_volume_ml must be less than initial_volume_ml.")
    return initial_concentration * initial_volume_ml / final_volume_ml


def relative_humidity_percent(
    water_vapor_pressure_pa: float,
    equilibrium_vapor_pressure_pa: float,
) -> float:
    return water_vapor_pressure_pa / equilibrium_vapor_pressure_pa * 100.0


def water_vapor_pressure_from_humidity(
    relative_humidity: float,
    equilibrium_vapor_pressure_pa: float,
) -> float:
    return relative_humidity / 100.0 * equilibrium_vapor_pressure_pa


def gravitational_reading_at_height(
    mass_reading_g: float,
    height_change_m: float,
    earth_radius_km: float = 6370.0,
) -> float:
    """Return reading change from gravity alone without recalibration."""
    radius_m = earth_radius_km * 1000.0
    return mass_reading_g * (radius_m / (radius_m + height_change_m)) ** 2


def surface_mass_from_loading(
    area_mm2: float,
    loading_ug_per_cm2: float,
) -> float:
    """Return surface mass in micrograms from area in mm^2 and loading in ug/cm^2."""
    area_cm2 = area_mm2 / 100.0
    return area_cm2 * loading_ug_per_cm2


def qcm_frequency_shift_hz(
    surface_loading_ng_per_cm2: float,
    sensitivity_hz_per_ng_cm2: float,
) -> float:
    """Return quartz crystal microbalance frequency shift magnitude."""
    return surface_loading_ng_per_cm2 * sensitivity_hz_per_ng_cm2


def buoyant_capacity_kg(volume_l: float, fluid_density_kg_per_l: float = 1.0) -> float:
    """Return maximum supported mass from displaced fluid volume."""
    return volume_l * fluid_density_kg_per_l


def relative_uncertainty(values: Iterable[float]) -> float:
    """Combine independent relative uncertainties in quadrature."""
    return math.sqrt(sum(value**2 for value in values))
