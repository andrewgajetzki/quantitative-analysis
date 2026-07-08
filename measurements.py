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


@dataclass(frozen=True)
class ConfidenceInterval:
    """Two-sided confidence interval around a center value."""

    center: float
    half_width: float
    confidence: float
    degrees_of_freedom: float
    critical_value: float

    @property
    def lower(self) -> float:
        return self.center - self.half_width

    @property
    def upper(self) -> float:
        return self.center + self.half_width


@dataclass(frozen=True)
class FTestResult:
    """Result of comparing two variances with an F test."""

    statistic: float
    critical_value: float
    degrees_of_freedom_numerator: int
    degrees_of_freedom_denominator: int
    p_value: float
    significant: bool
    confidence: float


@dataclass(frozen=True)
class TTestResult:
    """Result of a two-sided Student's t test."""

    statistic: float
    critical_value: float
    degrees_of_freedom: float
    p_value: float
    significant: bool
    confidence: float
    difference: float
    standard_error: float


@dataclass(frozen=True)
class GrubbsResult:
    """Result of a two-sided Grubbs outlier test."""

    statistic: float
    critical_value: float
    outlier_index: int
    outlier_value: float
    significant: bool
    confidence: float


@dataclass(frozen=True)
class CalibrationPrediction:
    """Inverse prediction from a linear calibration curve."""

    x_value: float
    standard_uncertainty: float
    confidence_interval: ConfidenceInterval | None = None


@dataclass(frozen=True)
class LinearFitResult:
    """Least-squares fit for ``y = slope * x + intercept``."""

    slope: float
    intercept: float
    slope_uncertainty: float
    intercept_uncertainty: float
    standard_error_y: float
    r_squared: float
    n: int
    x_mean: float
    y_mean: float
    sum_xx: float

    def y_at(self, x_value: float) -> float:
        return self.slope * x_value + self.intercept

    def x_at(self, y_value: float) -> float:
        if self.slope == 0:
            raise ValueError("Cannot solve for x when slope is zero.")
        return (y_value - self.intercept) / self.slope

    def inverse_prediction(
        self,
        mean_signal: float,
        replicate_count: int = 1,
        confidence: float | None = None,
    ) -> CalibrationPrediction:
        """Return x and uncertainty from a measured mean response."""
        if replicate_count < 1:
            raise ValueError("replicate_count must be at least 1.")
        if self.slope == 0:
            raise ValueError("Cannot make inverse prediction when slope is zero.")

        x_value = self.x_at(mean_signal)
        standard_uncertainty = abs(self.standard_error_y / self.slope) * math.sqrt(
            1.0 / replicate_count
            + 1.0 / self.n
            + (mean_signal - self.y_mean) ** 2 / (self.slope**2 * self.sum_xx)
        )
        interval = None
        if confidence is not None:
            critical = t_critical_two_tailed(confidence, self.n - 2)
            interval = ConfidenceInterval(
                center=x_value,
                half_width=critical * standard_uncertainty,
                confidence=confidence,
                degrees_of_freedom=self.n - 2,
                critical_value=critical,
            )
        return CalibrationPrediction(x_value, standard_uncertainty, interval)


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean."""
    data = _as_tuple(values, "values")
    return sum(data) / len(data)


def sample_variance(values: Iterable[float]) -> float:
    """Return sample variance with ``n - 1`` degrees of freedom."""
    data = _as_tuple(values, "values")
    _require_length(data, 2, "sample variance")
    center = mean(data)
    return sum((value - center) ** 2 for value in data) / (len(data) - 1)


def sample_standard_deviation(values: Iterable[float]) -> float:
    """Return sample standard deviation with ``n - 1`` degrees of freedom."""
    return math.sqrt(sample_variance(values))


def standard_deviation_of_mean(values: Iterable[float]) -> float:
    """Return standard deviation of the mean, ``s / sqrt(n)``."""
    data = _as_tuple(values, "values")
    return standard_error_from_standard_deviation(sample_standard_deviation(data), len(data))


def standard_error_from_standard_deviation(standard_deviation: float, n: int) -> float:
    """Return standard error of the mean from a sample standard deviation."""
    if n < 1:
        raise ValueError("n must be at least 1.")
    return standard_deviation / math.sqrt(n)


def relative_standard_deviation_percent(values: Iterable[float]) -> float:
    """Return relative standard deviation as a percent."""
    data = _as_tuple(values, "values")
    center = mean(data)
    if center == 0:
        raise ValueError("relative standard deviation is undefined for a zero mean.")
    return abs(sample_standard_deviation(data) / center) * 100.0


def confidence_interval_mean(
    values: Iterable[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Return Student's t confidence interval for a sample mean."""
    data = _as_tuple(values, "values")
    return confidence_interval_from_stats(
        mean(data),
        sample_standard_deviation(data),
        len(data),
        confidence,
    )


def confidence_interval_from_stats(
    mean_value: float,
    standard_deviation: float,
    n: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Return Student's t confidence interval from mean, standard deviation, and n."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    _validate_probability(confidence, "confidence")
    critical = t_critical_two_tailed(confidence, n - 1)
    half_width = critical * standard_error_from_standard_deviation(standard_deviation, n)
    return ConfidenceInterval(mean_value, half_width, confidence, n - 1, critical)


def normal_pdf(x_value: float, mean_value: float = 0.0, standard_deviation: float = 1.0) -> float:
    """Return the Gaussian probability density."""
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive.")
    z_value = (x_value - mean_value) / standard_deviation
    return math.exp(-0.5 * z_value**2) / (standard_deviation * math.sqrt(2.0 * math.pi))


def normal_cdf(x_value: float, mean_value: float = 0.0, standard_deviation: float = 1.0) -> float:
    """Return the Gaussian cumulative probability up to x."""
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive.")
    z_value = (x_value - mean_value) / (standard_deviation * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z_value))


def normal_probability_between(
    lower: float,
    upper: float,
    mean_value: float = 0.0,
    standard_deviation: float = 1.0,
) -> float:
    """Return Gaussian probability between lower and upper bounds."""
    if lower > upper:
        raise ValueError("lower cannot exceed upper.")
    return normal_cdf(upper, mean_value, standard_deviation) - normal_cdf(
        lower,
        mean_value,
        standard_deviation,
    )


def normal_expected_count_between(
    total_count: float,
    lower: float,
    upper: float,
    mean_value: float,
    standard_deviation: float,
) -> float:
    """Return expected count in a Gaussian interval."""
    return total_count * normal_probability_between(lower, upper, mean_value, standard_deviation)


def pooled_standard_deviation(
    standard_deviation_1: float,
    n_1: int,
    standard_deviation_2: float,
    n_2: int,
) -> float:
    """Return pooled standard deviation for two samples."""
    if n_1 < 2 or n_2 < 2:
        raise ValueError("Both sample sizes must be at least 2.")
    numerator = (n_1 - 1) * standard_deviation_1**2 + (n_2 - 1) * standard_deviation_2**2
    return math.sqrt(numerator / (n_1 + n_2 - 2))


def f_test_variances(
    sample_1: Iterable[float],
    sample_2: Iterable[float],
    confidence: float = 0.95,
    two_sided: bool = False,
) -> FTestResult:
    """Compare sample variances with an F test.

    The default follows common analytical-chemistry tables: place the larger
    variance in the numerator and compare with the upper-tail F critical value
    for the requested confidence.
    """
    data_1 = _as_tuple(sample_1, "sample_1")
    data_2 = _as_tuple(sample_2, "sample_2")
    return f_test_variances_from_stats(
        sample_standard_deviation(data_1),
        len(data_1),
        sample_standard_deviation(data_2),
        len(data_2),
        confidence,
        two_sided,
    )


def f_test_variances_from_stats(
    standard_deviation_1: float,
    n_1: int,
    standard_deviation_2: float,
    n_2: int,
    confidence: float = 0.95,
    two_sided: bool = False,
) -> FTestResult:
    """Compare variances from standard deviations and sample sizes."""
    if n_1 < 2 or n_2 < 2:
        raise ValueError("Both sample sizes must be at least 2.")
    _validate_probability(confidence, "confidence")

    variance_1 = standard_deviation_1**2
    variance_2 = standard_deviation_2**2
    if variance_1 >= variance_2:
        statistic = variance_1 / variance_2
        df_numerator = n_1 - 1
        df_denominator = n_2 - 1
    else:
        statistic = variance_2 / variance_1
        df_numerator = n_2 - 1
        df_denominator = n_1 - 1

    alpha = 1.0 - confidence
    critical_probability = 1.0 - alpha / 2.0 if two_sided else confidence
    critical_value = f_ppf(critical_probability, df_numerator, df_denominator)
    upper_tail = 1.0 - f_cdf(statistic, df_numerator, df_denominator)
    p_value = min(1.0, 2.0 * upper_tail) if two_sided else upper_tail

    return FTestResult(
        statistic=statistic,
        critical_value=critical_value,
        degrees_of_freedom_numerator=df_numerator,
        degrees_of_freedom_denominator=df_denominator,
        p_value=p_value,
        significant=statistic > critical_value,
        confidence=confidence,
    )


def one_sample_t_test(
    values: Iterable[float],
    expected_mean: float,
    confidence: float = 0.95,
) -> TTestResult:
    """Run a two-sided one-sample t test against an expected mean."""
    data = _as_tuple(values, "values")
    return one_sample_t_test_from_stats(
        mean(data),
        sample_standard_deviation(data),
        len(data),
        expected_mean,
        confidence,
    )


def one_sample_t_test_from_stats(
    mean_value: float,
    standard_deviation: float,
    n: int,
    expected_mean: float,
    confidence: float = 0.95,
) -> TTestResult:
    """Run a two-sided one-sample t test from summary statistics."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    standard_error = standard_error_from_standard_deviation(standard_deviation, n)
    if standard_error == 0:
        raise ValueError("standard error cannot be zero.")
    statistic = (mean_value - expected_mean) / standard_error
    return _two_sided_t_result(
        statistic,
        n - 1,
        confidence,
        mean_value - expected_mean,
        standard_error,
    )


def compare_means_from_stats(
    mean_1: float,
    standard_deviation_1: float,
    n_1: int,
    mean_2: float,
    standard_deviation_2: float,
    n_2: int,
    confidence: float = 0.95,
    equal_variances: bool = True,
) -> TTestResult:
    """Run a two-sided t test comparing two means from summary statistics."""
    if n_1 < 2 or n_2 < 2:
        raise ValueError("Both sample sizes must be at least 2.")

    difference = mean_1 - mean_2
    if equal_variances:
        pooled = pooled_standard_deviation(standard_deviation_1, n_1, standard_deviation_2, n_2)
        standard_error = pooled * math.sqrt(1.0 / n_1 + 1.0 / n_2)
        degrees_of_freedom = n_1 + n_2 - 2
    else:
        variance_term_1 = standard_deviation_1**2 / n_1
        variance_term_2 = standard_deviation_2**2 / n_2
        standard_error = math.sqrt(variance_term_1 + variance_term_2)
        degrees_of_freedom = (variance_term_1 + variance_term_2) ** 2 / (
            variance_term_1**2 / (n_1 - 1) + variance_term_2**2 / (n_2 - 1)
        )

    if standard_error == 0:
        raise ValueError("standard error cannot be zero.")
    return _two_sided_t_result(
        difference / standard_error,
        degrees_of_freedom,
        confidence,
        difference,
        standard_error,
    )


def two_sample_t_test(
    sample_1: Iterable[float],
    sample_2: Iterable[float],
    confidence: float = 0.95,
    equal_variances: bool = True,
) -> TTestResult:
    """Run a two-sided t test comparing two samples."""
    data_1 = _as_tuple(sample_1, "sample_1")
    data_2 = _as_tuple(sample_2, "sample_2")
    return compare_means_from_stats(
        mean(data_1),
        sample_standard_deviation(data_1),
        len(data_1),
        mean(data_2),
        sample_standard_deviation(data_2),
        len(data_2),
        confidence,
        equal_variances,
    )


def paired_t_test(
    sample_1: Iterable[float],
    sample_2: Iterable[float],
    confidence: float = 0.95,
) -> TTestResult:
    """Run a two-sided paired t test using pairwise differences."""
    data_1 = _as_tuple(sample_1, "sample_1")
    data_2 = _as_tuple(sample_2, "sample_2")
    if len(data_1) != len(data_2):
        raise ValueError("Paired samples must have the same length.")
    differences = tuple(value_1 - value_2 for value_1, value_2 in zip(data_1, data_2))
    return one_sample_t_test(differences, 0.0, confidence)


def grubbs_test(values: Iterable[float], confidence: float = 0.95) -> GrubbsResult:
    """Run a two-sided Grubbs test for one outlier."""
    data = _as_tuple(values, "values")
    _require_length(data, 3, "Grubbs test")
    _validate_probability(confidence, "confidence")

    center = mean(data)
    standard_deviation = sample_standard_deviation(data)
    if standard_deviation == 0:
        raise ValueError("standard deviation cannot be zero.")
    deviations = tuple(abs(value - center) for value in data)
    outlier_index = max(range(len(data)), key=deviations.__getitem__)
    statistic = deviations[outlier_index] / standard_deviation
    critical_value = grubbs_critical_value(len(data), confidence)
    return GrubbsResult(
        statistic=statistic,
        critical_value=critical_value,
        outlier_index=outlier_index,
        outlier_value=data[outlier_index],
        significant=statistic > critical_value,
        confidence=confidence,
    )


def grubbs_critical_value(n: int, confidence: float = 0.95) -> float:
    """Return two-sided Grubbs critical value for one possible outlier."""
    if n < 3:
        raise ValueError("n must be at least 3.")
    _validate_probability(confidence, "confidence")
    alpha = 1.0 - confidence
    t_value = t_ppf(1.0 - alpha / (2.0 * n), n - 2)
    return ((n - 1) / math.sqrt(n)) * math.sqrt(t_value**2 / (n - 2 + t_value**2))


def linear_least_squares(
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> LinearFitResult:
    """Fit ``y = slope * x + intercept`` by ordinary least squares."""
    x_data = _as_tuple(x_values, "x_values")
    y_data = _as_tuple(y_values, "y_values")
    if len(x_data) != len(y_data):
        raise ValueError("x_values and y_values must have the same length.")
    _require_length(x_data, 3, "linear least squares")

    x_bar = mean(x_data)
    y_bar = mean(y_data)
    sum_xx = sum((x_value - x_bar) ** 2 for x_value in x_data)
    if sum_xx == 0:
        raise ValueError("x_values must not all be identical.")
    sum_xy = sum((x_value - x_bar) * (y_value - y_bar) for x_value, y_value in zip(x_data, y_data))
    sum_yy = sum((y_value - y_bar) ** 2 for y_value in y_data)

    slope = sum_xy / sum_xx
    intercept = y_bar - slope * x_bar
    residuals = tuple(y_value - (slope * x_value + intercept) for x_value, y_value in zip(x_data, y_data))
    residual_sum_squares = sum(residual**2 for residual in residuals)
    standard_error_y = math.sqrt(residual_sum_squares / (len(x_data) - 2))
    slope_uncertainty = standard_error_y / math.sqrt(sum_xx)
    intercept_uncertainty = standard_error_y * math.sqrt(1.0 / len(x_data) + x_bar**2 / sum_xx)
    r_squared = 1.0 if sum_yy == 0 else 1.0 - residual_sum_squares / sum_yy

    return LinearFitResult(
        slope=slope,
        intercept=intercept,
        slope_uncertainty=slope_uncertainty,
        intercept_uncertainty=intercept_uncertainty,
        standard_error_y=standard_error_y,
        r_squared=r_squared,
        n=len(x_data),
        x_mean=x_bar,
        y_mean=y_bar,
        sum_xx=sum_xx,
    )


def log10_least_squares(
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> LinearFitResult:
    """Fit a log10-log10 calibration curve."""
    x_data = _as_tuple(x_values, "x_values")
    y_data = _as_tuple(y_values, "y_values")
    if any(value <= 0 for value in x_data + y_data):
        raise ValueError("log10 calibration values must be positive.")
    return linear_least_squares(
        (math.log10(value) for value in x_data),
        (math.log10(value) for value in y_data),
    )


def x_from_log10_calibration_y(fit: LinearFitResult, y_value: float) -> float:
    """Return x from a fitted log10-log10 calibration and observed y."""
    if y_value <= 0:
        raise ValueError("y_value must be positive.")
    return 10 ** fit.x_at(math.log10(y_value))


def inverse_linear_interpolate(
    y_value: float,
    points: Iterable[tuple[float, float]],
) -> float:
    """Linearly interpolate x for y from monotonic ``(x, y)`` points."""
    ordered = sorted(points)
    if len(ordered) < 2:
        raise ValueError("At least two points are required.")

    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        lower = min(y0, y1)
        upper = max(y0, y1)
        if lower <= y_value <= upper:
            if y1 == y0:
                raise ValueError("Calibration y-values must be distinct.")
            fraction = (y_value - y0) / (y1 - y0)
            return x0 + fraction * (x1 - x0)
    raise ValueError("y_value is outside the calibration range.")


def t_critical_two_tailed(confidence: float, degrees_of_freedom: float) -> float:
    """Return positive critical t value for a two-sided confidence level."""
    _validate_probability(confidence, "confidence")
    return t_ppf(0.5 + confidence / 2.0, degrees_of_freedom)


def t_ppf(probability: float, degrees_of_freedom: float) -> float:
    """Return inverse Student's t CDF."""
    _validate_probability(probability, "probability", inclusive=True)
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive.")
    if probability == 0:
        return -math.inf
    if probability == 1:
        return math.inf
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -t_ppf(1.0 - probability, degrees_of_freedom)

    lower = 0.0
    upper = 1.0
    while t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def t_cdf(t_value: float, degrees_of_freedom: float) -> float:
    """Return Student's t cumulative probability."""
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive.")
    if t_value == 0:
        return 0.5
    x_value = degrees_of_freedom / (degrees_of_freedom + t_value**2)
    beta_value = _regularized_beta(x_value, degrees_of_freedom / 2.0, 0.5)
    if t_value > 0:
        return 1.0 - 0.5 * beta_value
    return 0.5 * beta_value


def f_ppf(probability: float, df_numerator: float, df_denominator: float) -> float:
    """Return inverse F CDF."""
    _validate_probability(probability, "probability", inclusive=True)
    if df_numerator <= 0 or df_denominator <= 0:
        raise ValueError("degrees of freedom must be positive.")
    if probability == 0:
        return 0.0
    if probability == 1:
        return math.inf

    lower = 0.0
    upper = 1.0
    while f_cdf(upper, df_numerator, df_denominator) < probability:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if f_cdf(midpoint, df_numerator, df_denominator) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def f_cdf(f_value: float, df_numerator: float, df_denominator: float) -> float:
    """Return F distribution cumulative probability."""
    if f_value < 0:
        raise ValueError("f_value cannot be negative.")
    if df_numerator <= 0 or df_denominator <= 0:
        raise ValueError("degrees of freedom must be positive.")
    if f_value == 0:
        return 0.0
    x_value = df_numerator * f_value / (df_numerator * f_value + df_denominator)
    return _regularized_beta(x_value, df_numerator / 2.0, df_denominator / 2.0)


def _two_sided_t_result(
    statistic: float,
    degrees_of_freedom: float,
    confidence: float,
    difference: float,
    standard_error: float,
) -> TTestResult:
    _validate_probability(confidence, "confidence")
    critical_value = t_critical_two_tailed(confidence, degrees_of_freedom)
    p_value = 2.0 * (1.0 - t_cdf(abs(statistic), degrees_of_freedom))
    return TTestResult(
        statistic=statistic,
        critical_value=critical_value,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
        significant=abs(statistic) > critical_value,
        confidence=confidence,
        difference=difference,
        standard_error=standard_error,
    )


def _regularized_beta(x_value: float, a_value: float, b_value: float) -> float:
    if not 0.0 <= x_value <= 1.0:
        raise ValueError("x_value must be between 0 and 1.")
    if a_value <= 0 or b_value <= 0:
        raise ValueError("a_value and b_value must be positive.")
    if x_value == 0.0:
        return 0.0
    if x_value == 1.0:
        return 1.0

    log_beta = math.lgamma(a_value) + math.lgamma(b_value) - math.lgamma(a_value + b_value)
    front = math.exp(
        a_value * math.log(x_value)
        + b_value * math.log1p(-x_value)
        - log_beta
    )
    if x_value < (a_value + 1.0) / (a_value + b_value + 2.0):
        return front * _beta_continued_fraction(a_value, b_value, x_value) / a_value
    return 1.0 - front * _beta_continued_fraction(b_value, a_value, 1.0 - x_value) / b_value


def _beta_continued_fraction(a_value: float, b_value: float, x_value: float) -> float:
    max_iterations = 200
    epsilon = 3.0e-14
    tiny = 1.0e-300

    qab = a_value + b_value
    qap = a_value + 1.0
    qam = a_value - 1.0
    c_value = 1.0
    d_value = 1.0 - qab * x_value / qap
    if abs(d_value) < tiny:
        d_value = tiny
    d_value = 1.0 / d_value
    h_value = d_value

    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b_value - iteration) * x_value / ((qam + m2) * (a_value + m2))
        d_value = 1.0 + aa * d_value
        if abs(d_value) < tiny:
            d_value = tiny
        c_value = 1.0 + aa / c_value
        if abs(c_value) < tiny:
            c_value = tiny
        d_value = 1.0 / d_value
        h_value *= d_value * c_value

        aa = -(a_value + iteration) * (qab + iteration) * x_value / (
            (a_value + m2) * (qap + m2)
        )
        d_value = 1.0 + aa * d_value
        if abs(d_value) < tiny:
            d_value = tiny
        c_value = 1.0 + aa / c_value
        if abs(c_value) < tiny:
            c_value = tiny
        d_value = 1.0 / d_value
        delta = d_value * c_value
        h_value *= delta
        if abs(delta - 1.0) < epsilon:
            return h_value

    raise RuntimeError("regularized beta continued fraction did not converge.")


def _as_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    data = tuple(float(value) for value in values)
    if not data:
        raise ValueError(f"{name} cannot be empty.")
    return data


def _require_length(values: tuple[float, ...], minimum_length: int, label: str) -> None:
    if len(values) < minimum_length:
        raise ValueError(f"{label} requires at least {minimum_length} values.")


def _validate_probability(probability: float, name: str, inclusive: bool = False) -> None:
    if inclusive:
        valid = 0.0 <= probability <= 1.0
    else:
        valid = 0.0 < probability < 1.0
    if not valid:
        raise ValueError(f"{name} must be between 0 and 1.")
