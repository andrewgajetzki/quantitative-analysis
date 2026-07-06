"""Significant-figure and experimental-error helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from enum import Enum
import math
import re

NumberLike = int | float | str | Decimal

_SCIENTIFIC_NOTATION_RE = re.compile(
    r"^(?P<coefficient>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:(?:[eE](?P<e_exponent>[+-]?\d+))|"
    r"(?:(?:[xX*]|\u00d7)10(?:\^?(?P<power_exponent>[+-]?\d+))?))$"
)


class AccuracyPrecision(Enum):
    """Qualitative accuracy and precision categories."""

    ACCURATE_AND_PRECISE = "accurate and precise"
    ACCURATE_NOT_PRECISE = "accurate but not precise"
    PRECISE_NOT_ACCURATE = "precise but not accurate"
    NEITHER = "neither accurate nor precise"


class ErrorKind(Enum):
    """Common experimental-error categories."""

    RANDOM = "random"
    SYSTEMATIC = "systematic"
    BLUNDER = "blunder"


@dataclass(frozen=True)
class Measurement:
    """A measured value with an absolute uncertainty."""

    value: float
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if self.uncertainty < 0:
            raise ValueError("uncertainty cannot be negative.")

    @property
    def relative_uncertainty(self) -> float:
        return relative_uncertainty_from_absolute(self.value, self.uncertainty)

    @property
    def percent_relative_uncertainty(self) -> float:
        return percent_relative_uncertainty(self.value, self.uncertainty)

    def __add__(self, other: object) -> "Measurement":
        other_measurement = _coerce_measurement(other)
        return Measurement(
            self.value + other_measurement.value,
            combine_absolute_uncertainties(self.uncertainty, other_measurement.uncertainty),
        )

    def __radd__(self, other: object) -> "Measurement":
        return self + other

    def __sub__(self, other: object) -> "Measurement":
        other_measurement = _coerce_measurement(other)
        return Measurement(
            self.value - other_measurement.value,
            combine_absolute_uncertainties(self.uncertainty, other_measurement.uncertainty),
        )

    def __rsub__(self, other: object) -> "Measurement":
        other_measurement = _coerce_measurement(other)
        return other_measurement - self

    def __mul__(self, other: object) -> "Measurement":
        other_measurement = _coerce_measurement(other)
        value = self.value * other_measurement.value
        relative = combine_relative_uncertainties(
            self.relative_uncertainty,
            other_measurement.relative_uncertainty,
        )
        return Measurement(value, abs(value) * relative)

    def __rmul__(self, other: object) -> "Measurement":
        return self * other

    def __truediv__(self, other: object) -> "Measurement":
        other_measurement = _coerce_measurement(other)
        if other_measurement.value == 0:
            raise ZeroDivisionError("measurement division by zero.")
        value = self.value / other_measurement.value
        relative = combine_relative_uncertainties(
            self.relative_uncertainty,
            other_measurement.relative_uncertainty,
        )
        return Measurement(value, abs(value) * relative)

    def __rtruediv__(self, other: object) -> "Measurement":
        other_measurement = _coerce_measurement(other)
        return other_measurement / self

    def __pow__(self, exponent: float) -> "Measurement":
        value = self.value**exponent
        if self.value == 0:
            if self.uncertainty == 0:
                return Measurement(value, 0.0)
            raise ValueError("relative uncertainty is undefined for zero values.")
        uncertainty = abs(value) * abs(exponent) * self.relative_uncertainty
        return Measurement(value, uncertainty)

    def __neg__(self) -> "Measurement":
        return Measurement(-self.value, self.uncertainty)


def significant_figures(value: NumberLike) -> int:
    """Return the number of significant figures implied by a numeric string."""
    coefficient = _coefficient_text(value).lstrip("+-")
    if not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", coefficient):
        raise ValueError(f"Invalid numeric value: {value!r}")

    if "." in coefficient:
        digits = coefficient.replace(".", "")
        return len(digits.lstrip("0"))

    digits = coefficient.lstrip("0").rstrip("0")
    return len(digits)


def round_to_significant_figures(
    value: NumberLike,
    figures: int,
    rounding: str = ROUND_HALF_EVEN,
) -> Decimal:
    """Round a value to a specified number of significant figures."""
    if figures < 1:
        raise ValueError("figures must be at least 1.")

    decimal_value = _decimal_from_number(value)
    if decimal_value.is_zero():
        return Decimal("0")

    exponent = decimal_value.adjusted() - figures + 1
    quantum = Decimal(f"1e{exponent}")
    digit_count = len(decimal_value.as_tuple().digits)
    with localcontext() as context:
        context.prec = max(digit_count, figures) + abs(decimal_value.adjusted()) + 4
        return decimal_value.quantize(quantum, rounding=rounding)


def format_significant_figures(
    value: NumberLike,
    figures: int,
    rounding: str = ROUND_HALF_EVEN,
) -> str:
    """Return a string rounded to significant figures, preserving trailing zeros."""
    return str(round_to_significant_figures(value, figures, rounding))


def round_to_decimal_places(
    value: NumberLike,
    places: int,
    rounding: str = ROUND_HALF_EVEN,
) -> Decimal:
    """Round a value to a fixed number of places after the decimal point."""
    if places < 0:
        raise ValueError("places cannot be negative.")
    decimal_value = _decimal_from_number(value)
    return decimal_value.quantize(Decimal(f"1e{-places}"), rounding=rounding)


def add_subtract_with_significant_figures(
    *values: NumberLike,
    rounding: str = ROUND_HALF_EVEN,
) -> Decimal:
    """Add signed values and round to the least precise decimal place."""
    if not values:
        raise ValueError("At least one value is required.")
    decimal_values = [_decimal_from_number(value) for value in values]
    exponent = max(_least_significant_exponent(value) for value in values)
    total = sum(decimal_values, Decimal("0"))
    return total.quantize(Decimal(f"1e{exponent}"), rounding=rounding)


def multiply_with_significant_figures(
    *values: NumberLike,
    rounding: str = ROUND_HALF_EVEN,
) -> Decimal:
    """Multiply values and round to the fewest significant figures in the inputs."""
    if not values:
        raise ValueError("At least one value is required.")

    product = Decimal("1")
    for value in values:
        product *= _decimal_from_number(value)
    return round_to_significant_figures(
        product,
        min(_nonzero_significant_figures(value) for value in values),
        rounding,
    )


def divide_with_significant_figures(
    numerator: NumberLike,
    denominator: NumberLike,
    *more_denominators: NumberLike,
    rounding: str = ROUND_HALF_EVEN,
) -> Decimal:
    """Divide values and round to the fewest significant figures in the inputs."""
    result = _decimal_from_number(numerator)
    for value in (denominator, *more_denominators):
        denominator_value = _decimal_from_number(value)
        if denominator_value.is_zero():
            raise ZeroDivisionError("division by zero.")
        result /= denominator_value

    all_values = (numerator, denominator, *more_denominators)
    return round_to_significant_figures(
        result,
        min(_nonzero_significant_figures(value) for value in all_values),
        rounding,
    )


def absolute_uncertainty_from_relative(value: float, relative_uncertainty: float) -> float:
    """Return absolute uncertainty from a fractional relative uncertainty."""
    return abs(value) * relative_uncertainty


def absolute_uncertainty_from_percent(value: float, percent_uncertainty: float) -> float:
    """Return absolute uncertainty from a percent relative uncertainty."""
    return absolute_uncertainty_from_relative(value, percent_uncertainty / 100.0)


def relative_uncertainty_from_absolute(value: float, uncertainty: float) -> float:
    """Return fractional relative uncertainty from absolute uncertainty."""
    if value == 0:
        if uncertainty == 0:
            return 0.0
        raise ValueError("relative uncertainty is undefined for zero values.")
    return abs(uncertainty / value)


def percent_relative_uncertainty(value: float, uncertainty: float) -> float:
    """Return percent relative uncertainty."""
    return relative_uncertainty_from_absolute(value, uncertainty) * 100.0


def combine_absolute_uncertainties(*uncertainties: float) -> float:
    """Combine independent absolute uncertainties in quadrature."""
    return math.sqrt(sum(uncertainty**2 for uncertainty in uncertainties))


def combine_relative_uncertainties(*relative_uncertainties: float) -> float:
    """Combine independent fractional relative uncertainties in quadrature."""
    return math.sqrt(sum(uncertainty**2 for uncertainty in relative_uncertainties))


def round_measurement(
    measurement: Measurement,
    uncertainty_figures: int = 1,
    rounding: str = ROUND_HALF_EVEN,
) -> tuple[Decimal, Decimal]:
    """Round uncertainty to significant figures and value to the same place."""
    if uncertainty_figures < 1:
        raise ValueError("uncertainty_figures must be at least 1.")
    if measurement.uncertainty == 0:
        return Decimal(str(measurement.value)), Decimal("0")

    rounded_uncertainty = round_to_significant_figures(
        measurement.uncertainty,
        uncertainty_figures,
        rounding,
    )
    exponent = rounded_uncertainty.as_tuple().exponent
    rounded_value = _decimal_from_number(measurement.value).quantize(
        Decimal(f"1e{exponent}"),
        rounding=rounding,
    )
    return rounded_value, rounded_uncertainty


def format_measurement(
    measurement: Measurement,
    uncertainty_figures: int = 1,
    rounding: str = ROUND_HALF_EVEN,
) -> str:
    """Format a measurement as ``value +/- uncertainty``."""
    value, uncertainty = round_measurement(measurement, uncertainty_figures, rounding)
    return f"{value} +/- {uncertainty}"


def log10(measurement: Measurement) -> Measurement:
    """Propagate uncertainty through base-10 logarithm."""
    if measurement.value <= 0:
        raise ValueError("log10 is defined only for positive values.")
    value = math.log10(measurement.value)
    uncertainty = measurement.uncertainty / (measurement.value * math.log(10))
    return Measurement(value, uncertainty)


def ln(measurement: Measurement) -> Measurement:
    """Propagate uncertainty through natural logarithm."""
    if measurement.value <= 0:
        raise ValueError("ln is defined only for positive values.")
    return Measurement(math.log(measurement.value), measurement.uncertainty / measurement.value)


def antilog10(measurement: Measurement) -> Measurement:
    """Propagate uncertainty through ``10 ** measurement``."""
    value = 10**measurement.value
    uncertainty = abs(value) * math.log(10) * measurement.uncertainty
    return Measurement(value, uncertainty)


def exp(measurement: Measurement) -> Measurement:
    """Propagate uncertainty through ``e ** measurement``."""
    value = math.exp(measurement.value)
    return Measurement(value, abs(value) * measurement.uncertainty)


def signed_error(measured_value: float, true_value: float) -> float:
    """Return measured minus true value."""
    return measured_value - true_value


def absolute_error(measured_value: float, true_value: float) -> float:
    """Return the magnitude of measured minus true value."""
    return abs(signed_error(measured_value, true_value))


def percent_error(measured_value: float, true_value: float) -> float:
    """Return signed percent relative error."""
    if true_value == 0:
        raise ValueError("percent error is undefined for a true value of zero.")
    return signed_error(measured_value, true_value) / true_value * 100.0


def classify_accuracy_precision(
    readings: Iterable[float],
    true_value: float,
    accuracy_tolerance: float,
    precision_tolerance: float,
) -> AccuracyPrecision:
    """Classify readings by mean error and sample standard deviation."""
    readings_tuple = tuple(readings)
    if not readings_tuple:
        raise ValueError("At least one reading is required.")
    if accuracy_tolerance < 0 or precision_tolerance < 0:
        raise ValueError("tolerances cannot be negative.")

    mean = sum(readings_tuple) / len(readings_tuple)
    if len(readings_tuple) == 1:
        sample_standard_deviation = 0.0
    else:
        sample_standard_deviation = math.sqrt(
            sum((reading - mean) ** 2 for reading in readings_tuple) / (len(readings_tuple) - 1)
        )

    accurate = abs(mean - true_value) <= accuracy_tolerance
    precise = sample_standard_deviation <= precision_tolerance

    if accurate and precise:
        return AccuracyPrecision.ACCURATE_AND_PRECISE
    if accurate:
        return AccuracyPrecision.ACCURATE_NOT_PRECISE
    if precise:
        return AccuracyPrecision.PRECISE_NOT_ACCURATE
    return AccuracyPrecision.NEITHER


def _coerce_measurement(value: object) -> Measurement:
    if isinstance(value, Measurement):
        return value
    if isinstance(value, (int, float, Decimal)):
        return Measurement(float(value), 0.0)
    raise TypeError(f"Unsupported operand type for Measurement: {type(value).__name__}")


def _normalize_numeric_text(value: NumberLike) -> str:
    text = str(value).strip().replace(",", "").replace(" ", "").replace("\u2212", "-")
    if not text:
        raise ValueError("numeric value cannot be empty.")

    match = _SCIENTIFIC_NOTATION_RE.fullmatch(text)
    if match:
        exponent = match.group("e_exponent") or match.group("power_exponent") or "1"
        return f"{match.group('coefficient')}E{exponent}"
    return text


def _coefficient_text(value: NumberLike) -> str:
    text = str(value).strip().replace(",", "").replace(" ", "").replace("\u2212", "-")
    match = _SCIENTIFIC_NOTATION_RE.fullmatch(text)
    if match:
        return match.group("coefficient")
    if "e" in text.lower():
        return text.lower().split("e", 1)[0]
    return text


def _decimal_from_number(value: NumberLike) -> Decimal:
    return Decimal(_normalize_numeric_text(value))


def _least_significant_exponent(value: NumberLike) -> int:
    decimal_value = _decimal_from_number(value)
    figures = significant_figures(value)
    if figures == 0:
        return decimal_value.as_tuple().exponent
    return decimal_value.adjusted() - figures + 1


def _nonzero_significant_figures(value: NumberLike) -> int:
    figures = significant_figures(value)
    if figures == 0:
        raise ValueError("zero values do not define significant figures.")
    return figures
