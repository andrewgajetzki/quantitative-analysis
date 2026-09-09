"""Electrochemistry and electroanalytical technique helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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


@dataclass(frozen=True)
class IonSelectiveCalibration:
    """Empirical calibration for ``E = intercept + slope * log10(activity)``."""

    intercept_v: float
    slope_v_per_decade: float

    def potential(self, activity: float) -> float:
        """Return electrode potential from analyte activity."""
        return ion_selective_potential(
            self.intercept_v,
            activity,
            slope_v_per_decade=self.slope_v_per_decade,
        )

    def activity(self, potential_v: float) -> float:
        """Return analyte activity from electrode potential."""
        return ion_activity_from_potential(
            potential_v,
            self.intercept_v,
            slope_v_per_decade=self.slope_v_per_decade,
        )

    def p_activity(self, potential_v: float) -> float:
        """Return ``-log10(activity)`` from electrode potential."""
        return -math.log10(self.activity(potential_v))


@dataclass(frozen=True)
class IonInterference:
    """One interfering ion in the Nikolsky-Eisenman electrode response."""

    activity: float
    selectivity_coefficient: float
    ion_charge: float


@dataclass(frozen=True)
class CoulometricAnalysisResult:
    """Constant-current coulometric result with optional sample concentration."""

    charge_c: float
    blank_charge_c: float
    analyte_charge_c: float
    moles_electrons: float
    analyte_moles: float
    analyte_molarity: float | None


@dataclass(frozen=True)
class ElectroanalyticalEndpoint:
    """Endpoint found from the intersection of two linear signal branches."""

    endpoint_volume_ml: float
    endpoint_signal: float
    before_slope: float
    before_intercept: float
    after_slope: float
    after_intercept: float


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


def ion_selective_slope_v(
    ion_charge: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return the signed base-10 ISE slope, ``2.303 RT / zF``, in volts."""
    _require_nonzero(ion_charge, "ion_charge")
    _require_positive(temperature_k, "temperature_k")
    return math.log(10.0) * R_J_PER_MOL_K * temperature_k / (
        ion_charge * FARADAY_CONSTANT_C_PER_MOL
    )


def ion_selective_potential(
    intercept_v: float,
    activity: float,
    ion_charge: float | None = None,
    temperature_k: float = STANDARD_TEMPERATURE_K,
    slope_v_per_decade: float | None = None,
) -> float:
    """Return ISE potential for ``E = intercept + slope * log10(activity)``.

    Pass ``ion_charge`` for the theoretical Nernst slope, or
    ``slope_v_per_decade`` for an empirical calibration slope.
    """
    _require_positive(activity, "activity")
    slope = _resolve_ion_selective_slope(ion_charge, temperature_k, slope_v_per_decade)
    return intercept_v + slope * math.log10(activity)


def ion_activity_from_potential(
    potential_v: float,
    intercept_v: float,
    ion_charge: float | None = None,
    temperature_k: float = STANDARD_TEMPERATURE_K,
    slope_v_per_decade: float | None = None,
) -> float:
    """Return ion activity from an ISE potential and calibration intercept."""
    slope = _resolve_ion_selective_slope(ion_charge, temperature_k, slope_v_per_decade)
    return 10.0 ** ((potential_v - intercept_v) / slope)


def ion_selective_intercept(
    potential_v: float,
    activity: float,
    ion_charge: float | None = None,
    temperature_k: float = STANDARD_TEMPERATURE_K,
    slope_v_per_decade: float | None = None,
) -> float:
    """Return the calibration intercept from one potential/activity standard."""
    _require_positive(activity, "activity")
    slope = _resolve_ion_selective_slope(ion_charge, temperature_k, slope_v_per_decade)
    return potential_v - slope * math.log10(activity)


def ion_selective_potential_change(
    initial_activity: float,
    final_activity: float,
    ion_charge: float | None = None,
    temperature_k: float = STANDARD_TEMPERATURE_K,
    slope_v_per_decade: float | None = None,
) -> float:
    """Return the ISE potential change when activity changes."""
    _require_positive(initial_activity, "initial_activity")
    _require_positive(final_activity, "final_activity")
    slope = _resolve_ion_selective_slope(ion_charge, temperature_k, slope_v_per_decade)
    return slope * math.log10(final_activity / initial_activity)


def fit_ion_selective_calibration(
    activities: Iterable[float],
    potentials_v: Iterable[float],
) -> IonSelectiveCalibration:
    """Fit an empirical ISE calibration from activities and potentials."""
    activity_values = tuple(activities)
    potential_values = tuple(potentials_v)
    if len(activity_values) != len(potential_values):
        raise ValueError("activities and potentials_v must have the same length.")
    if len(activity_values) < 2:
        raise ValueError("At least two calibration points are required.")
    for activity in activity_values:
        _require_positive(activity, "activity")

    from measurements import linear_least_squares

    fit = linear_least_squares(
        [math.log10(activity) for activity in activity_values],
        potential_values,
    )
    return IonSelectiveCalibration(
        intercept_v=fit.intercept,
        slope_v_per_decade=fit.slope,
    )


def glass_electrode_slope_v_per_ph(
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return the ideal positive glass-electrode slope per pH unit."""
    return nernst_log10_slope_v(1.0, temperature_k)


def glass_electrode_potential(
    intercept_v: float,
    ph: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
    slope_v_per_ph: float | None = None,
) -> float:
    """Return glass-electrode potential for ``E = intercept - slope * pH``."""
    slope = _resolve_glass_slope(temperature_k, slope_v_per_ph)
    return intercept_v - slope * ph


def glass_electrode_ph(
    potential_v: float,
    reference_potential_v: float,
    reference_ph: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
    slope_v_per_ph: float | None = None,
) -> float:
    """Return sample pH from a one-buffer glass-electrode calibration."""
    slope = _resolve_glass_slope(temperature_k, slope_v_per_ph)
    return reference_ph + (reference_potential_v - potential_v) / slope


def glass_electrode_slope_from_buffers(
    first_ph: float,
    first_potential_v: float,
    second_ph: float,
    second_potential_v: float,
) -> float:
    """Return the positive empirical glass-electrode slope from two buffers."""
    if first_ph == second_ph:
        raise ValueError("Buffer pH values must be distinct.")
    slope = (first_potential_v - second_potential_v) / (second_ph - first_ph)
    if slope <= 0:
        raise ValueError("Potentials must decrease as pH increases.")
    return slope


def glass_electrode_ph_from_two_buffers(
    potential_v: float,
    first_ph: float,
    first_potential_v: float,
    second_ph: float,
    second_potential_v: float,
) -> float:
    """Return sample pH from a two-buffer glass-electrode calibration."""
    slope = glass_electrode_slope_from_buffers(
        first_ph,
        first_potential_v,
        second_ph,
        second_potential_v,
    )
    return glass_electrode_ph(
        potential_v,
        reference_potential_v=first_potential_v,
        reference_ph=first_ph,
        slope_v_per_ph=slope,
    )


def ion_selective_equivalent_primary_activity(
    interfering_activity: float,
    selectivity_coefficient: float,
    primary_charge: float,
    interfering_charge: float,
) -> float:
    """Return the primary-ion activity equivalent to one interfering ion."""
    _require_positive(interfering_activity, "interfering_activity")
    _require_positive(selectivity_coefficient, "selectivity_coefficient")
    exponent = _charge_magnitude(primary_charge, "primary_charge") / _charge_magnitude(
        interfering_charge,
        "interfering_charge",
    )
    return selectivity_coefficient * interfering_activity**exponent


def ion_selective_activity_term(
    primary_activity: float,
    primary_charge: float,
    interferences: Iterable[IonInterference] = (),
) -> float:
    """Return the Nikolsky-Eisenman activity term sensed by an ISE."""
    _require_positive(primary_activity, "primary_activity")
    total = primary_activity
    for interference in interferences:
        total += ion_selective_equivalent_primary_activity(
            interference.activity,
            interference.selectivity_coefficient,
            primary_charge,
            interference.ion_charge,
        )
    return total


def ion_selective_potential_with_interference(
    intercept_v: float,
    primary_activity: float,
    primary_charge: float,
    interferences: Iterable[IonInterference] = (),
    temperature_k: float = STANDARD_TEMPERATURE_K,
    slope_v_per_decade: float | None = None,
) -> float:
    """Return ISE potential including selectivity-coefficient interference."""
    sensed_activity = ion_selective_activity_term(
        primary_activity,
        primary_charge,
        interferences,
    )
    return ion_selective_potential(
        intercept_v,
        sensed_activity,
        primary_charge,
        temperature_k,
        slope_v_per_decade,
    )


def ion_selective_interference_error_percent(
    primary_activity: float,
    interfering_activity: float,
    selectivity_coefficient: float,
    primary_charge: float,
    interfering_charge: float,
) -> float:
    """Return percent relative error in primary activity from one interferent."""
    _require_positive(primary_activity, "primary_activity")
    equivalent_activity = ion_selective_equivalent_primary_activity(
        interfering_activity,
        selectivity_coefficient,
        primary_charge,
        interfering_charge,
    )
    return equivalent_activity / primary_activity * 100.0


def interfering_activity_for_equal_response(
    primary_activity: float,
    selectivity_coefficient: float,
    primary_charge: float,
    interfering_charge: float,
) -> float:
    """Return interferent activity that gives the same response as primary ion."""
    _require_positive(primary_activity, "primary_activity")
    _require_positive(selectivity_coefficient, "selectivity_coefficient")
    exponent = _charge_magnitude(interfering_charge, "interfering_charge") / _charge_magnitude(
        primary_charge,
        "primary_charge",
    )
    return (primary_activity / selectivity_coefficient) ** exponent


def potentiometric_standard_addition_concentration(
    initial_potential_v: float,
    final_potential_v: float,
    sample_volume_ml: float,
    standard_volume_ml: float,
    standard_concentration: float,
    ion_charge: float | None = None,
    temperature_k: float = STANDARD_TEMPERATURE_K,
    slope_v_per_decade: float | None = None,
    final_volume_ml: float | None = None,
) -> float:
    """Return unknown concentration from one potentiometric standard addition.

    The measured potential changes from ``initial_potential_v`` to
    ``final_potential_v`` after adding analyte standard. For an empirical
    electrode, pass ``slope_v_per_decade``; otherwise pass the ion charge.
    """
    _require_positive(sample_volume_ml, "sample_volume_ml")
    _require_positive(standard_volume_ml, "standard_volume_ml")
    _require_positive(standard_concentration, "standard_concentration")
    final_volume = sample_volume_ml + standard_volume_ml
    if final_volume_ml is not None:
        _require_positive(final_volume_ml, "final_volume_ml")
        final_volume = final_volume_ml

    slope = _resolve_ion_selective_slope(ion_charge, temperature_k, slope_v_per_decade)
    activity_ratio = 10.0 ** ((final_potential_v - initial_potential_v) / slope)
    denominator = activity_ratio * final_volume - sample_volume_ml
    if denominator <= 0:
        raise ValueError("Potential change and volumes are inconsistent with standard addition.")
    return standard_concentration * standard_volume_ml / denominator


def coulometric_analyte_moles_from_charge(
    charge_c: float,
    electrons_per_mole_analyte: float,
    current_efficiency: float = 1.0,
    blank_charge_c: float = 0.0,
) -> float:
    """Return analyte moles from coulometric charge and electron stoichiometry."""
    _require_nonnegative(charge_c, "charge_c")
    _require_positive(electrons_per_mole_analyte, "electrons_per_mole_analyte")
    _require_current_efficiency(current_efficiency)
    _require_nonnegative(blank_charge_c, "blank_charge_c")
    net_charge = charge_c - blank_charge_c
    if net_charge < 0:
        raise ValueError("blank_charge_c cannot exceed charge_c.")
    return moles_product_from_charge(
        net_charge * current_efficiency,
        electrons_per_mole_analyte,
    )


def coulometric_analysis(
    current_a: float,
    time_s: float,
    electrons_per_mole_analyte: float,
    sample_volume_ml: float | None = None,
    current_efficiency: float = 1.0,
    blank_current_a: float = 0.0,
    blank_time_s: float = 0.0,
    blank_charge_c: float = 0.0,
) -> CoulometricAnalysisResult:
    """Return a constant-current coulometric analysis result."""
    _require_nonnegative(current_a, "current_a")
    _require_nonnegative(time_s, "time_s")
    _require_positive(electrons_per_mole_analyte, "electrons_per_mole_analyte")
    _require_current_efficiency(current_efficiency)
    _require_nonnegative(blank_current_a, "blank_current_a")
    _require_nonnegative(blank_time_s, "blank_time_s")
    _require_nonnegative(blank_charge_c, "blank_charge_c")

    charge = charge_from_current_time(current_a, time_s)
    blank_charge = blank_charge_c + charge_from_current_time(blank_current_a, blank_time_s)
    analyte_charge = charge - blank_charge
    if analyte_charge < 0:
        raise ValueError("Blank charge cannot exceed total charge.")
    effective_charge = analyte_charge * current_efficiency
    analyte_moles = moles_product_from_charge(effective_charge, electrons_per_mole_analyte)
    analyte_molarity = None
    if sample_volume_ml is not None:
        _require_positive(sample_volume_ml, "sample_volume_ml")
        analyte_molarity = analyte_moles / (sample_volume_ml / 1000.0)

    return CoulometricAnalysisResult(
        charge_c=charge,
        blank_charge_c=blank_charge,
        analyte_charge_c=effective_charge,
        moles_electrons=moles_electrons_from_charge(effective_charge),
        analyte_moles=analyte_moles,
        analyte_molarity=analyte_molarity,
    )


def coulometric_charge_for_analyte(
    analyte_moles: float,
    electrons_per_mole_analyte: float,
    current_efficiency: float = 1.0,
) -> float:
    """Return charge required for a coulometric analyte amount."""
    _require_nonnegative(analyte_moles, "analyte_moles")
    _require_positive(electrons_per_mole_analyte, "electrons_per_mole_analyte")
    _require_current_efficiency(current_efficiency)
    return charge_for_moles_product(analyte_moles, electrons_per_mole_analyte) / current_efficiency


def coulometric_generation_time(
    analyte_moles: float,
    current_a: float,
    electrons_per_mole_analyte: float,
    current_efficiency: float = 1.0,
) -> float:
    """Return time needed to generate or consume analyte coulometrically."""
    _require_positive(current_a, "current_a")
    charge = coulometric_charge_for_analyte(
        analyte_moles,
        electrons_per_mole_analyte,
        current_efficiency,
    )
    return charge / current_a


def diffusion_limited_current(
    electrons_transferred: float,
    electrode_area_cm2: float,
    diffusion_coefficient_cm2_s: float,
    concentration_mol_l: float,
    diffusion_layer_thickness_cm: float,
) -> float:
    """Return steady-state diffusion-limited current in amperes."""
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(electrode_area_cm2, "electrode_area_cm2")
    _require_positive(diffusion_coefficient_cm2_s, "diffusion_coefficient_cm2_s")
    _require_nonnegative(concentration_mol_l, "concentration_mol_l")
    _require_positive(diffusion_layer_thickness_cm, "diffusion_layer_thickness_cm")
    concentration_mol_cm3 = concentration_mol_l / 1000.0
    return (
        electrons_transferred
        * FARADAY_CONSTANT_C_PER_MOL
        * electrode_area_cm2
        * diffusion_coefficient_cm2_s
        * concentration_mol_cm3
        / diffusion_layer_thickness_cm
    )


def concentration_from_diffusion_limited_current(
    limiting_current_a: float,
    electrons_transferred: float,
    electrode_area_cm2: float,
    diffusion_coefficient_cm2_s: float,
    diffusion_layer_thickness_cm: float,
) -> float:
    """Return concentration from a steady-state diffusion-limited current."""
    _require_nonnegative(limiting_current_a, "limiting_current_a")
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(electrode_area_cm2, "electrode_area_cm2")
    _require_positive(diffusion_coefficient_cm2_s, "diffusion_coefficient_cm2_s")
    _require_positive(diffusion_layer_thickness_cm, "diffusion_layer_thickness_cm")
    return (
        limiting_current_a
        * diffusion_layer_thickness_cm
        * 1000.0
        / (
            electrons_transferred
            * FARADAY_CONSTANT_C_PER_MOL
            * electrode_area_cm2
            * diffusion_coefficient_cm2_s
        )
    )


def cottrell_current(
    electrons_transferred: float,
    electrode_area_cm2: float,
    diffusion_coefficient_cm2_s: float,
    concentration_mol_l: float,
    time_s: float,
) -> float:
    """Return chronoamperometric current from the Cottrell equation."""
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(electrode_area_cm2, "electrode_area_cm2")
    _require_positive(diffusion_coefficient_cm2_s, "diffusion_coefficient_cm2_s")
    _require_nonnegative(concentration_mol_l, "concentration_mol_l")
    _require_positive(time_s, "time_s")
    concentration_mol_cm3 = concentration_mol_l / 1000.0
    return (
        electrons_transferred
        * FARADAY_CONSTANT_C_PER_MOL
        * electrode_area_cm2
        * concentration_mol_cm3
        * math.sqrt(diffusion_coefficient_cm2_s)
        / math.sqrt(math.pi * time_s)
    )


def concentration_from_cottrell_current(
    current_a: float,
    electrons_transferred: float,
    electrode_area_cm2: float,
    diffusion_coefficient_cm2_s: float,
    time_s: float,
) -> float:
    """Return concentration from a chronoamperometric Cottrell current."""
    _require_nonnegative(current_a, "current_a")
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(electrode_area_cm2, "electrode_area_cm2")
    _require_positive(diffusion_coefficient_cm2_s, "diffusion_coefficient_cm2_s")
    _require_positive(time_s, "time_s")
    return (
        current_a
        * math.sqrt(math.pi * time_s)
        * 1000.0
        / (
            electrons_transferred
            * FARADAY_CONSTANT_C_PER_MOL
            * electrode_area_cm2
            * math.sqrt(diffusion_coefficient_cm2_s)
        )
    )


def randles_sevcik_peak_current(
    electrons_transferred: float,
    electrode_area_cm2: float,
    diffusion_coefficient_cm2_s: float,
    concentration_mol_l: float,
    scan_rate_v_s: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return reversible cyclic-voltammetry peak current in amperes."""
    _require_positive(electrons_transferred, "electrons_transferred")
    _require_positive(electrode_area_cm2, "electrode_area_cm2")
    _require_positive(diffusion_coefficient_cm2_s, "diffusion_coefficient_cm2_s")
    _require_nonnegative(concentration_mol_l, "concentration_mol_l")
    _require_positive(scan_rate_v_s, "scan_rate_v_s")
    _require_positive(temperature_k, "temperature_k")
    concentration_mol_cm3 = concentration_mol_l / 1000.0
    return (
        0.4463
        * electrons_transferred
        * FARADAY_CONSTANT_C_PER_MOL
        * electrode_area_cm2
        * concentration_mol_cm3
        * math.sqrt(
            electrons_transferred
            * FARADAY_CONSTANT_C_PER_MOL
            * diffusion_coefficient_cm2_s
            * scan_rate_v_s
            / (R_J_PER_MOL_K * temperature_k)
        )
    )


def concentration_from_randles_sevcik_peak_current(
    peak_current_a: float,
    electrons_transferred: float,
    electrode_area_cm2: float,
    diffusion_coefficient_cm2_s: float,
    scan_rate_v_s: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Return concentration from a reversible cyclic-voltammetry peak current."""
    _require_nonnegative(peak_current_a, "peak_current_a")
    reference_current = randles_sevcik_peak_current(
        electrons_transferred,
        electrode_area_cm2,
        diffusion_coefficient_cm2_s,
        concentration_mol_l=1.0,
        scan_rate_v_s=scan_rate_v_s,
        temperature_k=temperature_k,
    )
    return peak_current_a / reference_current


def cyclic_voltammetry_formal_potential(
    anodic_peak_potential_v: float,
    cathodic_peak_potential_v: float,
) -> float:
    """Return E formal from anodic and cathodic cyclic-voltammetry peaks."""
    return (anodic_peak_potential_v + cathodic_peak_potential_v) / 2.0


def cyclic_voltammetry_peak_separation(
    anodic_peak_potential_v: float,
    cathodic_peak_potential_v: float,
) -> float:
    """Return positive cyclic-voltammetry peak separation in volts."""
    return abs(anodic_peak_potential_v - cathodic_peak_potential_v)


def reversible_cv_electron_count_from_peak_separation(
    peak_separation_v: float,
    temperature_k: float = STANDARD_TEMPERATURE_K,
) -> float:
    """Estimate n from the reversible CV relation Delta Ep = 2.303RT/nF."""
    _require_positive(peak_separation_v, "peak_separation_v")
    return nernst_log10_slope_v(1.0, temperature_k) / peak_separation_v


def voltammetric_standard_addition_concentration(
    initial_current_a: float,
    final_current_a: float,
    sample_volume_ml: float,
    standard_volume_ml: float,
    standard_concentration: float,
    final_volume_ml: float | None = None,
) -> float:
    """Return concentration from one voltammetric standard addition."""
    _require_positive(initial_current_a, "initial_current_a")
    _require_positive(final_current_a, "final_current_a")
    _require_positive(sample_volume_ml, "sample_volume_ml")
    _require_positive(standard_volume_ml, "standard_volume_ml")
    _require_positive(standard_concentration, "standard_concentration")
    final_volume = sample_volume_ml + standard_volume_ml
    if final_volume_ml is not None:
        _require_positive(final_volume_ml, "final_volume_ml")
        final_volume = final_volume_ml

    response_ratio = final_current_a / initial_current_a
    denominator = response_ratio * final_volume - sample_volume_ml
    if denominator <= 0:
        raise ValueError("Currents and volumes are inconsistent with standard addition.")
    return standard_concentration * standard_volume_ml / denominator


def amperometric_titration_endpoint(
    before_endpoint_points: Iterable[tuple[float, float]],
    after_endpoint_points: Iterable[tuple[float, float]],
) -> ElectroanalyticalEndpoint:
    """Return amperometric endpoint from two linear current branches."""
    return _two_line_endpoint(before_endpoint_points, after_endpoint_points)


def conductivity_from_resistance(
    resistance_ohm: float,
    cell_constant_cm_inverse: float,
) -> float:
    """Return conductivity in S/cm from resistance and cell constant."""
    _require_positive(resistance_ohm, "resistance_ohm")
    _require_positive(cell_constant_cm_inverse, "cell_constant_cm_inverse")
    return cell_constant_cm_inverse / resistance_ohm


def resistance_from_conductivity(
    conductivity_s_cm: float,
    cell_constant_cm_inverse: float,
) -> float:
    """Return resistance in ohms from conductivity and cell constant."""
    _require_positive(conductivity_s_cm, "conductivity_s_cm")
    _require_positive(cell_constant_cm_inverse, "cell_constant_cm_inverse")
    return cell_constant_cm_inverse / conductivity_s_cm


def molar_conductivity(
    conductivity_s_cm: float,
    concentration_mol_l: float,
) -> float:
    """Return molar conductivity in S cm^2/mol."""
    _require_nonnegative(conductivity_s_cm, "conductivity_s_cm")
    _require_positive(concentration_mol_l, "concentration_mol_l")
    return conductivity_s_cm / (concentration_mol_l / 1000.0)


def conductometric_titration_endpoint(
    before_endpoint_points: Iterable[tuple[float, float]],
    after_endpoint_points: Iterable[tuple[float, float]],
) -> ElectroanalyticalEndpoint:
    """Return conductometric endpoint from two linear conductance branches."""
    return _two_line_endpoint(before_endpoint_points, after_endpoint_points)


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


def _two_line_endpoint(
    before_endpoint_points: Iterable[tuple[float, float]],
    after_endpoint_points: Iterable[tuple[float, float]],
) -> ElectroanalyticalEndpoint:
    before_slope, before_intercept = _linear_fit_points(before_endpoint_points)
    after_slope, after_intercept = _linear_fit_points(after_endpoint_points)
    denominator = before_slope - after_slope
    if denominator == 0:
        raise ValueError("Endpoint branches must not be parallel.")
    endpoint_volume = (after_intercept - before_intercept) / denominator
    endpoint_signal = before_slope * endpoint_volume + before_intercept
    return ElectroanalyticalEndpoint(
        endpoint_volume_ml=endpoint_volume,
        endpoint_signal=endpoint_signal,
        before_slope=before_slope,
        before_intercept=before_intercept,
        after_slope=after_slope,
        after_intercept=after_intercept,
    )


def _linear_fit_points(points: Iterable[tuple[float, float]]) -> tuple[float, float]:
    data = tuple(points)
    if len(data) < 2:
        raise ValueError("At least two points are required.")
    x_mean = sum(point[0] for point in data) / len(data)
    y_mean = sum(point[1] for point in data) / len(data)
    ss_xx = sum((point[0] - x_mean) ** 2 for point in data)
    if ss_xx == 0:
        raise ValueError("Endpoint volumes must not all be identical.")
    ss_xy = sum((point[0] - x_mean) * (point[1] - y_mean) for point in data)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _require_nonzero(value: float, name: str) -> None:
    if value == 0:
        raise ValueError(f"{name} cannot be zero.")


def _require_current_efficiency(value: float) -> None:
    if value <= 0 or value > 1:
        raise ValueError("current_efficiency must be greater than 0 and at most 1.")


def _charge_magnitude(value: float, name: str) -> float:
    _require_nonzero(value, name)
    return abs(value)


def _resolve_ion_selective_slope(
    ion_charge: float | None,
    temperature_k: float,
    slope_v_per_decade: float | None,
) -> float:
    if slope_v_per_decade is not None:
        _require_nonzero(slope_v_per_decade, "slope_v_per_decade")
        return slope_v_per_decade
    if ion_charge is None:
        raise ValueError("Either ion_charge or slope_v_per_decade is required.")
    return ion_selective_slope_v(ion_charge, temperature_k)


def _resolve_glass_slope(
    temperature_k: float,
    slope_v_per_ph: float | None,
) -> float:
    if slope_v_per_ph is not None:
        _require_positive(slope_v_per_ph, "slope_v_per_ph")
        return slope_v_per_ph
    return glass_electrode_slope_v_per_ph(temperature_k)
