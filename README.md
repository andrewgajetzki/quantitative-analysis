# Quantitative Analysis Toolkit

Reusable Python modules for quantitative analysis calculations.

The current toolkit focuses on chemical measurement calculations:

- formula and molecular mass calculations
- molarity and solution preparation
- dilution calculations with `M1 V1 = M2 V2`
- weight-percent solutions
- simple unit conversions
- limiting-reagent stoichiometry
- buoyancy corrections for analytical balances
- volumetric glassware and buret calibration
- aqueous temperature corrections
- serial dilution calculations
- significant-figure counting and rounding
- experimental-error and uncertainty propagation
- accuracy/precision and error-category vocabulary
- descriptive statistics and confidence intervals
- F tests, Student's t tests, and Grubbs outlier tests
- least-squares calibration curves and inverse prediction
- detection and quantitation limits
- standard addition, internal standards, and response factors
- spike recovery, dilution factors, matrix effects, and control-chart decisions
- equilibrium constants, reaction quotients, and ICE-table equilibrium solving
- pH, pOH, weak acid/base, conjugate Ka/Kb, buffer, and strong acid/base calculations
- Ksp solubility, common-ion, precipitation, complex-ion, thermodynamic K, and Henry's law helpers
- salt hydrolysis, polyprotic acid distributions, Kp/Kc, gas pressure quotients, and Le Chatelier shifts
- ionic strength, Debye-Huckel activity coefficients, activity-corrected pH/Ksp, and balance checks

## Project Layout

- `main.py` contains runnable examples that demonstrate the current modules.
- `formula.py` parses formulas such as `H2SO4`, `Ca(NO3)2`, and `CuSO4.5H2O`.
- `solutions.py` contains concentration and dilution helpers.
- `equilibrium.py` contains chemical-equilibrium, acid-base, solubility, and thermodynamic helpers.
- `measurements.py` contains analytical balance, statistics, temperature, and calibration helpers.
- `uncertainty.py` contains significant-figure, rounding, uncertainty-propagation, and error helpers.
- `stoichiometry.py` contains balanced-reaction and limiting-reagent helpers.
- `units.py` contains reusable unit converters.
- `constants.py` contains atomic masses and scientific constants.
- `tests/test_chem.py` verifies chemistry calculations.
- `tests/test_equilibrium.py` verifies equilibrium, acid-base, solubility, and thermodynamic calculations.
- `tests/test_measurements.py` verifies balance, glassware, temperature, and calibration calculations.

## Run Examples

```bash
python main.py
```

## Run Tests

```bash
python -m unittest discover -s tests
```

To avoid creating `__pycache__` files while testing:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

## Example Usage

```python
from formula import Formula
from equilibrium import (
    classify_ph,
    debye_huckel_activity_coefficients,
    equilibrium_direction,
    ionic_strength,
    neutral_ph,
    molar_solubility_from_ksp,
    molar_solubility_from_ksp_with_activity,
    ph_from_hydrogen_concentration_activity,
    reaction_quotient,
    salt_solution_character,
    weak_acid_ph,
    will_precipitate,
)
from measurements import (
    concentration_from_internal_standard,
    confidence_interval_mean,
    detection_limit_concentration,
    internal_standard_response_factor,
    linear_least_squares,
    single_standard_addition_concentration,
    standard_addition_from_added_concentrations,
    true_mass_from_apparent,
    water_density_g_per_ml,
)
from solutions import Solution, dilution_volume
from stoichiometry import Reaction
from uncertainty import Measurement, antilog10, format_measurement

print(Formula("Ca(NO3)2").mass)

hbr = Solution.from_weight_percent("HBr", weight_percent=48.0, density_g_per_ml=1.50)
print(hbr.molarity)

stock_ml = dilution_volume(
    concentrated_molarity=18.0,
    dilute_molarity=1.00,
    dilute_volume_ml=1000,
)
print(stock_ml)

reaction = Reaction.from_equation("2 H2 + O2 -> 2 H2O")
result = reaction.limiting_reagent({"H2": 3.0, "O2": 1.0})
print(result.limiting_species)

true_mass = true_mass_from_apparent(
    apparent_mass_g=5.3974,
    object_density_g_per_ml=water_density_g_per_ml(25.0),
)
print(true_mass)

precipitate = Measurement(12.5296, 0.0003) - Measurement(12.4372, 0.0003)
print(format_measurement(precipitate))

hydrogen_ion = antilog10(-Measurement(4.44, 0.04))
print(format_measurement(hydrogen_ion, uncertainty_figures=2))

interval = confidence_interval_mean([116.0, 97.9, 114.2, 106.8, 108.3], confidence=0.90)
print(interval.lower, interval.upper)

fit = linear_least_squares(
    [0.00, 9.36, 18.72, 28.08, 37.44],
    [0.446, 0.676, 0.883, 1.086, 1.280],
)
prediction = fit.inverse_prediction(0.973, confidence=0.95)
prediction_interval = prediction.confidence_interval
print(prediction.x_value, prediction_interval.lower, prediction_interval.upper)

blank_absorbances = [0.0002, 0.0002, 0.0005, 0.0001, 0.0008, 0.0001, 0.0007, 0.0001, 0.0001]
blank_mean = sum(blank_absorbances) / len(blank_absorbances)
print(detection_limit_concentration(2.34e4, intercept=blank_mean, blank_values=blank_absorbances))

print(single_standard_addition_concentration(2.36, 3.79, 25.00, 0.500, 0.0287))

addition = standard_addition_from_added_concentrations(
    [0.0, 2.5, 5.0, 7.5, 10.0],
    [28.0, 34.3, 42.8, 51.5, 58.6],
)
print(addition.unknown_concentration)

factor = internal_standard_response_factor(10222, 8477, 3.47, 1.72)
print(concentration_from_internal_standard(5428, 4431, 2.155, factor))

hi_equilibrium = {"HI": 2, "H2": -1, "I2": -1}
q = reaction_quotient({"HI": 1.0, "H2": 0.10, "I2": 0.10}, hi_equilibrium)
print(equilibrium_direction(q, equilibrium_constant=50.0))

print(weak_acid_ph(0.100, 1.8e-5))
print(neutral_ph(5.5e-13), classify_ph(7.00, 5.5e-13))
print(salt_solution_character(cation_ka=5.6e-10).character)
mgcl2_concentrations = {"Mg2+": 0.025, "Cl-": 0.050}
mgcl2_charges = {"Mg2+": 2, "Cl-": -1}
mgcl2_sizes = {"Mg2+": 800, "Cl-": 300}
print(ionic_strength(mgcl2_concentrations, mgcl2_charges))
print(debye_huckel_activity_coefficients(mgcl2_concentrations, mgcl2_charges, mgcl2_sizes))
print(ph_from_hydrogen_concentration_activity(0.0100, 0.83))
print(molar_solubility_from_ksp(1.8e-10, {"Ag+": 1, "Cl-": 1}))
print(molar_solubility_from_ksp_with_activity(1.8e-10, {"Ag+": 1, "Cl-": 1}, {"Ag+": 0.75, "Cl-": 0.76}))
print(will_precipitate({"Ag+": 1.0e-4, "Cl-": 1.0e-4}, 1.8e-10, {"Ag+": 1, "Cl-": 1}))
```

## Adding New Analysis Areas

Add reusable functions or classes in a focused module, then add short examples in `main.py`.
