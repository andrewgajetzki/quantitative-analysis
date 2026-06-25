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

## Project Layout

- `main.py` contains runnable examples that demonstrate the current modules.
- `formula.py` parses formulas such as `H2SO4`, `Ca(NO3)2`, and `CuSO4.5H2O`.
- `solutions.py` contains concentration and dilution helpers.
- `measurements.py` contains analytical balance, temperature, and glassware calibration helpers.
- `stoichiometry.py` contains balanced-reaction and limiting-reagent helpers.
- `units.py` contains reusable unit converters.
- `constants.py` contains atomic masses and scientific constants.
- `tests/test_chem.py` verifies chemistry calculations.
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
from measurements import true_mass_from_apparent, water_density_g_per_ml
from solutions import Solution, dilution_volume
from stoichiometry import Reaction

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
```

## Adding New Analysis Areas

Add reusable functions or classes in a focused module, then add short examples in `main.py`.
For example, a future equilibrium topic could live in `equilibrium.py` with tests in
`tests/test_equilibrium.py`.
