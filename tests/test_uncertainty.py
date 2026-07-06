import math
import unittest

from uncertainty import (
    AccuracyPrecision,
    Measurement,
    add_subtract_with_significant_figures,
    antilog10,
    classify_accuracy_precision,
    combine_absolute_uncertainties,
    divide_with_significant_figures,
    format_measurement,
    format_significant_figures,
    log10,
    multiply_with_significant_figures,
    percent_relative_uncertainty,
    round_measurement,
    significant_figures,
)


class SignificantFigureTests(unittest.TestCase):
    def test_counts_significant_figures_from_textbook_notation(self):
        self.assertEqual(significant_figures("1.9030"), 5)
        self.assertEqual(significant_figures("0.03910"), 4)
        self.assertEqual(significant_figures("1.40 x 10^4"), 3)
        self.assertEqual(significant_figures("2005"), 4)
        self.assertEqual(significant_figures("2000"), 1)
        self.assertEqual(significant_figures("2000."), 4)

    def test_rounds_tie_to_even_for_significant_figures(self):
        self.assertEqual(format_significant_figures("0.216500", 3), "0.216")
        self.assertEqual(format_significant_figures("0.21674", 3), "0.217")

    def test_addition_subtraction_uses_least_precise_decimal_place(self):
        self.assertEqual(str(add_subtract_with_significant_figures("1.021", "2.69")), "3.71")
        self.assertEqual(str(add_subtract_with_significant_figures("12.3", "-1.63")), "10.7")

    def test_multiplication_division_use_fewest_significant_figures(self):
        self.assertEqual(str(multiply_with_significant_figures("2.34", "1.2")), "2.8")
        self.assertEqual(str(divide_with_significant_figures("9.43", "0.016")), "5.9E+2")


class UncertaintyPropagationTests(unittest.TestCase):
    def test_absolute_uncertainties_combine_for_subtraction(self):
        precipitate = Measurement(12.5296, 0.0003) - Measurement(12.4372, 0.0003)

        self.assertAlmostEqual(precipitate.value, 0.0924, places=6)
        self.assertAlmostEqual(
            precipitate.uncertainty,
            combine_absolute_uncertainties(0.0003, 0.0003),
            places=9,
        )
        self.assertEqual(format_measurement(precipitate), "0.0924 +/- 0.0004")

    def test_relative_uncertainties_combine_for_multiplication(self):
        product = Measurement(3.24, 0.05) * Measurement(3.24, 0.03)

        self.assertAlmostEqual(product.value, 10.4976)
        self.assertAlmostEqual(product.uncertainty, 0.1889228, places=7)
        self.assertAlmostEqual(
            percent_relative_uncertainty(product.value, product.uncertainty),
            1.79968,
            places=5,
        )
        self.assertEqual(
            tuple(str(value) for value in round_measurement(product)),
            ("10.5", "0.2"),
        )

    def test_logarithm_uncertainty(self):
        logged = log10(Measurement(4.218, 0.010))

        self.assertAlmostEqual(logged.value, math.log10(4.218), places=12)
        self.assertAlmostEqual(logged.uncertainty, 0.00102962, places=8)

    def test_antilogarithm_uncertainty_for_ph(self):
        hydrogen_ion = antilog10(-Measurement(4.44, 0.04))

        self.assertAlmostEqual(hydrogen_ion.value, 3.6307805e-5, places=12)
        self.assertAlmostEqual(hydrogen_ion.percent_relative_uncertainty, 9.21034, places=5)
        self.assertAlmostEqual(hydrogen_ion.uncertainty, 3.344e-6, places=9)


class AccuracyPrecisionTests(unittest.TestCase):
    def test_classifies_accurate_and_precise_readings(self):
        classification = classify_accuracy_precision(
            [9.99, 10.00, 10.01],
            true_value=10.0,
            accuracy_tolerance=0.03,
            precision_tolerance=0.02,
        )

        self.assertEqual(classification, AccuracyPrecision.ACCURATE_AND_PRECISE)

    def test_classifies_precise_but_not_accurate_readings(self):
        classification = classify_accuracy_precision(
            [10.48, 10.50, 10.52],
            true_value=10.0,
            accuracy_tolerance=0.03,
            precision_tolerance=0.03,
        )

        self.assertEqual(classification, AccuracyPrecision.PRECISE_NOT_ACCURATE)


if __name__ == "__main__":
    unittest.main()
