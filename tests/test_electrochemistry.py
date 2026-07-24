import math
import unittest

from electrochemistry import (
    FARADAY_CONSTANT_C_PER_MOL,
    amp_hours_from_charge,
    amp_hours_from_moles_electrons,
    cell_potential,
    cell_potential_from_delta_g,
    cell_potential_from_equilibrium_constant,
    charge_for_mass,
    charge_from_amp_hours,
    charge_from_current_time,
    concentration_cell_potential,
    current_from_charge_time,
    delta_g_from_cell_potential,
    electrical_energy_j,
    energy_from_current_voltage_time,
    equilibrium_constant_from_cell_potential,
    mass_from_charge,
    mass_from_current_time,
    moles_electrons_from_amp_hours,
    moles_electrons_from_charge,
    moles_product_from_charge,
    nernst_log10_slope_v,
    nernst_potential,
    power_from_current_voltage,
    spontaneous_galvanic_cell,
    standard_cell_potential,
    time_for_mass,
    time_from_charge_current,
)


class CellPotentialTests(unittest.TestCase):
    def test_standard_cell_potential_and_roles_from_reduction_potentials(self):
        cell = spontaneous_galvanic_cell(
            {
                "Cu2+/Cu": 0.340,
                "Zn2+/Zn": -0.763,
            },
            electrons_transferred=2,
        )

        self.assertEqual(cell.cathode, "Cu2+/Cu")
        self.assertEqual(cell.anode, "Zn2+/Zn")
        self.assertTrue(cell.spontaneous)
        self.assertAlmostEqual(cell.standard_cell_potential_v, 1.103)
        self.assertAlmostEqual(standard_cell_potential(0.340, -0.763), 1.103)
        self.assertAlmostEqual(cell.delta_g_standard_j_per_mol, -212_846.6427, places=4)

    def test_delta_g_and_cell_potential_round_trip(self):
        delta_g = delta_g_from_cell_potential(2, 1.103)

        self.assertAlmostEqual(delta_g, -212_846.6427, places=4)
        self.assertAlmostEqual(cell_potential_from_delta_g(2, delta_g), 1.103)

    def test_equilibrium_constant_and_cell_potential_round_trip(self):
        equilibrium_constant = equilibrium_constant_from_cell_potential(2, 1.103)

        self.assertAlmostEqual(math.log10(equilibrium_constant), 37.2891, places=4)
        self.assertAlmostEqual(cell_potential_from_equilibrium_constant(2, equilibrium_constant), 1.103)

    def test_nernst_cell_potential_from_reaction_quotient(self):
        reaction_quotient = 0.0300 / 0.0100**2
        potential = cell_potential(0.7996, 0.340, 2, reaction_quotient)

        self.assertAlmostEqual(nernst_log10_slope_v(1), 0.05915935, places=8)
        self.assertAlmostEqual(potential, 0.38633, places=5)

    def test_nernst_potential_with_ph_term_in_quotient(self):
        hydrogen_activity = 10.0**-3.0
        arsine_pressure_bar = 1.0
        reaction_quotient = arsine_pressure_bar / hydrogen_activity**3

        potential = nernst_potential(-0.238, 3, reaction_quotient)

        self.assertAlmostEqual(potential, -0.41548, places=5)

    def test_concentration_cell_potential(self):
        potential = concentration_cell_potential(0.100, 0.0100, electrons_transferred=1)

        self.assertAlmostEqual(potential, 0.05915935, places=8)


class FaradayElectrolysisTests(unittest.TestCase):
    def test_charge_current_time_and_moles_electrons(self):
        charge = charge_from_current_time(2.00, 1800.0)

        self.assertAlmostEqual(charge, 3600.0)
        self.assertAlmostEqual(current_from_charge_time(charge, 1800.0), 2.00)
        self.assertAlmostEqual(time_from_charge_current(charge, 2.00), 1800.0)
        self.assertAlmostEqual(moles_electrons_from_charge(charge), 3600.0 / FARADAY_CONSTANT_C_PER_MOL)

    def test_metal_deposition_from_current_time(self):
        silver_mass = mass_from_current_time(
            current_a=1.00,
            time_s=3600.0,
            molar_mass_g_per_mol=107.8682,
            electrons_per_mole_product=1,
        )

        self.assertAlmostEqual(silver_mass, 4.0247, places=4)

    def test_reactant_mass_consumed_from_charge(self):
        charge = charge_from_current_time(1000.0, 3600.0)
        chlorine_moles = moles_product_from_charge(charge, electrons_per_mole_product=2)
        chlorine_mass = mass_from_charge(charge, 70.906, electrons_per_mole_product=2)

        self.assertAlmostEqual(chlorine_moles, 18.6557, places=4)
        self.assertAlmostEqual(chlorine_mass / 1000.0, 1.3228, places=4)

    def test_charge_and_time_needed_for_target_mass(self):
        charge = charge_for_mass(1.00, molar_mass_g_per_mol=63.546, electrons_per_mole_product=2)
        time = time_for_mass(1.00, 0.500, molar_mass_g_per_mol=63.546, electrons_per_mole_product=2)

        self.assertAlmostEqual(charge, 3036.7, places=1)
        self.assertAlmostEqual(time, 6073.4, places=1)

    def test_amp_hours_and_electrical_energy(self):
        charge = charge_from_amp_hours(2.50)

        self.assertAlmostEqual(charge, 9000.0)
        self.assertAlmostEqual(amp_hours_from_charge(charge), 2.50)
        self.assertAlmostEqual(amp_hours_from_moles_electrons(1.0), 26.8015, places=4)
        self.assertAlmostEqual(moles_electrons_from_amp_hours(26.8015), 1.0000, places=4)
        self.assertAlmostEqual(power_from_current_voltage(2.00, 1.50), 3.00)
        self.assertAlmostEqual(electrical_energy_j(9000.0, 1.50), 13_500.0)
        self.assertAlmostEqual(energy_from_current_voltage_time(2.00, 1.50, 10.0), 30.0)


if __name__ == "__main__":
    unittest.main()
