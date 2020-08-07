import os
import timeit
from statistics import mean
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.simulation.tools import *

# Simulation settings
total_simulation_time = 100.0 * unit.picosecond
simulation_time_step = 5.0 * unit.femtosecond
temperature = 300.0 * unit.kelvin
friction = 1.0 / unit.picosecond

output_directory = str("model_size_scaling")
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Model settings
polymer_length = 12
backbone_lengths = [1]
sidechain_lengths = [1]
sidechain_positions = [0]
include_bond_forces = True
include_bond_angle_forces = False
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = False

# Bond definitions
bond_length = 7.5 * unit.angstrom
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}
bond_force_constant = 1000 * unit.kilocalorie_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {
    "bb_bb_bond_k": bond_force_constant,
    "bb_sc_bond_k": bond_force_constant,
    "sc_sc_bond_k": bond_force_constant,
}

# Particle definitions
mass = 100.0 * unit.amu
masses = {"backbone_bead_masses": mass, "sidechain_bead_masses": mass}
r_min = 3.0 * bond_length  # Lennard-Jones potential r_min
# Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
sigma = r_min / (2.0 ** (1 / 6))
sigmas = {"bb_sigma": sigma, "sc_sigma": sigma}
# Set the value of epsilon equal to kT at the mean temperature
epsilon = 0.001 * unit.kilocalorie_per_mole
epsilons = {"bb_eps": epsilon, "sc_eps": epsilon}

# Bond angle definitions
bond_angle_force_constant = 0.2 * unit.kilocalorie_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
}
# OpenMM requires angle definitions in units of radians
bb_bb_bb_equil_bond_angle = 120.0 * (np.math.pi / 180.0)
bb_bb_sc_equil_bond_angle = 120.0 * (np.math.pi / 180.0)
equil_bond_angles = {
    "bb_bb_bb_angle_0": bb_bb_bb_equil_bond_angle,
    "bb_bb_sc_angle_0": bb_bb_sc_equil_bond_angle,
}

# Torsion angle definitions
bb_bb_bb_bb_torsion_force_constant = 0.001 * unit.kilocalorie_per_mole / unit.radian / unit.radian
torsion_force_constant = 0.00001 * unit.kilocalorie_per_mole / unit.radian / unit.radian
torsion_force_constants = {
    "bb_bb_bb_bb_torsion_k": bb_bb_bb_bb_torsion_force_constant,
    "bb_bb_bb_sc_torsion_k": torsion_force_constant,
    "sc_bb_bb_sc_torsion_k": torsion_force_constant,
}
# OpenMM defaults to units of radians for angle definitions
bb_bb_bb_bb_equil_torsion_angle = 78.0 * (np.math.pi / 180.0)
equil_torsion_angle = 0.0 * (np.math.pi / 180.0)
equil_torsion_angles = {
    "bb_bb_bb_bb_torsion_0": bb_bb_bb_bb_equil_torsion_angle,
    "bb_bb_bb_sc_torsion_0": equil_torsion_angle,
    "sc_bb_bb_sc_torsion_0": equil_torsion_angle,
}
torsion_periodicity = 1
torsion_periodicities = {
    "bb_bb_bb_bb_period": torsion_periodicity,
    "bb_bb_bb_sc_period": torsion_periodicity,
    "sc_bb_bb_sc_period": torsion_periodicity,
}

wall_clock_time_list = []

for polymer_length in [8, 10, 20, 30]:
    print("Running simulations with 'polymer_length' =" + str(polymer_length))
    cg_model = CGModel(
        polymer_length=polymer_length,
        backbone_lengths=backbone_lengths,
        sidechain_lengths=sidechain_lengths,
        sidechain_positions=sidechain_positions,
        masses=masses,
        sigmas=sigmas,
        epsilons=epsilons,
        bond_lengths=bond_lengths,
        bond_force_constants=bond_force_constants,
        bond_angle_force_constants=bond_angle_force_constants,
        torsion_force_constants=torsion_force_constants,
        equil_bond_angles=equil_bond_angles,
        equil_torsion_angles=equil_torsion_angles,
        torsion_periodicities=torsion_periodicities,
        include_nonbonded_forces=include_nonbonded_forces,
        include_bond_forces=include_bond_forces,
        include_bond_angle_forces=include_bond_angle_forces,
        include_torsion_forces=include_torsion_forces,
        constrain_bonds=constrain_bonds,
        random_positions=True,
    )

    output_data = str(str(output_directory) + "/" + str(polymer_length))
    if not os.path.exists(output_data):
        os.mkdir(output_data)

    start_time = timeit.default_timer()

    run_simulation(
        cgmodel,
        output_directory,
        total_simulation_time,
        simulation_time_step,
        temperature,
		friction,
        print_frequency,
    )

    end_time = timeit.default_timer()
    time = end_time - start_time


exit()
