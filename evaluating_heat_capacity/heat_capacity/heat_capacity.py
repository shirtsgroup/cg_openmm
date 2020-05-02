import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.thermo.calc import *
from foldamers.parameters.reweight import *
from cg_openmm.simulation.rep_exch import *

# Job settings
top_directory = "output"
if not os.path.exists(top_directory):
    os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 5  # Number of steps to skip when printing output
total_simulation_time = 0.5 * unit.nanosecond  # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond

# Yank (replica exchange) simulation settings
output_data = str(str(top_directory) + "/output.nc")
number_replicas = 30
min_temp = 1.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

# Model settings
polymer_length = 12
backbone_lengths = [1]
sidechain_lengths = [1]
sidechain_positions = [0]
include_bond_forces = False
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = True

# Bond definitions
bond_length = 7.5 * unit.angstrom
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}
bond_force_constant = 0 * unit.kilocalorie_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {
    "bb_bb_bond_k": bond_force_constant,
    "bb_sc_bond_k": bond_force_constant,
    "sc_sc_bond_k": bond_force_constant,
}

# Particle definitions
mass = 100.0 * unit.amu
masses = {"backbone_bead_masses": mass, "sidechain_bead_masses": mass}
r_min = 3.0 * bond_length  # Lennard-Jones potential r_min
sigma = r_min / (2.0 ** (1 / 6))  # Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
sigmas = {"bb_sigma": sigma, "sc_sigma": sigma}
epsilon = 0.05 * unit.kilocalorie_per_mole
epsilons = {"bb_eps": epsilon, "sc_eps": epsilon}

# Bond angle definitions
bond_angle_force_constant = 0.0001 * unit.kilocalorie_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
}
bb_bb_bb_equil_bond_angle = 120.0 * (
    3.14 / 180.0
)  # OpenMM expects angle definitions in units of radians
bb_bb_sc_equil_bond_angle = 120.0 * (3.14 / 180.0)
equil_bond_angles = {
    "bb_bb_bb_angle_0": bb_bb_bb_equil_bond_angle,
    "bb_bb_sc_angle_0": bb_bb_sc_equil_bond_angle,
}

# Torsion angle definitions (Used to establish a scanning range below)
torsion_force_constant = 0.01 * unit.kilocalorie_per_mole / unit.radian / unit.radian
torsion_force_constants = {"bb_bb_bb_bb_torsion_k": torsion_force_constant}
bb_bb_bb_bb_equil_torsion_angle = 78.0 * (
    3.14 / 180.0
)  # OpenMM defaults to units of radians for angle definitions
equil_torsion_angles = {"bb_bb_bb_bb_torsion_0": bb_bb_bb_bb_equil_torsion_angle}
torsion_periodicities = {"bb_bb_bb_bb_period": 1}

# Get initial positions from local file
positions = PDBFile("helix.pdb").getPositions()

# Build a coarse grained model
cgmodel = CGModel(
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
    positions=positions,
)

replica_energies, replica_positions, replica_states = run_replica_exchange(
    cgmodel.topology,
    cgmodel.system,
    cgmodel.positions,
    temperature_list=temperature_list,
    simulation_time_step=simulation_time_step,
    total_simulation_time=total_simulation_time,
    print_frequency=print_frequency,
    output_data=output_data,
)

C_v, dC_v, new_temperature_list = get_heat_capacity(
    replica_energies, temperature_list, num_intermediate_states=1
)

exit()
