import os
from simtk import unit
import foldamers
import cg_openmm
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from cg_openmm.simulation.tools import run_simulation
import numpy as np

###
#
# This example demonstrates how to run an NVT
# simulation for a coarse grained model in OpenMM,
# using a "CGModel" object built with the 'foldamers' software package.
#
###

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# OpenMM simulation settings
print_frequency = 5000  # Number of steps to skip when printing output
total_simulation_time = 2.0 * unit.nanosecond  # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))
temperature = 300.0 * unit.kelvin
friction = 500.0 / unit.picosecond

# Coarse grained model settings
polymer_length = 12
backbone_lengths = [2]
sidechain_lengths = [2]
sidechain_positions = [0]
include_bond_forces = True
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = False

# Bond definitions
bond_length = 0.1 * unit.nanometer
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}
bond_force_constant = 1000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {
    "bb_bb_bond_k": bond_force_constant,
    "bb_sc_bond_k": bond_force_constant,
    "sc_sc_bond_k": bond_force_constant,
}

# Particle definitions
mass = 100.0 * unit.amu
masses = {"backbone_bead_masses": mass, "sidechain_bead_masses": mass}
r_min = 1.5 * bond_length  # Lennard-Jones potential r_min
# Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
sigma = r_min / (2.0 ** (1.0 / 6.0))
sigmas = {"bb_sigma": sigma, "sc_sigma": sigma}
epsilon = 2.0 * unit.kilojoule_per_mole
epsilons = {"bb_eps": epsilon, "sc_eps": epsilon}

# Bond angle definitions
bond_angle_force_constant = 15.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
}
bb_bb_bb_equil_bond_angle = 120.0 * unit.degrees
bb_bb_sc_equil_bond_angle = 120.0 * unit.degrees
equil_bond_angles = {
    "bb_bb_bb_angle_0": bb_bb_bb_equil_bond_angle,
    "bb_bb_sc_angle_0": bb_bb_sc_equil_bond_angle,
}

# Torsion angle definitions
torsion_force_constant = 20.0 * unit.kilojoule_per_mole
torsion_force_constants = {"bb_bb_bb_bb_torsion_k": torsion_force_constant}

equil_torsion_angle = 0.0 * unit.degrees

equil_torsion_angles = {
    "bb_bb_bb_bb_torsion_0": equil_torsion_angle,
    "bb_bb_bb_sc_torsion_0": equil_torsion_angle,
    "bb_bb_sc_sc_torsion_0": equil_torsion_angle,
}

torsion_periodicities = {"bb_bb_bb_bb_period": 3, "bb_bb_bb_sc_period":3}

# Get initial positions from local file
positions = PDBFile("12mer_2b2s_initial_structure.pdb").getPositions()

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

# Run a simulation
print("Running a simulation.")
run_simulation(
    cgmodel,
    output_directory,
    total_simulation_time,
    simulation_time_step,
    temperature,
    friction,
    print_frequency,
)
