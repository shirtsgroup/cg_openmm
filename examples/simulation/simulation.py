import os
from simtk import unit
import cg_openmm
from simtk.openmm.app.pdbfile import PDBFile
from cg_model.cgmodel import CGModel
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
total_steps = int(np.floor(total_simulation_time / simulation_time_step))
temperature = 300.0 * unit.kelvin
friction = 1.0 / unit.picosecond


include_bond_forces = True
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = False

bond_length = 0.2 * unit.nanometer  # reference length unit

# Particle definitions
r_min = 1.5 * bond_length  # Lennard-Jones potential r_min
# Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
sigma = r_min / (2.0 ** (1.0 / 6.0))
epsilon = 0.5 * unit.kilojoules_per_mole
mass = 100.0 * unit.amu

# mass and charge are defaults.
bb = {"particle_type_name": "bb", "sigma": sigma, "epsilon": epsilon, "mass": mass}
sc = {"particle_type_name": "sc", "sigma": sigma, "epsilon": epsilon, "mass": mass}

# Monomer definition
A = {
    "monomer_name": "A",
    "particle_sequence": [bb, sc],
    "bond_list": [[0, 1]],
    "start": 0,
    "end": 0,
}

sequence = 12 * [A]

# Bond definitions
bond_lengths = {"default_bond_length": bond_length}

bond_force_constants = {
    "default_bond_force_constant": 1000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
}

# Bond angle definitions
bond_angle_force_constants = {
    "default_bond_angle_force_constant": 50.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
}

equil_bond_angles = {"default_equil_bond_angle": 120.0 * unit.degrees}

# torsion angle definitions
torsion_force_constants = {"bb_bb_bb_bb_torsion_force_constant": 20.0 * unit.kilojoule_per_mole}

torsion_angle = 78.0 * unit.degrees
equil_torsion_angles = {
    "bb_bb_bb_bb_equil_torsion_angle": torsion_angle,
    "bb_bb_bb_sc_equil_torsion_angle": torsion_angle,
}

torsion_periodicities = {
    "bb_bb_bb_bb_torsion_periodicity": 3,
    "bb_bb_bb_sc_torsion_periodicity": 3
}

# Get initial positions from local file
positions = PDBFile("helix2.pdb").getPositions()

# Build a coarse grained model
cgmodel = CGModel(
    particle_type_list=[bb, sc],
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
    sequence=sequence,
    monomer_types=[A],
)

# Run a simulation
print("Running a simulation.")
run_simulation(
    cgmodel,
    total_simulation_time,
    simulation_time_step,
    temperature,
    friction=friction,
    print_frequency=print_frequency,
    output_directory=output_directory,
)
