#!~/anaconda3/bin/python

import os
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
import numpy as np
import simtk.openmm as openmm
import pickle

from openmmtools.cache import global_context_cache

global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")

###
#
# This example demonstrates how to run a OpenMM replica exchange simulation
# using a "CGModel" object built with the 'foldamers' software package.
#
###

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
overwrite_files = True  # overwrite files.

# Replica exchange simulation settings
total_simulation_time = 0.5 * unit.nanosecond
simulation_time_step = 10.0 * unit.femtosecond
total_steps = int(np.floor(total_simulation_time / simulation_time_step))
output_data = os.path.join(output_directory, "output.nc")
number_replicas = 36
min_temp = 50.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
exchange_frequency = 100  # Number of steps between exchange attempts

include_bond_forces = True
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = False

bond_length = 0.15 * unit.nanometer  # reference length unit

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
    "default_bond_angle_force_constant": 100.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
}

equil_bond_angles = {"default_equil_bond_angle": 120.0 * unit.degrees}

# torsion angle definitions
torsion_force_constants = {"bb_bb_bb_bb_torsion_force_constant": 20.0 * unit.kilojoule_per_mole}

torsion_angle = 78.0 * unit.degrees
equil_torsion_angles = {
    "bb_bb_bb_bb_equil_torsion_angle": torsion_angle,
    "bb_bb_bb_sc_equil_torsion_angle": torsion_angle,
}

torsion_periodicities = {"bb_bb_bb_bb_torsion_periodicity": 3, "bb_bb_bb_sc_torsion_periodicity": 3}

# Get initial positions from local file
positions = PDBFile("helix.pdb").getPositions()

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
    sequence=sequence,
    positions=positions,
    monomer_types=[A],
)

# store the cg model so that we can do various analyses.
cgmodel.export("stored_cgmodel.pkl")

if not os.path.exists(output_data) or overwrite_files == True:
    run_replica_exchange(
        cgmodel.topology,
        cgmodel.system,
        cgmodel.positions,
        temperature_list=temperature_list,
        simulation_time_step=simulation_time_step,
        total_simulation_time=total_simulation_time,
        exchange_frequency=exchange_frequency,
        output_data=output_data,
    )
else:
    print("Replica output files exist")
