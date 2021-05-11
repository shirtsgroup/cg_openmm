#!~/anaconda3/bin/python

import os
import time

import numpy as np
import simtk.openmm as openmm
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
from openmmtools.cache import global_context_cache
from simtk import unit

# This example demonstrates how to run a OpenMM replica exchange simulation
# using a CGModel object.


rep_exch_begin = time.perf_counter()

#---------------------------------#
# Replica exchange job parameters #
#---------------------------------#

# Set output directory:
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
overwrite_files = True  # overwrite files.

# Use CUDA platform for GPU acceleration:
global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")

# Replica exchange simulation settings:
total_simulation_time = 50.0 * unit.nanosecond
simulation_time_step = 5.0 * unit.femtosecond
total_steps = int(np.floor(total_simulation_time / simulation_time_step))
output_data = os.path.join(output_directory, "output.nc")

number_replicas = 12
min_temp = 200.0 * unit.kelvin
max_temp = 600.0 * unit.kelvin
# Use logarithmic temperature spacing:
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

exchange_frequency = 200 # Number of steps between exchange attempts
collision_frequency = 5 / unit.picosecond

include_bond_forces = True
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = False    

#--------------------------------------------#
# Particle definitions and oligomer topology #
#--------------------------------------------#

mass = 100.0 * unit.amu

# Mass and charge are defaults.
# Backbone particle:
bb = {
    "particle_type_name": "bb",
    "sigma": 2.25 * unit.angstrom,
    "epsilon": 1.50 * unit.kilojoules_per_mole,
    "mass": mass
}
    
# Sidechain particle:
sc = {
    "particle_type_name": "sc",
    "sigma": 3.50 * unit.angstrom,
    "epsilon": 5.00 * unit.kilojoules_per_mole,
    "mass": mass
}

# Monomer definition:
A = {
    "monomer_name": "A",
    "particle_sequence": [bb, sc],
    "bond_list": [[0, 1]],
    "start": 0,
    "end": 0,
}

# Residue sequence:
sequence = 24 * [A]

#--------------------------#
# Harmonic bond parameters #
#--------------------------#

# Bond definitions:
bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

bond_force_constants = {
    "default_bond_force_constant": 15000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
}

#---------------------------#
# Harmonic angle parameters #
#---------------------------#

# Bond angle definitions:
bond_angle_force_constants = {
    "default_bond_angle_force_constant": 150 * unit.kilojoule_per_mole / unit.radian / unit.radian
}

equil_bond_angles = {
    "default_equil_bond_angle": 127.5 * unit.degrees,
    "bb_bb_bb_equil_bond_angle": 105.5 * unit.degrees}

#-----------------------------#
# Periodic torsion parameters #
#-----------------------------#

# Torsion angle definitions:
torsion_force_constants = {
    "default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole,
    "bb_bb_bb_bb_torsion_force_constant": 5 * unit.kilojoule_per_mole}

torsion_phase_angles = {
    "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    "bb_bb_bb_bb_torsion_phase_angle": (16.7-180) * unit.degrees,
    "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
}

torsion_periodicities = {
    "default_torsion_periodicity": 1,
}

# Get initial positions from file:
pdb_path = "initial_structure_trial_1.pdb"
positions = PDBFile(pdb_path).getPositions()

# Build a coarse grained model:
cgmodel = CGModel(
    particle_type_list=[bb, sc],
    bond_lengths=bond_lengths,
    bond_force_constants=bond_force_constants,
    bond_angle_force_constants=bond_angle_force_constants,
    torsion_force_constants=torsion_force_constants,
    equil_bond_angles=equil_bond_angles,
    torsion_phase_angles=torsion_phase_angles,
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

# Store the cg model so that we can do various analyses.
cgmodel.export("stored_cgmodel.pkl")

if not os.path.exists(output_data) or overwrite_files == True:
    run_replica_exchange(
        cgmodel.topology,
        cgmodel.system,
        cgmodel.positions,
        friction=collision_frequency,
        temperature_list=temperature_list,
        simulation_time_step=simulation_time_step,
        total_simulation_time=total_simulation_time,
        exchange_frequency=exchange_frequency,
        output_data=output_data,
    )
else:
    print("Replica output files exist")

rep_exch_end = time.perf_counter()

print(f'replica exchange run time: {rep_exch_end-rep_exch_begin}')
