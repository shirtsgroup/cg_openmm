import os
from simtk import unit
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from foldamers.cg_model.cgmodel import CGModel
from cg_openmm.simulation.rep_exch import *
import pickle

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
overwrite_files = True  # overwrite files.

# Coarse grained model settings
polymer_length = 24
backbone_lengths = [1]
sidechain_lengths = [1]
sidechain_positions = [0]
include_bond_forces = True
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = False
constrain_bonds = False
random_positions = True

# Replica exchange simulation settings
total_simulation_time = 0.5 * unit.nanosecond
simulation_time_step = 10.0 * unit.femtosecond
total_steps = int(np.floor(total_simulation_time / simulation_time_step))
number_replicas = 24
min_temp = 50.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
exchange_frequency = 100  # Number of steps between exchange attempts
output_data = os.path.join(output_directory, "output.nc")

# Bond definitions
bond_length = 1.5 * unit.angstrom
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
r_min = 1.5 * bond_length  # Lennard-Jones potential r_min
# Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
sigma = r_min / (2.0 ** (1.0 / 6.0))
sigmas = {"bb_sigma": sigma, "sc_sigma": sigma}
epsilon = 0.5 * unit.kilojoule_per_mole
epsilons = {"bb_eps": epsilon, "sc_eps": epsilon}

# Bond angle definitions
bond_angle_force_constant = 100 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
}
# OpenMM requires angle definitions in units of radians
bb_bb_bb_equil_bond_angle = 120.0 * unit.degrees
bb_bb_sc_equil_bond_angle = 120.0 * unit.degrees
equil_bond_angles = {
    "bb_bb_bb_angle_0": bb_bb_bb_equil_bond_angle,
    "bb_bb_sc_angle_0": bb_bb_sc_equil_bond_angle,
}

# Torsion angle definitions
torsion_force_constant = 20.0 * unit.kilojoule_per_mole
torsion_force_constants = {"bb_bb_bb_bb_torsion_k": torsion_force_constant}
# OpenMM requires angle definitions in units of radians
bb_bb_bb_bb_equil_torsion_angle = 60.0 * unit.degrees
bb_bb_bb_sc_equil_torsion_angle = 60.0 * unit.degrees
equil_torsion_angles = {"bb_bb_bb_bb_torsion_0": bb_bb_bb_bb_equil_torsion_angle,
                        "bb_bb_bb_sc_torsion_0": bb_bb_bb_sc_equil_torsion_angle}
torsion_periodicities = {"bb_bb_bb_bb_period": 3, "bb_bb_bb_sc_period": 3}

# Get initial positions from local file
positions = PDBFile("24mer_1b1s_initial_structure.pdb").getPositions()

# Build a coarse-grained model
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

# Save the coarse-grained model
cgmodel.export("stored_cgmodel.pkl")

# Run replica exchange simulation
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
        output_directory=output_directory,
    )
else:
    print("Replica output files exist")
    
