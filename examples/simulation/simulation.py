import os
from simtk import unit
import foldamers
import cg_openmm
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from cg_openmm.simulation.tools import run_simulation

###
#
# This example demonstrates how to run an NVT 
# simulation for a coarse grained model in OpenMM,
# using a "CGModel" object built in OpenMM.
#
###

# Job settings
output_directory = 'output'
if not os.path.exists(output_directory):
  os.mkdir(output_directory)

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 10.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))
temperature = 300.0 * unit.kelvin

# Coarse grained model settings
polymer_length=12
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]
include_bond_forces=True
include_bond_angle_forces=True
include_nonbonded_forces=True
include_torsion_forces=True
constrain_bonds=True

# Particle properties
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}
sigma = 2.5 * bond_length
sigmas = {'bb_bb_sigma': sigma,'sc_sc_sigma': sigma}
epsilon = 0.5 * unit.kilocalorie_per_mole
epsilons = {'bb_bb_eps': epsilon,'sc_sc_eps': epsilon}
# Bond angle properties
bond_angle_force_constant = 200 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': bond_angle_force_constant}
equil_bond_angle = 92
equil_bond_angles = {'bb_bb_bb_angle_0': equil_bond_angle,'bb_bb_sc_angle_0': equil_bond_angle}

# Torsion properties
torsion_force_constant = 200
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': torsion_force_constant,'sc_bb_bb_bb_torsion_k': torsion_force_constant,'sc_bb_bb_sc_torsion_k': torsion_force_constant}
equil_torsion_angle = 52
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': equil_torsion_angle,'bb_bb_bb_sc_torsion_0': equil_torsion_angle,'sc_bb_bb_bb_torsion_0': equil_torsion_angle,'sc_bb_bb_sc_torsion_0': equil_torsion_angle}

positions = PDBFile("helix.pdb").getPositions()

# Build a coarse grained model
cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

# Run a simulation
print("Running a simulation.")
run_simulation(cgmodel,output_directory,total_simulation_time,simulation_time_step,temperature,print_frequency)

exit()
