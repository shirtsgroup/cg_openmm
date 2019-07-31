#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.iotools import write_pdbfile_without_topology
from cg_openmm.src.build.cg_build import build_topology
from cg_openmm.src.simulation.tools import run_simulation
from cg_openmm.src.simulation.rep_exch import *
from yank.multistate import MultiStateReporter
from yank.multistate import ReplicaExchangeAnalyzer

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)
run_simulations = True

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 10.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas =10
temperature_increment = 5 # unit.kelvin
temperature_list = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(0,number_replicas*temperature_increment,temperature_increment)]
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

###
#
# Coarse grained model settings
#
###

# Global definitions
polymer_length=8
backbone_lengths=[2]
sidechain_lengths=[1]
sidechain_positions=[0]
include_bond_forces=False
include_bond_angle_forces=True
include_nonbonded_forces=True
include_torsion_forces=True
constrain_bonds = True

# Particle properties
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
sigma = 18.0 * unit.angstrom
sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma}
epsilon = 0.5 * unit.kilocalorie_per_mole
epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': 0.5 * epsilon}

# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

# Bond angle properties
bond_angle_force_constant = 200 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': bond_angle_force_constant,'bb_sc_sc_angle_k': bond_angle_force_constant,'sc_sc_sc_angle_k': bond_angle_force_constant, 'sc_bb_sc_angle_k': bond_angle_force_constant, 'sc_sc_bb_angle_k': bond_angle_force_constant}
equil_bond_angle = 120
equil_bond_angles = {'bb_bb_bb_angle_0': equil_bond_angle,'bb_bb_sc_angle_0': equil_bond_angle,'bb_sc_sc_angle_0': equil_bond_angle,'sc_sc_sc_angle_0': equil_bond_angle, 'sc_bb_sc_angle_0': equil_bond_angle,'sc_sc_bb_angle_0': equil_bond_angle}

# Torsion properties
torsion_force_constant = 200
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': torsion_force_constant,'bb_bb_sc_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_sc_torsion_k': torsion_force_constant, 'sc_bb_bb_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_bb_torsion_k': torsion_force_constant, 'sc_sc_sc_sc_torsion_k': torsion_force_constant,  'sc_bb_bb_bb_torsion_k': torsion_force_constant}
equil_torsion_angle = 0
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': equil_torsion_angle,'bb_bb_bb_sc_torsion_0': equil_torsion_angle,'bb_bb_sc_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_bb_torsion_0': equil_torsion_angle, 'sc_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_bb_torsion_0': equil_torsion_angle}

cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds)

cgmodel.topology = build_topology(cgmodel,use_pdbfile=True)

run_simulation(cgmodel,os.getcwd(),total_simulation_time,simulation_time_step,temperature_list[0],print_frequency)

# Run a replica exchange simulation with this cgmodel
if run_simulations:
  replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
else:
  replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)


steps_per_stage = round(total_steps/exchange_attempts)

minimum_energy_structure = get_minimum_energy_pose(cgmodel.topology,replica_energies,replica_positions)

plot_replica_exchange_energies(replica_energies,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)

plot_replica_exchange_summary(replica_states,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)

exit()
