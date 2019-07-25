#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from cg_openmm.src.simulation.rep_exch import replica_exchange, get_replica_energies

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 1.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
number_replicas = 10
temperature_list = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(0,30,number_replicas)]

###
#
# Coarse grained model settings
#
###

# Global definitions
polymer_length=8
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]

# Particle properties
mass = 10.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
sigmas = {'bb_bb_sigma': 2.5 * unit.angstrom,'bb_sc_sigma': 2.5 * unit.angstrom,'sc_sc_sigma': 2.5 * unit.angstrom}
epsilons = {'bb_bb_eps': 0.5 * unit.kilocalorie_per_mole,'bb_sc_eps': 0.5 * unit.kilocalorie_per_mole,'sc_sc_eps': 0.5 * unit.kilocalorie_per_mole}

# Bonded interaction properties
bond_length = 1.0 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
constrain_bonds = False
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

top_directory = 'replica_exchange_simulation'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=True,include_bond_angle_forces=True,include_torsion_forces=True,constrain_bonds=constrain_bonds)

# Run a replica exchange simulation with this cgmodel
replica_energies,reduced_replica_energies,replica_positions,temperature_list = replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency)

# Process and analyze the replica exchange simulation data

figure = pyplot.figure(0)
for replica in range(len(replica_energies)):
  simulation_times = np.array([float(int(step)*simulation_time_step._value) for step in range(len(replica_energies[replica]))])
  energies = np.array([float(energy) for energy in replica_energies[replica]])
  pyplot.plot(simulation_times,energies,figure=figure)
pyplot.xlabel("Simulation Time ( Femtoseconds )")
pyplot.ylabel("Potential Energy ( kJ / mol )")
pyplot.title("Replica exchange energies")
pyplot.savefig(str(str(top_directory)+"/replica_exchange_energies.png"))
pyplot.show()
pyplot.close()


exit()
