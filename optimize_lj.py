#!/usr/bin/python

import os
from simtk import unit
import matplotlib.pyplot as pyplot
from foldamers.src.cg_model.cgmodel import CGModel
from cg_openmm.src.simulation.tools import * 
from cg_openmm.src.build.cg_build import *
from cg_openmm.src.utilities.iotools import *
from foldamers.src.utilities.util import random_positions

###
#
# OpenMM simulation settings
#
###

temperature = 300.0 * unit.kelvin
print_frequency = 1 # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.picosecond # Units = picoseconds
simulation_time_step = 1.0 * unit.femtosecond
total_steps = total_simulation_time.__div__(simulation_time_step)

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

cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=True,include_bond_angle_forces=True,include_torsion_forces=True,constrain_bonds=constrain_bonds)
positions = random_positions(cgmodel)

top_directory = 'optimize_lj'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

number_parameters_sampled = 20
max_sigma = 7.0 * unit.angstrom
min_sigma = 2.0 * unit.angstrom
sigma_step_size = ( max_sigma - min_sigma ) / number_parameters_sampled
sigma_list = [min_sigma+step*sigma_step_size for step in range(1,number_parameters_sampled+1)]

max_epsilon = 7.0 * unit.kilocalorie_per_mole
min_epsilon = 0.0 * unit.kilocalorie_per_mole
epsilon_step_size = ( max_epsilon - min_epsilon ) / number_parameters_sampled
epsilon_list = [step*epsilon_step_size for step in range(1,number_parameters_sampled+1)]

results = {'sigmas': [], 'epsilons': [], 'minimum_energy': []}

for sigma in sigma_list:
 for epsilon in epsilon_list:
  sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma}
  epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': 0.5 * epsilon}

  cgmodel = CGModel(positions=positions,polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=True,include_bond_angle_forces=True,include_torsion_forces=True,constrain_bonds=constrain_bonds)
  run_simulation(cgmodel,top_directory,total_simulation_time,simulation_time_step,temperature,print_frequency)
  simulation_data = read_mm_energies(str(str(top_directory)+'/simulation.dat'))
  print("Minimizing a model with a sigma value of "+str(sigma)+" and an epsilon value of "+str(epsilon))
  minimum_energy = min(simulation_data['potential_energy'])
  results['sigmas'].append(sigma)
  results['epsilons'].append(epsilon)
  results['minimum_energy'].append(minimum_energy)

# Reshape data for plotting
x = np.unique(np.array([float(sigma._value) for sigma in sigma_list]))
y = np.unique(np.array([float(epsilon._value) for epsilon in epsilon_list]))
z = np.unique(np.array([float(minimum_energy) for minimum_energy in results['minimum_energy']]))
X,Y = np.meshgrid(x,y)
Z = z.reshape(len(x),len(y))

# Plot
figure = pyplot.figure(0)
pyplot.pcolormesh(X,Y,Z)
pyplot.xlabel("Sigma ( Angstroms )")
pyplot.ylabel("Epsilon ( kCal/mol )")
pyplot.title("Minimum Potential Energy ( kJ / mol )")
pyplot.savefig(str(str(top_directory)+"/lj_energies_heatmap.png"))
pyplot.colorbar()
pyplot.show()
pyplot.close()


exit()
