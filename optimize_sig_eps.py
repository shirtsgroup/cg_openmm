#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as pyplot
# OpenMM utilities
import mdtraj as md
from simtk import unit
# foldamers utilities
from foldamers.src.cg_model.cgmodel import CGModel
# cg_openmm utilities
from cg_openmm.src.build.cg_build import build_mm_simulation
from cg_openmm.src.utilities.iotools import read_mm_energies

simulation_time_step = 0.1 * unit.femtosecond
temperature = 300.0 * unit.kelvin
print_frequency = 20 # Number of steps to skip when printing output
total_steps = 10

# Define static model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead(s) on which the side chains are placed
polymer_length = 8 # Number of monomers in the polymer
mass = 10.0 * unit.amu # Mass of beads
bond_length = 1.0 * unit.angstrom # bond length

# Set variable model settings
base_sigma = 2.4 * unit.angstrom # Lennard-Jones interaction distance
base_epsilon = 0.5 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
sigma_list = [(base_sigma).__add__(i * base_sigma.unit) for i in range(-0.4,3.0,0.2)]
epsilon_list = [(base_epsilon).__add__(i * base_epsilon.unit) for i in range(-0.2,5.0,0.2)
sigma_epsilon_list = np.array(len(sigma_list),len(epsilon_list),dtype=float)

for sigma_index in range(len(sigma_list)):
  for epsilon_index in range(len(epsilon_list)):
    sigma = sigma_list[sigma_index]
    epsilon = epsilon_list[epsilon_index]
    # Build a coarse grained model
    cgmodel = basic_cgmodel(polymer_length=polymer_length, backbone_length=backbone_length, sidechain_length=sidechain_length, sidechain_positions=sidechain_positions, mass=mass, sigma=sigma, epsilon=epsilon, bond_length=bond_length)
    # Build a simulation object
    output_file = 'sig_eps_opt_temp.dat'
    simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=temperature,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_file)
    # Run the simulation
    simulation.step(total_steps)
    simulation_data = read_mm_energies(output_file)

figure = pyplot.figure(0)
colors = pyplot.cm.rainbow(np.linspace(0, 1, len(dihedral_list)))
for dihedral, c in zip(dihedrals, colors):
 pyplot.scatter(dihedral, energies, color=c, figure=figure)
pyplot.xlabel("Dihedral Angle (Degrees)")
pyplot.ylabel("Potential Energy (kJ/mol)")
pyplot.title("Dihedral distribution data for simulation of 1,1-CG model")
pyplot.savefig("/CG-11-distribution.png")
pyplot.legend(dihedral_list)
pyplot.show()
pyplot.close()
exit()
