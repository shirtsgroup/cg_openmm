#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as pyplot
from statistics import mean
from pymbar import MBAR
# OpenMM utilities
import mdtraj as md
from simtk import unit
# foldamers utilities
from foldamers.src.cg_model.cgmodel import basic_cgmodel
from foldamers.src.utilities.util import distance
# cg_openmm utilities
from cg_openmm.src.simulation.rep_exch import replica_exchange

# Simulation settings
simulation_time_step = 0.01 * unit.femtosecond
total_simulation_time = 1.0 * unit.picosecond

# Define static model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead(s) on which the side chains are placed
polymer_length = 8 # Number of monomers in the polymer
mass = 10.0 * unit.amu # Mass of beads
bond_length = 1.0 * unit.angstrom # bond length
epsilon = 1.0 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength

# Set variable model settings
base_sigma = 2.4 * unit.angstrom # Lennard-Jones interaction distance
sigma_list = [(base_sigma).__add__(i * base_sigma.unit) for i in [ j * 0.2 for j in range(-2,10,1)]]

expected_sigma_results = []
for sigma_index in range(len(sigma_list)):
    sigma = sigma_list[sigma_index]
    print("Evaluating the energy for a model with:")
    print("sigma="+str(sigma))

    # Build a coarse grained model
    cgmodel = basic_cgmodel(polymer_length=polymer_length, backbone_length=backbone_length, sidechain_length=sidechain_length, sidechain_positions=sidechain_positions, mass=mass, sigma=sigma, epsilon=epsilon, bond_length=bond_length)
    
    # Run a replica exchange simulation for this model:
    # replica_energies are returned in u_kn format for subsequent pymbar analysis
    Delta_f_ij,dDelta_f_ij,replica_energies,replica_positions,temperatures = replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,test_time_step=False)

# Construct a u_kn matrix with the non-bonded distances for each configuration to calculate their mean sigma values, for subsequent reweighting
    nonbonded_interaction_list = cgmodel.nonbonded_interaction_list
    sigma_kn = np.zeros([len(temperatures),len(replica_positions[0])*len(replica_positions[0])])
#print(range(len(temperatures)))
#print(range(len(replica_positions[0])))
    for replica in range(len(temperatures)):
      for pose in range(len(replica_positions[replica])):
        positions = replica_positions[replica][replica][pose]
        nonbonded_distances = []
        j = 0
        for k in range(len(replica_positions[0])):
          for position_1 in range(len(positions)):
            for position_2 in range(position_1+1,len(positions)):
              if [position_1,position_2] in nonbonded_interaction_list: 
                nonbonded_distances.append(float(distance(positions[position_1],positions[2]).in_units_of(unit.angstrom)._value))
        mean_sigma = mean(nonbonded_distances)
#    print(replica)
#    print(pose)
        sigma_kn[replica][:][j] = mean_sigma
        j = j + 1

# Initialize MBAR, in order to calculate the expectation value and variance of sigma, measured relative to the bond length.
    u_kn = np.zeros([len(temperatures),len(replica_positions[0])*len(replica_positions[0])])
    for i in range(len(temperatures)):
      j = 0
      for k in range(len(replica_positions[0])):
        u_kn[i][j] = replica_energies[i][j][k]
        j = j + 1
    N_k = [len(replica_positions[0]) for replica in range(len(temperatures))]
    mbar = MBAR(u_kn,N_k)
    results = mbar.computeExpectations(sigma_kn,compute_uncertainty=False)
    relative_sigma_expectation = mean(results[0])/bond_length.in_units_of(unit.angstrom)._value
    expected_sigma_results.append(round(relative_sigma_expectation,3))

y = np.array([float(expect) for expect in expected_sigma_results])
x = np.array([float(sigma._value) for sigma in sigma_list])
figure = pyplot.figure(0)
pyplot.plot(x,y,figure=figure)
pyplot.xlabel("Lennard-Jones Sigma value (Relative length units)")
pyplot.ylabel("<Sigma> (Relative length units)")
pyplot.title("<Sigma> (relative to bond length unit of 1 A)")
pyplot.savefig(str("sigma_expectation_value.png"))
pyplot.show()
pyplot.close()


exit()
