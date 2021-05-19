import os
import numpy as np
from cg_openmm.utilities.helix_modeling import *

# Optimize positions of equally spaced particles along a helix, with constraints
# on bb-bb and bb-sc bond lengths.

# Particle LJ 12-6 parameters:
sigma_bb = 1.0 * unit.angstrom
sigma_sc = 1.0 * unit.angstrom

epsilon_bb = 1.0 * unit.kilojoule_per_mole
epsilon_sc = 1.0 * unit.kilojoule_per_mole

# Number of backbone particles:
n_particle_bb = 24

# Bond constraints (equilibrium values)
bond_dist_bb = 1.08 * unit.angstrom
bond_dist_sc = 1.08 * unit.angstrom

opt_solution, geometry = optimize_helix_openmm_energy(
    n_particle_bb, sigma_bb, sigma_sc, epsilon_bb, epsilon_sc,
    bond_dist_bb=bond_dist_bb, bond_dist_sc=bond_dist_sc,
    pdbfile='LJ_helix_openmm_constrained.pdb', plotfile='LJ_helix_openmm_constrained.pdf',
    DE_popsize=100)

print(opt_solution)
print(geometry)