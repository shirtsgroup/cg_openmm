import os
import numpy as np
from cg_openmm.utilities.helix_modeling import *

# Optimize positions of equally spaced particles along a helix 

# Particle LJ 12-6 parameters:
sigma_bb = 1.0 * unit.angstrom
sigma_sc = 1.0 * unit.angstrom

epsilon_bb = 1.0 * unit.kilojoule_per_mole
epsilon_sc = 1.0 * unit.kilojoule_per_mole

# Number of backbone particles:
n_particle_bb = 24

opt_solution, geometry = optimize_helix_openmm_energy(
    n_particle_bb, sigma_bb, sigma_sc, epsilon_bb, epsilon_sc,
    pdbfile='LJ_helix_openmm_energy.pdb', plotfile='LJ_helix_openmm_energy.pdf',
    DE_popsize=100)

print(opt_solution)
print(geometry)