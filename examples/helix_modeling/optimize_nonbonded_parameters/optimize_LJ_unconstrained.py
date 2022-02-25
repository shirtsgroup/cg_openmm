import os

import numpy as np
from cg_openmm.utilities.helix_optimize_nonbonded import *

# Optimize LJ sigma parameters of backbone and sidechain beads, for a fixed
# helical radius and pitch, and no constraints on bond lengths.

# Epsilon parameters are assumed equal for backbone and sidechain.

# Helical radius:
radius = 1.0 * unit.angstrom
pitch = 1.0 * unit.angstrom

# Number of backbone particles:
n_particle_bb = 24

opt_solution, geometry = optimize_helix_LJ_parameters(
    radius, pitch, n_particle_bb,
    bond_dist_bb=None, bond_dist_sc=None,
    pdbfile='helix_opt_LJ_params_unconstrained.pdb',
    plotfile='helix_opt_LJ_params_unconstrained.pdf',
    DE_popsize=50)

print(opt_solution)
print(geometry)