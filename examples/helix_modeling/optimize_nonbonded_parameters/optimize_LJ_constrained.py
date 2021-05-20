import os
import numpy as np
from cg_openmm.utilities.helix_optimize_nonbonded import *

# Optimize LJ sigma parameters of backbone and sidechain beads, for a fixed
# helical radius and pitch, and fixed bond lengths.

# Epsilon parameters are assumed equal for backbone and sidechain.

# Helical parameters:
radius = 1.0 * unit.angstrom
pitch = 1.0 * unit.angstrom

# Bond lengths:
bond_dist_bb = 1.5 * unit.angstrom
bond_dist_sc = 1.5 * unit.angstrom

# Number of backbone particles:
n_particle_bb = 24

opt_solution, geometry = optimize_helix_LJ_parameters(
    radius, pitch, n_particle_bb,
    bond_dist_bb=bond_dist_bb, bond_dist_sc=bond_dist_sc,
    pdbfile='helix_opt_LJ_params_constrained.pdb',
    plotfile='helix_opt_LJ_params_constrained.pdf',
    DE_popsize=50)

print(opt_solution)
print(geometry)