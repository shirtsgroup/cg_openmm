import os
import numpy as np
from cg_openmm.utilities.helix_optimize_geometry import *

# Optimize positions of equally spaced particles along a helix, with 1-2 and 1-3
# nonbonded interactions used instead of bonded terms.

# Particle LJ 12-6 parameters:
sigma = 1.0 * unit.angstrom
epsilon = 1.0 * unit.kilojoule_per_mole

# Number of backbone particles:
n_particle_bb = 24

# Option to add sidechain particles normal to helix (equal bond length to bb-bb)
sidechain=True

opt_solution, geometry = optimize_helix_simple(
    n_particle_bb,sigma,epsilon,sidechain,
    pdbfile='LJ_helix_sidechain.pdb',
    plotfile='LJ_helix_sidechain.pdf',
    DE_popsize=50)

print(opt_solution)
print(geometry)