import copy
import numpy as np
from simtk import openmm, unit

# This script was provided by Andea Rizzi,
# with addt'l changes by Garrett A. Meek
# on 3/23/19

# LJ_parameter_pairs is a dictionary atom_index -> (sigma, epsilon)
# with the new parameters that are the target of the reweighting.
LJ_parameter_pairs = ...

# Create system, and load the trajectory to be reweighted.
system = ...  # You can create this with openmmtools.testsystems or openmm.app.AmberPrmtopFile/GromacsTopFile if you have a file.
positions_trajectory = ...  # positions_trajectory[i] is the (n_atoms x 3) matrix of positions for the i-th simulation frame.
box_vectors_trajectory = ...  # box_vectors_trajectory[i] is the (3 x 3) matrix of box vectors for the i-th simulation frame.

# Get the nonbonded force.
forces = {force.__class__.__name__: force for force in system.getForces()}
nonbonded_force = forces['NonbondedForce']

# Move electrostatics into a different force object since we don't need to change them.
nonbonded_force_copy = copy.deepcopy(nonbonded_force)
# Turn off charges of the original force and LJ parameters in the copied force.
for particle_index in range(nonbonded_force.getNumParticles()):
    # Retrieve parameters.
    charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
    # Turn off electrostatics in original force.
    nonbonded_force.setParticleParameters(particle_index, 0.0, sigma, epsilon)
    # Turn off LJ in copied force.
    nonbonded_force_copy.setParticleParameters(particle_index, charge, sigma, 0.0)

# Add the new force in the system.
system.addForce(nonbonded_force_copy)

# Separate the NonbondedForce into a separate group so that we can recompute only its energy.
for force in system.getForces():
    force.setForceGroup(1)
nonbonded_force.setForceGroup(0)

# Create the Context on the GPU.
context = openmm.Context(system, openmm.VerletIntegrator(1.0*unit.femtosecond))

# Allocate output.
potential_energies = np.empty(shape=(len(positions_trajectory), len(LJ_parameter_pairs)))

for frame_idx, (positions, box_vectors) in enumerate(zip(positions_trajectory, box_vectors_trajectory)):
    # Set coordinates and box vectors.
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)

    # First compute the energy of all the forces beside the nonbonded force (i.e. group 1).
    basic_potential = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()

    # Iterate over the parameter to modify and compute the potential of each modification.
    for particle_index, (sigma_new, epsilon_new) in LJ_parameter_pairs.items():
        # Update the particle parameters
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
        # Set the new parameters
        nonbonded_force.setParticleParameters(particle_index, charge, sigma_new, epsilon_new)
        # Update the forces in the context
        nonbonded_force.updateParametersInContext(context)
        # Recompute only the nonbonded force energy (i.e. group 0).
        potential_energies[frame_idx] = basic_potential + context.getState(getEnergy=True, groups={0}).getPotentialEnergy()
