#!/usr/bin/python

from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
# foldamers utilities
import foldamers
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.util import *
from foldamers.src.utilities.iotools import *
from cg_openmm.src.cg_mm_tools.cg_openmm import *

# OpenMM simulation settings
input_pdb = "test-1-1.pdb"
output_pdb = "test_simulation.pdb"
output_data = "test_simulation.dat"
box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.005 * unit.femtosecond # Units = picoseconds
temperature = 300.0 * unit.kelvin
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 0.5 * unit.picosecond # Units = picoseconds

# Coarse grained model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead on which the side chains are placed
polymer_length = 12 # Number of monomers in the polymer
mass = 12.0 * unit.amu # Mass of beads
sigma = 8.4 * unit.angstrom # Lennard-Jones interaction distance
bond_length = 1.0 * unit.angstrom # bond length
bond_force_constant = 9.9e5 # Units = kJ/mol/A^2
constrain_bonds = True
epsilon = 0.5 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
charge = 0.0 * unit.elementary_charge # Charge of beads
max_force = 1e6

# Read in the coordinates from our input file
pdb_mm_obj = PDBFile(input_pdb)
topology = pdb_mm_obj.getTopology()
print(topology.getNumAtoms())
positions = pdb_mm_obj.getPositions()
# Build a coarse grained model
cgmodel = CGModel(positions=positions,polymer_length=polymer_length,backbone_length=backbone_length, sidechain_length=sidechain_length, sidechain_positions = sidechain_positions, mass = mass, sigma = sigma, epsilon = epsilon, bond_length = bond_length, bond_force_constant = bond_force_constant, charge = charge,constrain_bonds=constrain_bonds)
print(cgmodel.system.getNumParticles())
#exit()

# Confirm the validity of our trial positions
nonbonded_interactions = cgmodel.nonbonded_interactions
nonbonded_distances = [distance(positions[interaction[0]],positions[interaction[1]]) for interaction in nonbonded_interactions]

# Build an OpenMM simulation object
simulation = build_mm_simulation(topology,cgmodel.system,positions,temperature=temperature,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

simulation.context.setPositions(positions=cgmodel.positions)
#simulation.context.applyConstraints(0.005)
state = simulation.context.getState(getForces=True,getPositions=True,getVelocities=True)
forces = state.getForces()
force_pass_list = []
force_list = []
force_comparison = max_force
for force in forces:
 for direction in force:
  force_pass_list.append(direction._value < force_comparison)
  force_list.append(direction._value)
  if str(direction._value) == 'nan':
    print("Error: getting 'nan' for the forces on particles.")
    print("The forces are: "+str(str([force_list[i]._value for i in range(len(force_list))])+" "+str(force_list[0].unit)))
    print("The positions are: "+str(state.getPositions()))
    print("The velocities are: "+str(state.getVelocities()))
#    exit()
if all(force_pass_list): 
   print("Successfully built a coarse-grained model")
   print("with a maximum force of: "+str(max(force_list)))
   cgmodel_built = True
else:
   print("Error: The forces are too large for this structure to be a suitable initial configuration.")
   print(force_list)
#   exit()

# Run the simulation
total_steps = round(total_simulation_time.__div__(simulation_time_step))
print(total_steps)
simulation.step(total_steps)
print("Finished simulation.")

# Confirm that the bond lengths were preserved
pdb_mm_obj = PDBFile(output_pdb)
num_frames = pdb_mm_obj.getNumFrames()
bond_list = cgmodel.get_bond_list()
for frame in range(num_frames):
 positions = pdb_mm_obj.getPositions(frame=frame)
 for bond in bond_list:
  dist = distance(positions[bond[0]-1],positions[bond[1]-1]).in_units_of(unit.angstrom)._value
  length = bond_length.in_units_of(unit.angstrom)._value
  if dist < 0.9 * length or dist > 1.1 * length:
    print("Error: the bond is breaking between particles")
    print(str(bond[0])+" and "+str(bond[1])+" in frame "+str(frame))
    print("The bond length is: "+str(dist))
    exit()
exit()
