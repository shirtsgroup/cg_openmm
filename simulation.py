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
output_pdb = "test_simulation.pdb"
output_data = "test_simulation.dat"
box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.05 * unit.femtosecond # Units = picoseconds
temperature = 300.0 * unit.kelvin
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 0.5 * unit.picosecond # Units = picoseconds

# Coarse grained model settings
backbone_length = 2 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead on which the side chains are placed
polymer_length = 3 # Number of monomers in the polymer
mass = 12.0 * unit.amu # Mass of beads
sigma = 3.0 * unit.angstrom # Lennard-Jones interaction distance
bond_length = 1.0 * unit.angstrom # bond length
bond_force_constant = 5e10 # Units = kJ/mol/A^2
constrain_bonds = True
epsilon = 0.1 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
charge = 0.0 * unit.elementary_charge # Charge of beads
max_force = 1e6

cgmodel_built = False
best_positions = None
largest_force = None
max_build_attempts = 10
attempt = 0

while not cgmodel_built and attempt < max_build_attempts:
# Build a coarse grained model
 cgmodel = CGModel(polymer_length=polymer_length,backbone_length=backbone_length, sidechain_length=sidechain_length, sidechain_positions = sidechain_positions, mass = mass, sigma = sigma, epsilon = epsilon, bond_length = bond_length, bond_force_constant = bond_force_constant, charge = charge,constrain_bonds=constrain_bonds)

# Confirm the validity of our trial positions
 positions = cgmodel.positions
 nonbonded_interactions = cgmodel.nonbonded_interactions
 nonbonded_distances = [distance(positions[interaction[0]],positions[interaction[1]]) for interaction in nonbonded_interactions]

# Write a PDB file so that we can build a topology
 pdb_file = "test.pdb"
 write_pdbfile_without_topology(cgmodel,pdb_file)

# Build a topology using the PDB file as input
 pdb_mm_obj = PDBFile(pdb_file)
 topology = pdb_mm_obj.getTopology()

# Build an OpenMM simulation object
 simulation = build_mm_simulation(topology,cgmodel.system,cgmodel.positions,temperature=temperature,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

# if constrain_bonds: simulation.context.applyConstraints(unit.Quantity(0.1,unit.angstrom))
# Test the construction of our simulation object

 state = simulation.context.getState(getForces=True)
 forces = state.getForces()
 force_pass_list = []
 force_list = []
 force_comparison = max_force
 for force in forces:
  for direction in force:
   force_pass_list.append(direction._value < force_comparison)
   force_list.append(direction)
   if str(direction._value) == 'nan':
    print("Error: getting 'nan' for the forces on particles.")
    print("The forces are: "+str(str([force_list[i]._value for i in range(len(force_list))])+" "+str(force_list[0].unit)))
    exit()
 if all(force_pass_list): 
  print("Successfully built a coarse-grained model")
  print("with a maximum force of: "+str(max(force_list)))
  cgmodel_built = True
 else:
  if largest_force == None:
   largest_force = max(force_list)
  else:
   if max(force_list) < largest_force:
    largest_force = max(force_list)
    best_positions = cgmodel.positions

 attempt = attempt + 1

if attempt == max_build_attempts:
 print("Error: Exceeded the maximum number of model building attempts.")
 print("Error: The forces are too large for the best structure to be a suitable initial configuration.")
 print("Going to attempt a simulation using the best structure.")
 print("The nonbonded interaction distances for particles in this configuration are:")
 print(str([distance(positions[interaction[0]],positions[interaction[1]])._value for interaction in cgmodel.nonbonded_interactions])+" "+str(positions.unit))
 print("The forces are: "+str(str([force_list[i]._value for i in range(len(force_list))])+" "+str(force_list[0].unit)))
 if best_positions == None: best_positions = positions
else:
 print("Successfully built a coarse grained model.")
 print("Whose forces are: "+str(str([force_list[i]._value for i in range(len(force_list))])+" "+str(force_list[0].unit)))

# Record the positions if we successfully generated a random configuration without clashes.
cgmodel.positions = best_positions
positions = cgmodel.positions
pdb_file = "init_coord.pdb"
write_cg_pdb(cgmodel,topology,positions,pdb_file)

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
