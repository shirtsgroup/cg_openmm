#!/usr/bin/python

from simtk import unit
# foldamers utilities
import foldamers
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.iotools import *
from cg_openmm.src.cg_mm_tools.cg_openmm import *

# OpenMM simulation settings
output_pdb = "test_simulation.pdb"
output_data = "test_simulation.dat"
box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.002 * unit.picosecond # Units = picoseconds
temperature = 300.0 * unit.kelvin
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = 100.0 * unit.picosecond # Units = picoseconds

# Coarse grained model settings
backbone_length = 3 # Number of backbone beads
sidechain_length = 2 # Number of sidechain beads
sidechain_positions = [1] # Index of backbone bead on which the side chains are placed
polymer_length = 12 # Number of monomers in the polymer
mass = 12.0 * unit.amu # Mass of beads
sigma = 8.4 * unit.angstrom # Lennard-Jones interaction distance
bond_length = 1.0 * unit.angstrom # bond length
bond_force_constant = 9.9e3 # Units = kJ/mol/A^2
constrain_bonds = False
epsilon = 0.5 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
charge = 0.0 * unit.elementary_charge # Charge of beads

# Build a coarse grained model
cgmodel = CGModel(polymer_length=polymer_length,backbone_length=backbone_length, sidechain_length=sidechain_length, sidechain_positions = sidechain_positions, mass = mass, sigma = sigma, epsilon = epsilon, bond_length = bond_length, bond_force_constant = bond_force_constant, charge = charge,constrain_bonds=constrain_bonds)

# Write the initial coordinates to a PDB file
pdb_file = "init_coord.pdb"
write_pdbfile(cgmodel,pdb_file)

# Build a topology using the PDB file as input
pdb_mm_obj = PDBFile(pdb_file)
topology = pdb_mm_obj.getTopology()

# Build an OpenMM simulation object
simulation = build_mm_simulation(topology,cgmodel.system,cgmodel.positions,temperature=temperature,simulation_time_step=simulation_time_step,total_simulation_time=simulation_time_step*print_frequency,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

# Run the simulation
total_steps = round(total_simulation_time._value/simulation_time_step._value)
simulation.step(total_steps)
exit()
