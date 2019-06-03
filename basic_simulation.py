#!/usr/bin/python

from simtk import unit
from foldamers.src.cg_model.cgmodel import basic_cgmodel
from cg_openmm.src.build.cg_build import build_mm_simulation

# OpenMM simulation settings
temperature = 300.0 * unit.kelvin
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.picosecond # Units = picoseconds

# Coarse grained model settings
polymer_length=8
backbone_length=1
sidechain_length=1
sidechain_positions=[0]
mass = unit.Quantity(10.0,unit.amu)
sigma = unit.Quantity(2.4,unit.angstrom)
bond_length = unit.Quantity(1.0,unit.angstrom)
epsilon = unit.Quantity(0.5,unit.kilocalorie_per_mole)

# Build a basic 1-1 coarse grained model (homopolymer)
cgmodel = basic_cgmodel(polymer_length=polymer_length,backbone_length=backbone_length,sidechain_length=sidechain_length,sidechain_positions=sidechain_positions,mass=mass,bond_length=bond_length,sigma=sigma,epsilon=epsilon)

# Build an OpenMM simulation object
output_pdb = 'simulation.pdb'
output_data = 'simulation.dat'
simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=temperature,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

# Run simulation
simulation.step(500)

exit()
