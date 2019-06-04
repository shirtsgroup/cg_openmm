#!/usr/bin/python

from simtk import unit
from foldamers.src.cg_model.cgmodel import basic_cgmodel
from cg_openmm.src.build.cg_build import build_mm_simulation
from cg_openmm.src.simulation.rep_exch import replica_exchange, get_replica_energies

# OpenMM simulation settings
simulation_time_step = 0.01 * unit.femtosecond
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.picosecond # Units = picoseconds

# Yank (replica exchange) simulation settings
temperature_list = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(0,30,10)]

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
#simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=temperature,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

# Run a replica exchange simulation with this cgmodel
replica_energies = replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency)

# Process and analyze the replica exchange simulation data
print(replica_energies)

exit()
