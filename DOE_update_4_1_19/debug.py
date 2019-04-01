## No default python environment

# System packages
import os, sys, timeit
from io import StringIO
import numpy as np
import math, random
# Local protocol functions
from include.replica_exchange import *
from include.build_cg_model import *
from include.get_random_coordinates import *
# OpenMM utilities
from simtk import openmm as mm
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit

# Job settings
output_directory = "output"
input_directory = "input"
if not os.path.exists(output_directory): os.makedirs(output_directory)
if not os.path.exists(input_directory): os.makedirs(input_directory)
initial_coordinates_pdb = str(input_directory+"/coordinates.pdb")
initial_coordinates_xyz = str(input_directory+"/coordinates.xyz")
replica_exchange_storage_file = str(output_directory+"/replica_exchange.nc")

box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.002 * unit.picosecond # Units = picoseconds
kB = 0.008314462 * unit.kilojoules_per_mole #Boltzmann constant (Gas constant) in kJ/(mol*K)
simulation_steps = 1000 # Number of steps used in individual Langevin dynamics simulations
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds

# Coarse-grained model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead on which the side chains are placed
polymer_length = 8 # Number of monomers in the polymer
num_particles = (backbone_length + sidechain_length) * polymer_length
model_settings = [box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions]
mass = 12.0 * unit.amu # Mass of beads
sigma = 10.0 * unit.angstrom # Lennard-Jones interaction distance
bond_length = 1.0 * unit.angstrom # bond length
epsilon = 0.2 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
q = 0.0 * unit.elementary_charge # Charge of beads
particle_properties = [mass,q,sigma,epsilon,bond_length]
add_cg_elem(particle_properties)

sigma_list = [ unit.Quantity(length*1.0,unit.angstrom) for length in range(5,15)]
potential_energies = []
nonbonded_energies = []
sig_list = []
for sigma in sigma_list:
 positions = assign_random_initial_coordinates(model_settings,particle_properties)
 system,topology = build_cg_model(model_settings,particle_properties,positions)
 system = assign_default_box_vectors(system,box_size)
 minimization_time = simulation_time_step * 1000
 integrator = LangevinIntegrator(500.0  * unit.kelvin, minimization_time, simulation_time_step) # Define Langevin integrator
 simulation = Simulation(topology, system, integrator) # Define a simulation 'context'
 simulation.context.setPositions(positions) # Assign particle positions for this context
 simulation.context.setVelocitiesToTemperature(500.0*unit.kelvin)
 nonbondedforce = get_mm_force(model_settings,particle_properties)
 #nonbondedforce.updateParametersInContext(simulation.context)
 simulation.reporters.append(PDBReporter(str(output_directory+"/minimize_coordinates_test.pdb"),1)) # Write simulation PDB coordinates  
 simulation.reporters.append(StateDataReporter(str(output_directory+"/minimize_test.dat"),1, \
    step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))
 #simulation.minimizeEnergy() # Set the simulation type to energy minimization
 #simulation.step(1000)
 positions = simulation.context.getState(getPositions=True).getPositions()
 potential_energies.append(round(simulation.context.getState(getEnergy=True).getPotentialEnergy()._value,2))
 nonbonded_energies.append("{:.2E}".format(calculate_nonbonded_energy(model_settings,particle_properties,positions)._value))
 sig_list.append(sigma._value)
print(sig_list)
print(potential_energies)
print(nonbonded_energies)
exit()
