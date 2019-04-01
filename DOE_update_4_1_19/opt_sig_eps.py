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
import matplotlib.pyplot as pyplot
from multiprocessing import Pool

# Job settings
processors = 6
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
sigma = 8.4 * unit.angstrom # Lennard-Jones interaction distance
bond_length = 1.0 * unit.angstrom # bond length
epsilon = 0.5 * unit.kilojoules_per_mole # Lennard-Jones interaction strength
steps = 50
num_configurations = 100
q = 0.0 * unit.elementary_charge # Charge of beads
particle_properties = [mass,q,sigma,epsilon,bond_length]
add_cg_elem(particle_properties)

sigma_list = [unit.Quantity(sigma._value+length*0.1,sigma.unit) for length in range(0,steps)]
epsilon_list = [unit.Quantity(epsilon._value+index*0.1,epsilon.unit) for index in range(0,steps)]

def get_configuration_energies(configuration):
   system,topology = build_cg_model(model_settings,particle_properties,configuration)
   system = assign_default_box_vectors(system,box_size)
   minimization_time = simulation_time_step * 1000
   integrator = LangevinIntegrator(500.0  * unit.kelvin, minimization_time, simulation_time_step) # Define Langevin integrator
   simulation = Simulation(topology, system, integrator) # Define a simulation 'context'
   simulation.context.setPositions(configuration) # Assign particle positions for this context
#   nonbondedforce = get_mm_force(model_settings,particle_properties)
   simulation,success = test_simulation(simulation)
   positions = simulation.context.getState(getPositions=True).getPositions()
   potential_energy = round(simulation.context.getState(getEnergy=True).getPotentialEnergy()._value,2)
#   nonbonded_energy = float("{:.2E}".format(calculate_nonbonded_energy(model_settings,particle_properties,positions)._value))
   return(potential_energy)

if __name__ == '__main__':

 potential_energies = []
 nonbonded_energies = []
 sig_list = []
 eps_list = []
 for epsilon in epsilon_list:
  print("Epsilon = "+str(epsilon))
  for sigma in sigma_list:
   print("Sigma = "+str(sigma))
   particle_properties = [mass,q,sigma,epsilon,bond_length]
   configurations = [assign_random_initial_coordinates(model_settings,particle_properties) for i in range(0,num_configurations)]
   pool = Pool(processes=processors)
   avg_potential_energy = sum(pool.map(get_configuration_energies,configurations))/len(configurations)
   potential_energies.append(avg_potential_energy)
#   nonbonded_energies.append(sum(all_nonbonded_energies)/len(all_nonbonded_energies))
 
# Plot heatmap

 x=np.unique(sig_list)
 y=np.unique(eps_list)
 z=np.array(potential_energies)
 X,Y = np.meshgrid(x,y)

 Z=z.reshape(len(y),len(x))

 figure_index = 1
 figure = pyplot.figure(figure_index)
 pyplot.ylabel('Epsilon (kJ/mol)')
 pyplot.xlabel('Sigma (Angstroms)')
 pyplot.pcolormesh(X,Y,Z)
 pyplot.savefig(str("sigma_epsilon_heatmap.png"))
 pyplot.close()
