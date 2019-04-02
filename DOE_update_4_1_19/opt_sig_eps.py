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
from multiprocessing import cpu_count

# Job settings
processors = 2
print("Found "+str(processors)+" processors on the system.")
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
step_length = 0.1
num_configurations = 1000
q = 0.0 * unit.elementary_charge # Charge of beads
particle_properties = [mass,q,sigma,epsilon,bond_length]
add_cg_elem(particle_properties)

sigma_list = [unit.Quantity(sigma._value+length*step_length,sigma.unit) for length in range(0,steps)]
epsilon_list = [unit.Quantity(epsilon._value+index*step_length,epsilon.unit) for index in range(0,steps)]

def get_configuration_energies(input_array):
   print("Don't kill my job.")
   model_settings,particle_properties,configuration = input_array[0],input_array[1],input_array[2]
   system,topology = build_cg_model(model_settings,particle_properties,configuration)
   system = assign_default_box_vectors(system,box_size)
   minimization_time = simulation_time_step * 1000
   integrator = LangevinIntegrator(500.0  * unit.kelvin, minimization_time, simulation_time_step) # Define Langevin integrator
   simulation = Simulation(topology, system, integrator) # Define a simulation 'context'
   simulation.context.setPositions(configuration) # Assign particle positions for this context
   potential_energy = round(simulation.context.getState(getEnergy=True).getPotentialEnergy()._value,2)
#   nonbonded_energy = float("{:.2E}".format(calculate_nonbonded_energy(model_settings,particle_properties,positions)._value))
   return(potential_energy)

def replace(num_list,comparison,direction):
 smaller = False
 index = -1
 if direction == "lower": 
  current = -999.0 * comparison.unit
  if any([comparison < num_list[num] for num in range(len(num_list))]):
   for num in range(len(num_list)):
    if num_list[num] > current:
     current = num_list[num]
     index = num
 if direction == "upper": 
  current = 999.0 * comparison.unit
  if any([comparison > num_list[num] for num in range(len(num_list))]):
   for num in range(len(num_list)):
    if num_list[num] < current:
     current = num_list[num]
     index = num
 return(current,index)

def get_lowest_energy_configurations(model_settings,particle_properties,configurations,number):
   low_energy_configurations = []
   energies = []
   configuration_index = 0
   for configuration in configurations:
    input_array = model_settings,particle_properties,configuration
    potential_energy = get_configuration_energies(input_array)
    if len(low_energy_configurations) > number:
     print("Error: too many configurations.")
     exit()
    if len(low_energy_configurations) < number:
     low_energy_configurations.append(configuration)
     energies.append(potential_energy)
    if len(low_energy_configurations) == number:
     energy_array = [pool.map([get_configuration_energies(model_settings,particle_properties,configuration) for configuration in low_energy_configurations])] 
     value = replace(energy_array,potential_energy,"lower")[0]
     replace_index = replace(energy_array,potential_energy,"lower")[1]
     if replace_index != -1:
      if potential_energy > energies[replace_index]:
       print("Error: replaced a lower energy configuration")
       exit()
      print("Replacing configuration "+str(replace_index)+" with configuration "+str(configuration_index))
#      print("with configuration that has energy "+str(potential_energy))
      low_energy_configurations[replace_index] = configuration
      energies[replace_index] = potential_energy
    configuration_index = configuration_index + 1
   return(low_energy_configurations)   

def get_most_folded_configurations(model_settings,particle_properties,configurations,number):
   most_folded_configurations = []
   end_to_end_distances = []
   configuration_index = 0
   for configuration in configurations:
    end_to_end_distance = distance(configuration[0],configuration[-1])
    if len(most_folded_configurations) > number:
     print("Error: too many configurations.")
     exit()
    if len(most_folded_configurations) < number:
     most_folded_configurations.append(configuration)
     end_to_end_distances.append(end_to_end_distance)
    if len(most_folded_configurations) == number:
     value = replace(end_to_end_distances,end_to_end_distance,"lower")[0]
     replace_index = replace(end_to_end_distances,end_to_end_distance,"lower")[1]
     if replace_index != -1:
      if end_to_end_distance > end_to_end_distances[replace_index]:
       print("Error: replaced a lower energy configuration")
       exit()
      print("Replacing configuration "+str(replace_index)+" with distance "+str(end_to_end_distances[replace_index])+" with configuration "+str(configuration_index)+" with distance "+str(end_to_end_distance))
#      print("with configuration that has energy "+str(potential_energy))
      most_folded_configurations[replace_index] = configuration
      end_to_end_distances[replace_index] = end_to_end_distance
    configuration_index = configuration_index + 1
   return(most_folded_configurations)

if __name__ == '__main__':

 potential_energies = []
 nonbonded_energies = []
 sig_list = []
 eps_list = []
 print("Generating initial configurations")
 configurations = []
 counter = 0
 pool = Pool(processes=processors)
 configurations = [pool.map(assign_random_initial_coordinates,[[model_settings,particle_properties] for i in range(num_configurations*100)])]
 
 print("Pruning configurations")
 configurations = get_most_folded_configurations(model_settings,particle_properties,configurations,num_configurations)
 print("Writing configurations to PDB file")
 for configuration in range(len(configurations)):
  write_positions_to_pdbfile(configurations[configuration],str(output_directory+"/opt_sig_eps_ens_"+str(configuration)+".pdb"),model_settings)
 configurations = get_lowest_energy_configurations(model_settings,particle_properties,configurations,num_configurations*0.01)
 print("Writing configurations to PDB file")
 for configuration in range(len(configurations)):
  write_positions_to_pdbfile(configurations[configuration],str(output_directory+"/ens_"+str(configuration)+".pdb"),model_settings)
 print("Calculating potential energies for surrogate model with a range of sigma and epsilon.")
 for epsilon in epsilon_list:
  for sigma in sigma_list:
   particle_properties = [mass,q,sigma,epsilon,bond_length]
   avg_potential_energy = sum([pool.map(get_configuration_energies,[[model_settings,particle_properties,configuration] for configuration in configurations])])/len(configurations)
   potential_energies.append(avg_potential_energy)
#   nonbonded_energies.append(sum(all_nonbonded_energies)/len(all_nonbonded_energies))
 print("Writing potential energies to output file")
 file_obj = open('potential_energies.dat','w')
 file_obj.write("Coordinates	Potential Energy ( kJ/mol )\n")
 for energy in potential_energies:
  file_obj.write(str(output_directory+"/ens_"+str(configuration)+".pdb "+str(round(energy,4))+"\n"))
 file_obj.close()
# Plot heatmap

 print("Plotting results")
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
