#!/usr/bin/python

# Import external Python packages
import numpy as np
import csv
from multiprocessing import Pool
from multiprocessing import cpu_count
import matplotlib.pyplot as pyplot
from simtk.openmm.app.pdbfile import PDBFile
from simtk import unit
import mdtraj as md

# foldamers utilities
import foldamers
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.util import *
from foldamers.src.utilities.iotools import *
from cg_openmm.src.cg_mm_tools.cg_openmm import *

# Use parallel processing
processors = cpu_count() - 4
print("Found "+str(cpu_count)+" processors on the system.")
print("Using "+str(processors)+" processors for this calculation.")

# Simulation settings
total_simulations = processors
input_coordinates = "test.pdb"
output_coordinates = "simulation.pdb"
output_data = "simulation.dat"
box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.3 * unit.femtosecond # Units = picoseconds
temperature = 300.0 * unit.kelvin
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = 100.0 * unit.picosecond # Units = picoseconds
number_steps = 100000
number_bins = 20
run_simulations = False
analyze_data = True
plot_results = True
#number_steps = round(total_simulation_time.__div__(simulation_time_step))

# Model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead(s) where sidechains are attached
polymer_length = 8 # Number of monomers in the polymer
mass = 12.0 * unit.amu # Mass of beads
sigma = 2.0 * unit.angstrom # Lennard-Jones interaction distance
bond_length = 1.0 * unit.angstrom # bond length
bond_force_constant = 9e7 # Units = kJ/mol/A^2
constrain_bonds = False
epsilon = 0.1 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
charge = 0.0 * unit.elementary_charge # Charge of beads

# Setup and run OpenMM simulations
if run_simulations == True:
 print("Running "+str(total_simulations)+" simulations for: "+str(number_steps)+" steps.")
 for index in range(total_simulations):

  input_coordinates = str("init_"+str(index)+".pdb")
# Read coordinates and build a coarse-grained model
  pdb_mm_obj = PDBFile(input_coordinates)
  positions = pdb_mm_obj.getPositions()
  cgmodel = CGModel(positions=positions,polymer_length=polymer_length,backbone_length=backbone_length, sidechain_length=sidechain_length, sidechain_positions = sidechain_positions, mass = mass, sigma = sigma, epsilon = epsilon, bond_length = bond_length, bond_force_constant = bond_force_constant, charge = charge,constrain_bonds=constrain_bonds)
  topology = pdb_mm_obj.getTopology()
  cgmodel.positions == positions
#  print("Positions for simulation # "+str(index)+" are "+str(positions))
  output_coordinates = str("simulation_"+str(index)+".pdb")
  output_data = str("simulation_"+str(index)+".dat")
  simulation = build_mm_simulation(topology,cgmodel.system,cgmodel.positions,temperature=temperature,simulation_time_step=simulation_time_step,total_simulation_time=simulation_time_step*print_frequency,output_pdb=output_coordinates,output_data=output_data,print_frequency=print_frequency)
  simulation.step(number_steps)
else:
  input_coordinates = str("init_0.pdb")
# Read coordinates and build a coarse-grained model
  pdb_mm_obj = PDBFile(input_coordinates)
  positions = pdb_mm_obj.getPositions()
  cgmodel = CGModel(positions=positions,polymer_length=polymer_length,backbone_length=backbone_length, sidechain_length=sidechain_length, sidechain_positions = sidechain_positions, mass = mass, sigma = sigma, epsilon = epsilon, bond_length = bond_length, bond_force_constant = bond_force_constant, charge = charge,constrain_bonds=constrain_bonds)

if analyze_data == True:
# Get the output energies
# Make arrays for output data
 trajectory_list = []
 energies = []
 time_step_list = []
 ete_distance_list = []

 for index in range(total_simulations):
  all_energies = []
  output_data = str("simulation_"+str(index)+".dat")
  output_coordinates = str("simulation_"+str(index)+".pdb")
  with open(output_data) as csvfile:
   readCSV = csv.reader(csvfile,delimiter=',')
   next(readCSV)
   for row in readCSV:
    if index == 0:
     time_step_list.append(row[0])
    all_energies.append(row[3])
   if index == 0: energies.append(np.array([float(time) for time in time_step_list]))
  energies.append(np.array([float(energy) for energy in all_energies]))

# Get the end-to-end distances
  trajectory = md.load(output_coordinates)
  trajectory_list.append(trajectory)
  ete_distances = md.compute_distances(trajectory,[[0,cgmodel.num_beads-1]])
  ete_distance_list.append(np.array([float(dist) for dist in ete_distances]))

# Reformat and save our data to output files
 energies_array = np.array([energy_array for energy_array in energies])
 ete_distance_array = np.array([distances for distances in ete_distance_list])
#np.savetxt("all_energies.dat",energies_array.transpose(1,0))
#np.savetxt("end_to_end_distances.dat",ete_distance_array.transpose(1,0))
 best_distances = []
 best_energies = []

# Get the lowest energy and smallest end-to-end distance structures
 for index_1 in range(len(ete_distance_list)):
  energy_array = energies[index_1]
  distance_array = ete_distance_list[index_1]
  for index_2 in range(len(distance_array)):
   if len(best_distances) < 1000:
    best_distances.append(distance_array[index_2])
    best_energies.append(energy_array[index_2])
   else:
    best_distance_array = np.array([float(distance) for distance in best_distances])
    if float(distance_array[index_2]) < float(np.amax(best_distance_array)):
     largest_distance_index = np.argmax(best_distance_array)
     best_energy_array = np.array([float(energy) for energy in best_energies])
     if float(energy_array[index_2]) < float(best_energies[largest_distance_index]):
      best_distances[largest_distance_index] = distance_array[index_2]
      best_energies[largest_distance_index] = energy_array[index_2]

 max_energy = max(best_energies)
 min_energy = min(best_energies)
 energy_bin_size = ( max_energy - min_energy ) / number_bins
 energy_bin_list = [min_energy + 0.5 * energy_bin_size for i in range(number_bins)]
 max_distance = max(best_distances)
 min_distance = min(best_distances)
 distance_bin_size = ( max_distance - min_distance ) / number_bins
 distance_bin_list = [min_distance + 0.5 * distance_bin_size for i in range(number_bins)]
 counts_list = np.array([number_bins,number_bins])
 for index_1 in range(len(distance_bin_list)):
  for index_2 in range(len(energy_bin_list)):
   for distance_index in range(len(best_distances)):
    if best_distances[distance_index] > (distance_bin_list[index_1]-0.5*distance_bin_size) and best_distances[distance_index] <= (distance_bin_list[index_1]+0.5*distance_bin_size):
     for energy_index in range(len(best_energies)):
      if best_energies[energy_index] > (energy_bin_list[index_1]-0.5*energy_bin_size) and best_energies[distance_index] <= (energy_bin_list[index_1]+0.5*energy_bin_size):
       counts_list[index_1][index_2] = counts_list[index_1][index_2] + 1

if plot_results == True:
# Plot the energies for all trajectories
 figure = pyplot.figure(0)
 simulation_time = energies[0]
 for index in range(1,total_simulations):
  pyplot.plot(simulation_time,energies[index],label=index)
 pyplot.xlabel("Simulation Time ( fs )")
 pyplot.ylabel("Total Energy ( kJ/mole )")
 pyplot.title("Total energy for 16 simulations of a (1,1) 8-mer")
 pyplot.savefig("total_energies.png")
 pyplot.legend(loc=2)
 pyplot.close()

# Plot the end-to-end distances for all trajectories
 figure = pyplot.figure(0)
 simulation_time = energies[0]
 for index in range(1,total_simulations):
#  print(len(simulation_time))
#  print(len(ete_distance_list[index]))
  pyplot.plot(simulation_time,ete_distance_list[index],label=index)
 pyplot.xlabel("Simulation Time ( fs )")
 pyplot.ylabel("End-to-end distance ( nm )")
 pyplot.title("Total energy for 16 simulations of a (1,1) 8-mer")
 pyplot.savefig("end_to_end_distances.png")
 pyplot.legend(loc=2)
 pyplot.close()

# Plot the number of conformations with a given energy and end-to-end distance
 x=np.unique(distance_bin_list)
 y=np.unique(energy_bin_list)
 z=np.array(counts_list)
 X,Y = np.meshgrid(x,y)
 Z=z.reshape(len(y),len(x))

 figure_index = 1
 figure = pyplot.figure(figure_index)
 pyplot.ylabel('Energy ( kJ/mole )')
 pyplot.xlabel('End-to-end distance ( nm )')
 pyplot.pcolormesh(X,Y,Z)
 pyplot.savefig(str("energy_vs_end-to-end_distance.png"))
 pyplot.close()

exit()
