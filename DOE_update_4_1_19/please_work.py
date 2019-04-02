## No default python environment

# System packages
import os, sys, timeit
from io import StringIO
import numpy as np
import math, random
# Local protocol function
from include.get_random_coordinates import *
# OpenMM utilities
from simtk import openmm as mm
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
import simtk.openmm.app.element as elem

# Job settings
output_directory = "output"
input_directory = "input"
if not os.path.exists(output_directory): os.makedirs(output_directory)
if not os.path.exists(input_directory): os.makedirs(input_directory)
initial_coordinates_pdb = str(input_directory+"/coordinates.pdb")
initial_coordinates_xyz = str(input_directory+"/coordinates.xyz")

box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.002 * unit.picosecond # Units = picoseconds
simulation_steps = 1000 # Number of steps used in individual Langevin dynamics simulations
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds

# System settings
# Coarse-grained model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead on which the side chains are placed
polymer_length = 8 # Number of monomers in the polymer
num_particles = (backbone_length + sidechain_length) * polymer_length
steps = 50
step_length = 0.1
num_configurations = 10000
model_settings = [box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions]
mass = 12.0 * unit.amu # Mass of beads
sigma = 8.4 * unit.angstrom # Lennard-Jones interaction distance
epsilon = 0.5 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
bond_length = 1.0 * unit.angstrom # bond length
q = 0.0 * unit.elementary_charge # Charge of particles
particle_properties = [mass,q,sigma,epsilon,bond_length]

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

def get_box_vectors(box_size):
 units = box_size.unit
 a = unit.Quantity(np.zeros([3]), units)
 a[0] = box_size
 b = unit.Quantity(np.zeros([3]), units)
 b[1] = box_size
 c = unit.Quantity(np.zeros([3]), units)
 c[2] = box_size
 return([a,b,c])

def assign_default_box_vectors(system,box_size):
 a,b,c = get_box_vectors(box_size)
 system.setDefaultPeriodicBoxVectors(a, b, c)
 return(system)

def get_mm_force(model_settings,particle_properties):
 mass,q,sigma,epsilon,bond_length = particle_properties
 force = mm.NonbondedForce()
 force.setCutoffDistance(1*unit.nanometer)
 for particle in range(num_particles):
  force.addParticle(q, sigma, epsilon)
 if len(model_settings) == 5:
  bead_index = 0
  for monomer in range(polymer_length):
   for backbone_bead in range(backbone_length):
    if bead_index != 0:
     bead_index = bead_index + 1
     force.addException(particle1=bead_index,particle2=bead_index-sidechain_length-1,sigma=sigma,epsilon=0.0,chargeProd=0.0)
    if backbone_bead in sidechain_positions:
     for sidechain in range(sidechain_length):
      bead_index = bead_index + 1
      force.addException(particle1=bead_index,particle2=bead_index-1,sigma=sigma,epsilon=0.0,chargeProd=0.0)
 return(force)

def build_system(model_settings,particle_properties,num_particles):
 mass,q,sigma,epsilon,bond_length = particle_properties[:]
 system = mm.System()
 for particle in range(num_particles):
   system.addParticle(mass)
 return(system)

def build_topology(particle_properties,num_particles):
 mass,q,sigma,epsilon,bond_length = particle_properties[:]
 topology = Topology()
 chain = topology.addChain()
 for particle in range(round(num_particles/2)):
            residue = topology.addResidue(str("CG"+str(particle)), chain)
            topology.addAtom("X", "test_1", residue)
            topology.addAtom("Q", "test_2", residue)
 return(topology) 

def build_model(model_settings,particle_properties,positions):
 mass,q,sigma,epsilon,bond_length = particle_properties[:]
# Record the positions
 pdb_file = 'temp.pdb'
 write_positions_to_pdbfile(positions,pdb_file)
# Build a topology
 topology = build_topology(particle_properties,len(positions))
#
 nonbonded = get_mm_force(model_settings,particle_properties)
 system = build_system(model_settings,particle_properties,len(positions))
 system.addForce(nonbonded)
# system = assign_default_box_vectors(system,box_size)
 return(system,topology)

if __name__ == '__main__':


 # Add new particle types
 elem.Element(117,"test_1","X",mass)
 elem.Element(118,"test_2","Q",mass)

# Get random positions for particles
 input_array = [model_settings,particle_properties,num_particles]
 positions = assign_random_initial_coordinates(input_array)
 system,topology = build_model(model_settings,particle_properties,positions)
 system = assign_default_box_vectors(system,box_size)
 minimization_time = simulation_time_step * 1000
 integrator = LangevinIntegrator(500.0  * unit.kelvin, minimization_time, simulation_time_step) # Define Langevin integrator
# exit()
 simulation = Simulation(topology, system, integrator) # Define a simulation 'context'
# nonbondedforce = get_mm_force(box_size,particle_properties)
# nonbondedforce.updateParametersInContext(simulation.context)
 simulation.context.setPositions(positions) # Assign particle positions for this context
 simulation.context.setVelocitiesToTemperature(500.0*unit.kelvin)
 simulation.reporters.append(PDBReporter(str(output_directory+"/minimize_coordinates_test.pdb"),1)) # Write simulation PDB coordinates  
 simulation.reporters.append(StateDataReporter(str(output_directory+"/minimize_test.dat"),1, \
    step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))
 simulation.minimizeEnergy() # Set the simulation type to energy minimization
 simulation.step(1000)
 positions = simulation.context.getState(getPositions=True).getPositions()
 potential_energy = round(simulation.context.getState(getEnergy=True).getPotentialEnergy()._value,2)
 print(potential_energy)
 exit()
