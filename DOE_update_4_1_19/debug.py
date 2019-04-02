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
num_particles = 100
mass = 12.0 * unit.amu # Mass of beads
sigma = 8.4 * unit.angstrom # Lennard-Jones interaction distance
epsilon = 0.5 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
q = 0.0 * unit.elementary_charge # Charge of particles
particle_properties = [mass,q,sigma,epsilon]

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

def get_mm_force(box_size,particle_properties):
 mass,q,sigma,epsilon = particle_properties
 force = mm.NonbondedForce()
 force.setCutoffDistance(1*unit.nanometer)
 for particle in range(num_particles):
  force.addParticle(q, sigma, epsilon)
 return(force)

def build_system(box_size,particle_properties,num_particles):
 mass,q,sigma,epsilon = particle_properties[:]
 system = mm.System()
 for particle in range(num_particles):
   system.addParticle(mass)
 return(system)

def build_topology(particle_properties,num_particles):
 mass,q,sigma,epsilon = particle_properties[:]
 topology = Topology()
 chain = topology.addChain()
 for particle in range(round(num_particles/2)):
            residue = topology.addResidue(str(particle), chain)
            topology.addAtom('X', 'test_1', residue)
            residue = topology.addResidue(str(particle), chain)
            topology.addAtom('Q', 'test_2', residue)
 return(topology) 

def build_model(box_size,particle_properties,positions):
 mass,q,sigma,epsilon = particle_properties[:]
# Record the positions
 pdb_file = 'temp.pdb'
 write_positions_to_pdbfile(positions,pdb_file)
# Build a topology
 topology = build_topology(particle_properties,len(positions))
#
 nonbonded = get_mm_force(box_size,particle_properties)
 system = build_system(box_size,particle_properties,len(positions))
 system.addForce(nonbonded)
# system = assign_default_box_vectors(system,box_size)
 return(system,topology)

if __name__ == '__main__':


 # Add new particle types
 elem.Element(117,'test_1','X',mass)
 elem.Element(118,'test_2','Q',mass)

# Get random positions for particles
 input_array = ["Model settings",particle_properties,num_particles]
 positions = assign_random_initial_coordinates(input_array)
 system,topology = build_model(box_size,particle_properties,positions)
 system = assign_default_box_vectors(system,box_size)
 minimization_time = simulation_time_step * 1000
 integrator = LangevinIntegrator(500.0  * unit.kelvin, minimization_time, simulation_time_step) # Define Langevin integrator
# exit()
 simulation = Simulation(topology, system, integrator) # Define a simulation 'context'
# nonbondedforce = get_mm_force(box_size,particle_properties)
# nonbondedforce.updateParametersInContext(simulation.context)
 simulation.context.setPositions(positions) # Assign particle positions for this context
 simulation.context.setVelocitiesToTemperature(500.0*unit.kelvin)
# simulation.reporters.append(PDBReporter(str(output_directory+"/minimize_coordinates_test.pdb"),1)) # Write simulation PDB coordinates  
 simulation.reporters.append(StateDataReporter(str(output_directory+"/minimize_test.dat"),1, \
    step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))
 simulation.minimizeEnergy() # Set the simulation type to energy minimization
 simulation.step(1000)
 positions = simulation.context.getState(getPositions=True).getPositions()
 potential_energy = round(simulation.context.getState(getEnergy=True).getPotentialEnergy()._value,2)
 print(potential_energy)
 exit()
