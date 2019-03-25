###!/usr/local/bin/env python

# This script runs replica exchange simulations with Yank,
# while leveraging multiple functions from OpenMM.

# The script has the following organization:
#
# 1) Run-time options 
# 2) PYthon imports
# 3) 

# =============================================================================================
# 1) Run-time options
# =============================================================================================

import os, socket
# File names
storage_file = 'test_storage.nc'
storage_checkpoint = 'test_storage_checkpoint.nc'
if socket.gethostname() == "Louie":
 pdb_file="/mnt/d/Foldamers/OpenMM_CG_polymers/structure_files/CG_8-mer.pdb"

# Simulation settings
simulation_time_step = 0.002 # Units = picoseconds
kB = 0.008314462  #Boltzmann constant (Gas constant) in kJ/(mol*K)
simulation_steps = 100 # Number of steps used in individual Langevin dynamics simulations
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds
exchange_attempts = 2 # The number of times that a replica exchange will be attempted within the simulation
num_replicas = 10 # Number of discrete temperatures at which we will run simulations
temp_increment = 20.0 # Increment by which we will discretize the temperature range for replicas
t_min = 200.0 # Minimum temperature for replicas

# Molecule settings
mass_CG1 = 1.0 * unit.amu
mass_CG2 = 1.0 * unit.amu
q_CG1 = 0.0 * unit.elementary_charge
q_CG2 = 0.0 * unit.elementary_charge
sigma_CG1 = 3.5 * unit.angstrom
sigma_CG2 = 4.5 * unit.angstrom
epsilon_CG1 = 0.2 * unit.kilocalorie_per_mole
epsilon_CG2 = 0.1 * unit.kilocalorie_per_mole

# Simulation box (unit cell) settings
box_size = 100.00 # box width

# =============================================================================================
# 2) Import Python packages
# =============================================================================================

import sys
import timeit
from io import StringIO
from simtk import openmm as mm
import numpy as np
import openmmtools as mmtools
from openmmtools import testsystems
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
import mdtraj as md

# This is where replica exchange utilities are imported from Yank
from yank import mpi, analyze
from yank.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler, ParallelTemperingSampler, SAMSSampler
from yank.multistate import ReplicaExchangeAnalyzer, SAMSAnalyzer
from yank.multistate.multistatereporter import _DictYamlLoader
from yank.utils import config_root_logger
# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

# Define temperatures for each replica
temperatures = [t_min + i * temp_increment for i in range(0,num_replicas)] * unit.kelvin

# Load coordinates from PDB file
pdb_object = PDBFile(file=pdb_file)
pdb_positions = pdb_object.getPositions()
positions = np.array(pdb_positions)

# Define unit cell dimensions
box_size = box_size * unit.angstroms  # box width
a = unit.Quantity(np.zeros([3]), unit.angstroms)
a[0] = box_size
b = unit.Quantity(np.zeros([3]), unit.angstroms)
b[1] = box_size
c = unit.Quantity(np.zeros([3]), unit.angstroms)
c[2] = box_size
box_vectors = [a, b, c]

# Define a system using OpenMM
system = mm.System()
force = mm.NonbondedForce()
for particle in positions:
 system.addParticle(mass_CG1)
 force.addParticle(q_CG1,sigma_CG1,epsilon_CG1)
system.addForce(force)

# ==============================================================================
# RUN REPLICA EXCHANGE
# ==============================================================================

def run_replica_exchange(verbose=False, verbose_simulation=False):

    sampler_states = list()
    thermodynamic_states = list()
    pdb_object = PDBFile(file=pdb_file)
    pdb_positions = pdb_object.getPositions()
    positions = np.array(pdb_positions)
    # Define thermodynamic states.
    for temperature in temperatures:
     system = mm.System()
     force = mm.NonbondedForce()
# Set interactions to be periodic Ewald.
#     force.setNonbondedMethod(mm.NonbondedForce.Ewald)
     for particle in positions:
      system.addParticle(mass_CG1)
      force.addParticle(q_CG1,sigma_CG1,epsilon_CG1)
#     force.setCutoffDistance(cutoff)
     system.addForce(force)
     thermodynamic_state = mmtools.states.ThermodynamicState(system=system, temperature=temperature)
     thermodynamic_states.append(thermodynamic_state)
     sampler_states.append(mmtools.states.SamplerState(positions,box_vectors=box_vectors))
    # Create and configure simulation object.
    move = mmtools.mcmc.LangevinDynamicsMove(timestep=simulation_time_step,
                                             collision_rate=20.0/unit.picosecond,
                                             n_steps=simulation_steps, reassign_velocities=True)
    simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=exchange_attempts)

    if os.path.exists(storage): os.remove(storage)
    reporter = MultiStateReporter(storage, checkpoint_interval=10)
  
    simulation.create(thermodynamic_states, sampler_states, reporter)
    config_root_logger(verbose_simulation)
    simulation.run()

    # Clean up.
    del simulation

    if verbose:
      print("PASSED.")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Configure logger.
    config_root_logger(False)

    start_time = timeit.default_timer()
    run_replica_exchange()
    stop_time = timeit.default_timer()
    print("Calculation time was: "+str(stop_time-start_time)+" seconds.")
#    for temperature_index in range(0,len(temperatures)):
#     trajectory = analyze.extract_trajectory(system,storage,nc_checkpoint_file=storage_checkpoint,replica_index=temperature_index)
    replica_energies = np.array([0.0 for step in range(0,simulation_steps)] for replica in range(0,num_replicas))
    energies = np.array([[] for iteration in range(0,exchange_attempts)])
    
    for replica in range(0,num_replicas):
     step = 0
     for iteration in range(0,exchange_attempts):
      iteration_data = MultiStateReporter(storage, open_mode='r').read_energies(iteration=iteration)
      print(iteration_data[0])
      iteration_data = np.array([iteration_data][0])
      for energy in iteration_data[0][replica]:
       replica_energies[replica][step] = energy
       step = step + 1
    for replica in range(0,num_replicas):
     data_file = open(str("replica_"+str(replica)+"_energies.dat"),"w")
     for energy in replica_energies[replica]:
      print(energy)
      data_file.write(energy)
     data_file.close()
       
       
