###!/usr/local/bin/env python

# =============================================================================================
# Import Python packages
# =============================================================================================

import sys
from io import StringIO
from simtk import openmm as mm
import numpy as np
from simtk import unit
from simtk.openmm.app import *
from simtk.openmm import *
import openmmtools as mmtools
from simtk.openmm.app.pdbfile import PDBFile
import mdtraj as md
from include.get_random_coordinates import *
from include.build_cg_model import *

# This is where replica exchange utilities are imported from Yank
from yank import mpi, analyze
from yank.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler
from yank.multistate import ReplicaExchangeAnalyzer
from yank.multistate.multistatereporter import _DictYamlLoader
from yank.utils import config_root_logger
# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

def get_low_energy_structure(simulation_settings,model_settings,particle_properties):
 temperatures,simulation_time_step,simulation_steps,print_frequency,total_simulation_time,exchange_attempts,replica_exchange_storage_file,input_directory,output_directory = simulation_settings
 positions = assign_random_initial_coordinates(model_settings)
 box_size = model_settings[0]
 system,topology = build_cg_model(model_settings,particle_properties,positions)
 system = assign_default_box_vectors(system,box_size)
 positions = add_position_units(positions)
 minimization_time = round(simulation_time_step * 1000)
 integrator = LangevinIntegrator(500.0  * unit.kelvin, minimization_time, simulation_time_step) # Define Langevin integrator
 simulation = Simulation(topology, system, integrator) # Define a simulation 'context'
 simulation.context.setPositions(positions) # Assign particle positions for this context
# simulation.context.setVelocitiesToTemperature(500.0*unit.kelvin)
# nonbondedforce = get_mm_force(model_settings,particle_properties)
# nonbondedforce.updateParametersInContext(simulation.context)
 simulation.reporters.append(PDBReporter(str(output_directory+"/minimize_coordinates_test.pdb"),1)) # Write simulation PDB coordinates  
 simulation.reporters.append(StateDataReporter(str(output_directory+"/minimize_test.dat"),1, \
   step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))
# simulation.minimizeEnergy() # Set the simulation type to energy minimization
# simulation.step(1000)
 positions = simulation.context.getState(getPositions=True).getPositions()
# velocities = simulation.context.getState(getVelocities=True).getVelocities()
 print("The potential energy is: "+str(simulation.context.getState(getEnergy=True).getPotentialEnergy()))
 calculate_nonbonded_energy(model_settings,particle_properties,positions)
 positions = positions * 10.0
 positions_no_units = remove_position_units(positions)
 for position in positions:
  write_positions_to_xyzfile(positions_no_units,str(output_directory+"/positions_test.xyz"),model_settings)
 del simulation
 return(positions)

def run_replica_exchange(system,simulation_settings,model_settings,particle_properties,verbose=False, verbose_simulation=False):
    box_size = model_settings[0]
    temperatures,simulation_time_step,simulation_steps,print_frequency,total_simulation_time,exchange_attempts,replica_exchange_storage_file,input_directory,output_directory = simulation_settings
    num_replicas = len(temperatures)
    sampler_states = list()
    thermodynamic_states = list()
    # Define thermodynamic states.
    for temperature in temperatures:
     thermodynamic_state = mmtools.states.ThermodynamicState(system=system, temperature=temperature)
     thermodynamic_states.append(thermodynamic_state)
     positions = get_low_energy_structure(simulation_settings,model_settings,particle_properties)
     system,topology = build_cg_model(model_settings,particle_properties,positions)
     system = assign_default_box_vectors(system,box_size)
     box_vectors = get_box_vectors(box_size)
     sampler_states.append(mmtools.states.SamplerState(positions,box_vectors=box_vectors))
    # Create and configure simulation object.
    move = mmtools.mcmc.LangevinDynamicsMove(timestep=simulation_time_step,
                                             collision_rate=20.0/unit.picosecond,
                                             n_steps=simulation_steps, reassign_velocities=True)
    simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=exchange_attempts)

    if os.path.exists(replica_exchange_storage_file): os.remove(replica_exchange_storage_file)
    reporter = MultiStateReporter(replica_exchange_storage_file, checkpoint_interval=1)
    simulation.create(thermodynamic_states, sampler_states, reporter)
    config_root_logger(verbose_simulation)
    simulation.run()
    del simulation
    reporter = MultiStateReporter(replica_exchange_storage_file, open_mode='r', checkpoint_interval=1)
#    coordinates = reporter.read_sampler_states(iteration=1)[0]
    sampler_states = reporter.read_sampler_states(iteration=exchange_attempts)
    index = 1
    for sampler_state in sampler_states:
     coordinates = sampler_state.positions
     data_file = open(str(output_directory+"/coordinates_"+str(index)+".xyz"),"w")
     data_file.write(str(coordinates))
     data_file.close()
     index = index + 1
    return

def get_replica_energies(simulation_settings):
    temperatures,simulation_time_step,simulation_steps,print_frequency,total_simulation_time,exchange_attempts,replica_exchange_storage_file,input_directory,output_directory = simulation_settings
    num_replicas = len(temperatures)
    replica_energies = np.array([[0.0 for step in range(0,simulation_steps)] for replica in range(0,num_replicas)])
    energies = np.array([[] for iteration in range(0,exchange_attempts)])
    for replica in range(0,num_replicas):
     step = 0
     for iteration in range(0,exchange_attempts):
      iteration_data = MultiStateReporter(replica_exchange_storage_file, open_mode='r').read_energies(iteration=iteration)
      iteration_data = np.array(iteration_data[0])
      for energy in iteration_data[replica]:
       replica_energies[replica][step] = energy
       step = step + 1
    for replica in range(0,num_replicas):
     data_file = open(str(output_directory+"/replica_"+str(replica)+"_energies.dat"),"w")
     for energy in replica_energies[replica]:
      data_file.write(str(energy)+"\n")
     data_file.close()
    return(replica_energies)       
