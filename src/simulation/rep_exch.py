import os
import numpy as np
from simtk import unit
import openmmtools as mmtools
from cg_openmm.src.utilities.util import set_box_vectors, get_box_vectors
from yank import mpi, analyze
from yank.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler
from yank.multistate import ReplicaExchangeAnalyzer
from yank.utils import config_root_logger
# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

def get_replica_energies(simulation_steps,num_replicas,replica_exchange_storage_file):
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
     data_file = open(str("replica_"+str(replica)+".dat"),"w")
     for energy in replica_energies[replica]:
      data_file.write(str(energy)+"\n")
     data_file.close()
    return(replica_energies)

def run_replica_exchange_in_temp(topology,system,positions,temperature_list=[(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)],simulation_time_step=None,total_simulation_time=1.0 * unit.picosecond,output_pdb='output.pdb',output_data='output.nc',print_frequency=100,verbose=False, verbose_simulation=False):
        """
        Construct an OpenMM simulation object for our coarse grained model.

        Parameters
        ----------

        :param topology: An OpenMM object which contains information about the bonds and constraints in a molecular model
        :type topology: OpenMM Topology() class object

        :param system: An OpenMM object which contains information about the forces and particle properties in a molecular model
        :type system: OpenMM System() class object

        :param positions: Contains the positions for all particles in a model
        :type positions: np.array( 'num_beads' x 3 , ( float * simtk.unit.distance ) )

        :param temperature_list: List of temperatures for which to per ( float * simtk.unit.temperature )
        :type temperature: 

        simulation_time_step: Simulation integration time step
        ( float * simtk.unit.time )

        total_simulation_time: Total simulation time ( float * simtk.unit.time )

        output_data: Name of output file where we will write the data from this
        simulation ( string )
        """
        box_size = 10.00 * unit.nanometer # box width
        if simulation_time_step == None:
          simulation_time_step = get_simulation_time_step(topology,system,positions,temperature_list[-1],time_step_list,total_simulation_time)

        simulation_steps = total_simulation_time.__div__(simulation_time_step)

        exchange_attempts = print_frequency * 5

        num_replicas = len(temperature_list)
        sampler_states = list()
        thermodynamic_states = list()

        # Define thermodynamic states.
        for temperature in temperature_list:
          thermodynamic_state = mmtools.states.ThermodynamicState(system=system, temperature=temperature)
          thermodynamic_states.append(thermodynamic_state)

        system = set_box_vectors(system,box_size)
        box_vectors = get_box_vectors(box_size)
        sampler_states.append(mmtools.states.SamplerState(positions,box_vectors=box_vectors))

        # Create and configure simulation object.
        move = mmtools.mcmc.LangevinDynamicsMove(timestep=simulation_time_step,collision_rate=20.0/unit.picosecond,n_steps=simulation_steps, reassign_velocities=True)
        simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=exchange_attempts)

        if os.path.exists(output_data): os.remove(output_data)
        reporter = MultiStateReporter(output_data, checkpoint_interval=1)
        simulation.create(thermodynamic_states, sampler_states, reporter)
        config_root_logger(verbose_simulation)
        simulation.run()
        del simulation

        # Write the simulation coordinates for individual temperature replicas
        reporter = MultiStateReporter(output_data, open_mode='r', checkpoint_interval=1)
        sampler_states = reporter.read_sampler_states(iteration=exchange_attempts)
        index = 1
        for sampler_state in sampler_states:
          coordinates = sampler_state.positions
          data_file = open(str("replica_"+str(index)+".xyz"),"w")
          data_file.write(str(coordinates))
          data_file.close()
          index = index + 1

        # Get the simulation data for individual temperature replicas
        replica_energies = get_replica_energies(simulation_steps,num_replicas,replica_exchange_storage_file)

        return(replica_energies)

