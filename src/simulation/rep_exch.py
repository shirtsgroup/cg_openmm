import os
import numpy as np
from simtk import unit
import openmmtools as mmtools
from cg_openmm.src.utilities.util import set_box_vectors, get_box_vectors
from cg_openmm.src.simulation.tools import get_simulation_time_step
from simtk.openmm.app.pdbfile import PDBFile
from mdtraj.formats import PDBTrajectoryFile
from mdtraj import Topology
from yank import mpi, analyze
from yank.analyze import extract_trajectory
from yank.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler
from yank.multistate import ReplicaExchangeAnalyzer
from yank.utils import config_root_logger
# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

def get_replica_energies(simulation_steps,num_replicas,replica_exchange_storage_file,exchange_attempts):
        """

        Parameters
        ----------

        :param simulation_steps: The number of steps to take during each replica exchange simulation
        :type simulation_steps: integer

        :param num_replicas: The number of temperature replicas for which we will perform molecular dynamics simulations
        :type num_replicas: integer

        :param replica_exchange_storage_file: The path to a NETCDF file containing the compressed output from all replica exchange simulation runs
        :type replica_exchange_storage_file: string

        :param exchange_attempts: The number of times that temperature replica exchanges were attempted
        :type exchange_attempts: integer

        Returns
        -------

        replica_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )
                          List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps

        """

        replica_energies = np.array([[0.0 for step in range(0,simulation_steps)] for replica in range(0,num_replicas)])
        energies = np.array([[] for iteration in range(0,exchange_attempts)])
        for replica in range(0,num_replicas):
          step = 0
          for iteration in range(0,exchange_attempts):
           iteration_data = MultiStateReporter(replica_exchange_storage_file, open_mode='r').read_energies(iteration=iteration)
           iteration_data = np.array(iteration_data[0])
           for energy in iteration_data[replica]:
            print(energy)
            exit()
            replica_energies[replica][step] = energy
            step = step + 1
        for replica in range(0,num_replicas):
          data_file = open(str("replica_"+str(replica)+".dat"),"w")
          for energy in replica_energies[replica]:
            data_file.write(str(energy)+"\n")
          data_file.close()
        return(replica_energies)

def replica_exchange(topology,system,positions,temperature_list=[(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-50,50,10)],simulation_time_step=None,total_simulation_time=1.0 * unit.picosecond,output_data='output.nc',print_frequency=100,verbose=False, verbose_simulation=False,exchange_attempts=None,test_time_step=False):
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

        :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)] 
        :type temperature: List( float * simtk.unit.temperature )

        :param simulation_time_step: Simulation integration time step, default = None
        :type simulation_time_step: float * simtk.unit

        :param total_simulation_time: Total simulation time
        :type total_simulation_time: float * simtk.unit

        :param output_data: Name of NETCDF file where we will write data from replica exchange simulations
        :type output_data: string

        Returns
        -------

        replica_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )
                          List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps

        """
        if simulation_time_step == None:
          simulation_time_step,force_threshold = get_simulation_time_step(topology,system,positions,temperature_list[-1],total_simulation_time)

        simulation_steps = int(round(total_simulation_time.__div__(simulation_time_step)))

        if exchange_attempts == None:
          if simulation_steps > 10000:
            exchange_attempts = round(simulation_steps/1000)
          else:
            exchange_attempts = 10

        num_replicas = len(temperature_list)
        sampler_states = list()
        thermodynamic_states = list()

        # Define thermodynamic states.
        box_vectors = system.getDefaultPeriodicBoxVectors()
        for temperature in temperature_list:
          thermodynamic_state = mmtools.states.ThermodynamicState(system=system, temperature=temperature)
          thermodynamic_states.append(thermodynamic_state)
          sampler_states.append(mmtools.states.SamplerState(positions,box_vectors=box_vectors))

        # Create and configure simulation object.
        move = mmtools.mcmc.LangevinDynamicsMove(timestep=simulation_time_step,collision_rate=5.0/unit.picosecond,n_steps=round(simulation_steps/exchange_attempts), reassign_velocities=True)
        simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=exchange_attempts)

        if os.path.exists(output_data): os.remove(output_data)
        reporter = MultiStateReporter(output_data, checkpoint_interval=print_frequency)
        simulation.create(thermodynamic_states, sampler_states, reporter)
        config_root_logger(verbose_simulation)
        #print("Running replica exchange simulations with Yank...")
        #print("Using a time step of "+str(simulation_time_step))
        #print("There are "+str(len(thermodynamic_states))+" replicas.")
        #print("with the following simulation temperatures:"+str([temperature._value for temperature in temperature_list]))
       
        if not test_time_step:
           num_attempts = 0
           while num_attempts < 5:
            try:
              simulation.run()
              #print("Replica exchange simulations succeeded with a time step of: "+str(simulation_time_step))
              break
            except:
              num_attempts = num_attempts + 1
           if num_attempts >= 5:
             print("Replica exchange simulation attempts failed, try verifying your model/simulation settings.")
             exit()
        else:
          simulation_time_step,force_threshold = get_simulation_time_step(topology,system,positions,temperature_list[-1],total_simulation_time)
          print("The suggested time step for a simulation with this model is: "+str(simulation_time_step))
          while simulation_time_step.__div__(2.0) > 0.001 * unit.femtosecond:
          
            try:
              print("Running replica exchange simulations with Yank...")
              print("Using a time step of "+str(simulation_time_step))
              print("Running each trial simulation for 1000 steps, with 10 exchange attempts.")
              move = mmtools.mcmc.LangevinDynamicsMove(timestep=simulation_time_step,collision_rate=20.0/unit.picosecond,n_steps=10, reassign_velocities=True)
              simulation = ReplicaExchangeSampler(replica_mixing_scheme='swap-neighbors',mcmc_moves=move,number_of_iterations=10)
              reporter = MultiStateReporter(output_data, checkpoint_interval=1)
              simulation.create(thermodynamic_states, sampler_states, reporter)

              simulation.run()
              print("Replica exchange simulations succeeded with a time step of: "+str(simulation_time_step))
              break
            except:
              del simulation
              os.remove(output_data)
              print("Simulation attempt failed with a time step of: "+str(simulation_time_step))
              if simulation_time_step.__div__(2.0) > 0.001 * unit.femtosecond:
                simulation_time_step = simulation_time_step.__div__(2.0)
              else:
                print("Error: replica exchange simulation attempt failed with a time step of: "+str(simulation_time_step))
                print("Please check the model and simulations settings, and try again.")
                exit()

        # Read the simulation coordinates for individual temperature replicas
        reporter = MultiStateReporter(output_data, open_mode='r', checkpoint_interval=print_frequency)
        analyzer = ReplicaExchangeAnalyzer(reporter)

        mixing_statistics = analyzer.show_mixing_statistics()

        try:
          Delta_f_ij, dDelta_f_ij = analyzer.get_free_energy()
           
        except:
          Delta_f_ij, dDelta_f_ij = 0.0, 0.0

        replica_energies,unsampled_state_energies,neighborhoods,replica_state_indices = analyzer.read_energies()

        total_steps = len(replica_energies[0][0])
        replica_positions = unit.Quantity(np.zeros([len(temperature_list),len(temperature_list),total_steps,system.getNumParticles(),3]),unit.nanometer)

        for step in range(total_steps):
          sampler_states = reporter.read_sampler_states(iteration=step)
          for replica_index in range(len(temperature_list)):
            for thermodynamic_state_index in range(len(thermodynamic_states)):
             for particle in range(system.getNumParticles()):
               for cart in range(3):
                 replica_positions[replica_index][thermodynamic_state_index][step][particle][cart] = sampler_states[replica_index].positions[particle][cart]

        replica_index = 1
        for replica_index in range(len(replica_positions)):
          replica_trajectory = replica_positions[replica_index][replica_index]
          file = open(str("replica_"+str(replica_index+1)+".pdb"),"w")
          for positions in replica_trajectory:
            PDBFile.writeFile(topology,positions,file=file)
          file.close()

        return(replica_energies,replica_positions,replica_state_indices,temperature_list)

