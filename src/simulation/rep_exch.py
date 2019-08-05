import os
import numpy as np
import matplotlib.pyplot as pyplot
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

def read_replica_exchange_data(system=None,topology=None,temperature_list=None,output_data="output.nc",print_frequency=None):
        """
        """
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

        #for step in range(total_steps):
          #print(step)
          #sampler_states = reporter.read_sampler_states(iteration=step)
          #for replica_index in range(len(temperature_list)):
            #for thermodynamic_state_index in range(len(temperature_list)):
             #for particle in range(system.getNumParticles()):
               #for cart in range(3):
                 #replica_positions[replica_index][thermodynamic_state_index][step][particle][cart] = sampler_states[replica_index].positions[particle][cart]
        #exchange_stages = 0
        #for step in range(total_steps):
          #sampler_states = reporter.read_sampler_states(step)
          #if sampler_states != None:
          # exchange_stages = exchange_stages + 1

        #for stage in range(exchange_stages):
          #sampler_states = reporter.read_sampler_states(iteration=stage)
          #for sampler_state in range(len(sampler_states)):
            #print(sampler_states)

            #for thermodynamic_state_index in range(len(temperature_list)):
             #for particle in range(system.getNumParticles()):
               #for cart in range(3):
                 #print(sampler_states[replica_index])
                 #print(sampler_states[replica_index].positions)
                 #replica_positions[replica_index][thermodynamic_state_index][step][particle][cart] = sampler_states[replica_index].positions[particle][cart]

        #replica_index = 1
        #for replica_index in range(len(replica_positions)):
          #replica_trajectory = replica_positions[replica_index][replica_index]
          #file = open(str("replica_"+str(replica_index+1)+".pdb"),"w")
          #for positions in replica_trajectory:
            #PDBFile.writeFile(topology,positions,file=file)
          #file.close()

        return(replica_energies,replica_positions,replica_state_indices)

def run_replica_exchange(topology,system,positions,temperature_list=[(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-50,50,10)],simulation_time_step=None,total_simulation_time=1.0 * unit.picosecond,output_data='output.nc',print_frequency=100,verbose=False, verbose_simulation=False,exchange_attempts=None,test_time_step=False):
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

        replica_positions: np.array( ( float * simtk.unit.positions for num_beads ) for simulation_steps )
                           List of positions for all output frames for all replicas

        replica_state_indices: np.array( ( float for num_replicas ) for exchange_attempts )
                               List of thermodynamic state assignment labels for each replica
                               during each stage of the replica exchange simulation run.

        temperature_list: List( float * simtk.unit.kelvin  for num_replicas )
                          List of the temperatures for each replica.

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
        move = mmtools.mcmc.LangevinDynamicsMove(timestep=simulation_time_step,collision_rate=5.0/unit.picosecond,n_steps=exchange_attempts, reassign_velocities=True)
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

        replica_energies,replica_positions,replica_state_indices = read_replica_exchange_data(system=system,topology=topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

        return(replica_energies,replica_positions,replica_state_indices)

def get_minimum_energy_pose(topology,replica_energies,replica_positions):
        """
        """
        # Get the minimum energy structure sampled during the simulation
        minimum_energy = 0.0
        for replica in range(len(replica_energies)):
          energies = np.array([energy for energy in replica_energies[replica][replica]])
          for energy in range(len(energies)):
            if energies[energy] < minimum_energy:
              minimum_energy = energies[energy]
              minimum_energy_structure = replica_positions[replica][replica][energy]
        file = open(str("re_min.pdb"),"w")
        PDBFile.writeFile(topology,minimum_energy_structure,file=file)

        return(minimum_energy_structure)

def plot_replica_exchange_energies(replica_energies,temperature_list,simulation_time_step,steps_per_stage=1,file_name="replica_exchange_energies.png"):
        """
        """

        figure = pyplot.figure(0)

        for replica in range(len(replica_energies)):
          simulation_times = np.array([float(int(step*steps_per_stage)*simulation_time_step.in_units_of(unit.picosecond)._value) for step in range(len(replica_energies[replica][replica]))])
          energies = np.array([float(energy) for energy in replica_energies[replica][replica]])
          pyplot.plot(simulation_times,energies,figure=figure)

        pyplot.xlabel("Simulation Time ( Picoseconds )")
        pyplot.ylabel("Potential Energy ( kJ / mol )")
        pyplot.title("Replica Exchange Simulation")
        pyplot.legend([temperature._value for temperature in temperature_list])
        pyplot.savefig(file_name)
        #pyplot.show()
        pyplot.close()

        return

def plot_replica_exchange_summary(replica_states,temperature_list,simulation_time_step,steps_per_stage=1,file_name="replica_exchange_state_transitions.png"):
        """
        """

        figure = pyplot.figure(1)
        for replica in range(len(replica_states)):
          simulation_times = np.array([float(int(step*steps_per_stage)*simulation_time_step.in_units_of(unit.picosecond)._value) for step in range(len(replica_states[replica]))])
          state_indices = np.array([int(round(state)) for state in replica_states[replica]])
          pyplot.plot(simulation_times,state_indices,figure=figure)

        pyplot.xlabel("Simulation Time ( Picoseconds )")
        pyplot.ylabel("Thermodynamic State Index")
        pyplot.title("State Exchange Summary")
        pyplot.legend([i for i in range(len(replica_states))])
        pyplot.savefig(file_name)
        #pyplot.show()
        pyplot.close()

        return
