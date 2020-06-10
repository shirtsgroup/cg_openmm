import os
import subprocess
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
import openmmtools
from cg_openmm.utilities.util import set_box_vectors, get_box_vectors
from cg_openmm.simulation.tools import get_simulation_time_step
from simtk.openmm.app.pdbfile import PDBFile
from mdtraj.formats import PDBTrajectoryFile
from mdtraj import Topology

from openmmtools.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler
from openmmtools.multistate import ReplicaExchangeAnalyzer

# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

kB = (unit.MOLAR_GAS_CONSTANT_R).in_units_of(unit.kilojoule / (unit.kelvin * unit.mole))


def make_replica_pdb_files(topology, replica_positions, output_dir=""):
    """
    Make PDB files from replica exchange simulation trajectory data
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
    
    :param replica_positions: Positions array for the replica exchange data for which we will write PDB files
    :type replica_positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [n_replicas,cgmodel.num_beads,3] ), simtk.unit )

    :returns:
        - file_list ( List( str ) ) - A list of names for the files that were written

    :Example:

    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> from cg_openmm.simulation.rep_exch import *
    >>> cgmodel = CGModel()
    >>> replica_energies,replica_positions,replica_state_indices = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions)
    >>> pdb_file_list = make_replica_pdb_files(cgmodel.topology,replica_positions)

    """
    replica_index = 1
    file_list = []
    for replica_index in range(len(replica_positions)):
        replica_trajectory = replica_positions[replica_index]
        file_name = os.path.join(output_dir,"replica_"+str(replica_index + 1)+".pdb")
        file = open(file_name, "w")
        PDBFile.writeHeader(topology, file=file)
        modelIndex = 1
        for positions in replica_trajectory:
            PDBFile.writeModel(topology, positions, file=file, modelIndex=modelIndex)
        PDBFile.writeFooter(topology, file=file)
        file.close()
        file_list.append(file_name)
    return file_list


def process_replica_exchange_data(
    temperature_list=None,
    output_data="output.nc",
    output_directory="output",
    time_interval=None
    ):
    """
    Read replica exchange simulation data.
    
    :param system: OpenMM system object, default = None
    :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_
    
    :param temperature_list: List of temperatures that will be used to define different replicas (thermodynamics states), default = None
    :type temperature_list: List( `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_ * number_replicas )

    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data, default = None
    :type output_data: str

    :param time_interval: Time between samples, default = None
    :type time_interval: int


    :returns:
        - replica_energies ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.float( [number_replicas,number_simulation_steps] ), simtk.unit ) ) - The potential energies for all replicas at all (printed) time steps
        - replica_positions ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.float( [number_replicas,number_simulation_steps,cgmodel.num_beads,3] ), simtk.unit ) ) - The positions for all replicas at all (printed) time steps

        - replica_state_indices ( np.int64( [number_replicas,number_simulation_steps] ), simtk.unit ) - The thermodynamic state assignments for all replicas at all (printed) time steps

    :Example:

    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> from cg_openmm.simulation.rep_exch import *
    >>> cgmodel = CGModel()
    >>> replica_energies,replica_positions,replica_state_indices = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions)
    >>> replica_energies,replica_positions,replica_state_indices = process_replica_exchange_data(temperature_list=,output_data="output.nc",print_frequency=None)


    """
    # Read the simulation coordinates for individual temperature replicas
    reporter = MultiStateReporter(output_data, open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)
    
    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()

    n_particles = np.shape(reporter.read_sampler_states(iteration=0)[0].positions)[0]
    temps = np.array([temp._value for temp in temperature_list])
    beta_k = 1 / (kB * temps)
    for k in range(len(replica_energies)):
        replica_energies[:, k, :] *= beta_k[k] ** (-1)

    total_steps = len(replica_energies[0][0])
    replica_positions = unit.Quantity(
        np.zeros([len(temperature_list), total_steps, n_particles, 3]), unit.nanometer
    )

    for step in range(total_steps):
        sampler_states = reporter.read_sampler_states(iteration=step)
        for replica_index in range(len(temperature_list)):
            for particle in range(n_particles):
                replica_positions[replica_index][step][particle] = sampler_states[
                    replica_index
                ].positions[particle]

    plot_replica_exchange_energies(
        replica_energies,
        temperature_list,
        time_interval = time_interval,
        output_directory = output_directory,
        )

    plot_replica_exchange_summary(
        replica_state_indices,
        temperature_list,
        time_interval = time_interval,
        output_directory = output_directory,
        )

    return (replica_energies, replica_positions, replica_state_indices)


def run_replica_exchange(
    topology,
    system,
    positions,
    temperature_list=None,
    simulation_time_step=None,
    total_simulation_time=1.0 * unit.picosecond,
    output_data="output.nc",
    print_frequency=100,
    collision_rate=5,
    verbose_simulation=False,
    exchange_frequency=1000,
    test_time_step=False,
    output_directory=None,
    minimize=True
    ):

    """
    Run a OpenMMTools replica exchange simulation using an OpenMM coarse grained model.
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

    :param system: OpenMM System()
    :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

    :param positions: Positions array for the model we would like to test
    :type positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

    :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = None
    :type temperature: List( float * simtk.unit.temperature )

    :param simulation_time_step: Simulation integration time step
    :type simulation_time_step: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param total_simulation_time: Total run time for individual simulations
    :type total_simulation_time: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param output_data: Name of NETCDF file where we will write simulation data
    :type output_data: string

    :param print_frequency: Number of simulation steps to skip when writing to output, Default = 100
    :type print_frequence: int

    :param verbose_simulation: Determines how much output is printed during a simulation run.  Default = False
    :type verbose_simulation: Logical
	
    :param exchange_frequency: Number of time steps between replica exchange attempts, Default = None
    :type exchange_frequency: int	

    :param test_time_step: Logical variable determining if a test of the time step will be performed, Default = False
    :type test_time_step: Logical

    :param output_directory: Path to which we will write the output from simulation runs.
    :type output_directory: str

    :returns:
        - replica_energies ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.float( [number_replicas,number_simulation_steps] ), simtk.unit ) ) - The potential energies for all replicas at all (printed) time steps
        - replica_positions ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.float( [number_replicas,number_simulation_steps,cgmodel.num_beads,3] ), simtk.unit ) ) - The positions for all replicas at all (printed) time steps

        - replica_state_indices ( np.int64( [number_replicas,number_simulation_steps] ), simtk.unit ) - The thermodynamic state assignments for all replicas at all (printed) time steps

    :Example:

    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> from cg_openmm.simulation.rep_exch import *
    >>> cgmodel = CGModel()
    >>> replica_energies,replica_positions,replica_state_indices = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions)

    """
    
    if simulation_time_step is None:
        simulation_time_step, force_threshold = get_simulation_time_step(
            topology, system, positions, temperature_list[-1], total_simulation_time
            )
        
    simulation_steps = int(np.floor(total_simulation_time/simulation_time_step))

    exchange_attempts = int(np.floor(simulation_steps/exchange_frequency))

    if temperature_list is None:
        temperature_list = [((300.0 + i) * unit.kelvin) for i in range(-50, 50, 10)]

    num_replicas = len(temperature_list)
    sampler_states = list()
    thermodynamic_states = list()

    # Define thermodynamic states.
    box_vectors = system.getDefaultPeriodicBoxVectors()
    for temperature in temperature_list:
        thermodynamic_state = openmmtools.states.ThermodynamicState(
            system=system, temperature=temperature
            )
        thermodynamic_states.append(thermodynamic_state)
        sampler_states.append(openmmtools.states.SamplerState(positions, box_vectors=box_vectors))
    
    # Create and configure simulation object.

    move = openmmtools.mcmc.LangevinDynamicsMove(
        timestep=simulation_time_step,
        collision_rate=collision_rate / unit.picosecond,
        n_steps=exchange_frequency,
        reassign_velocities=False,
        )

    simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=exchange_attempts)

    if os.path.exists(output_data):
        os.remove(output_data)

    reporter = MultiStateReporter(output_data, checkpoint_interval=1)
    simulation.create(thermodynamic_states, sampler_states, reporter)

    if minimize:
        simulation.minimize()
    
    if not test_time_step:
        num_attempts = 0
        while num_attempts < 5:
            try:
                simulation.run()
                # print("Replica exchange simulations succeeded with a time step of: "+str(simulation_time_step))
                break
            except BaseException:
                num_attempts = num_attempts + 1
        if num_attempts >= 5:
            print(
                "Replica exchange simulation attempts failed, try verifying your model/simulation settings."
                )
            exit()
    else:
        simulation_time_step, force_threshold = get_simulation_time_step(
            topology, system, positions, temperature_list[-1], total_simulation_time
            )
        print(
            "The suggested time step for a simulation with this model is: "
            + str(simulation_time_step)
            )
        while simulation_time_step/2.0 > 0.001 * unit.femtosecond:
            try:
                print("Running replica exchange simulations with OpenMM...")
                print("Using a time step of " + str(simulation_time_step))
                print("Running each trial simulation for 1000 steps, with 10 exchange attempts.")
                move = openmmtools.mcmc.LangevinDynamicsMove(
                    timestep=simulation_time_step,
                    collision_rate=20.0 / unit.picosecond,
                    n_steps=10,
                    reassign_velocities=True,
                    )
                simulation = ReplicaExchangeSampler(
                    replica_mixing_scheme="swap-neighbors",
                    mcmc_moves=move,
                    number_of_iterations=10,
                    )
                reporter = MultiStateReporter(output_data, checkpoint_interval=1)
                simulation.create(thermodynamic_states, sampler_states, reporter)
                
                simulation.run()
                print(
                    "Replica exchange simulations succeeded with a time step of: "
                    + str(simulation_time_step)
                    )
                break
            except BaseException:
                del simulation
                os.remove(output_data)
                print(
                    "Simulation attempt failed with a time step of: " + str(simulation_time_step)
                    )
                if simulation_time_step/2.0 > 0.001 * unit.femtosecond:
                    simulation_time_step = simulation_time_step/2.0
                else:
                    print(
                        "Error: replica exchange simulation attempt failed with a time step of: "
                        + str(simulation_time_step)
                        )
                    print("Please check the model and simulations settings, and try again.")
                    exit()

def get_minimum_energy_ensemble(topology, replica_energies, 
                                replica_positions, ensemble_size=5, file_name=None):
    
    """
    Get an ensemble of low (potential) energy poses, and write the lowest energy structure to a PDB file if a file_name is provided.
    
    :param topology: OpenMM Topology()
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
    
    :param replica_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps
    :type replica_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )
    
    :param replica_positions: List of positions for all output frames for all replicas
    :type replica_positions: np.array( ( float * simtk.unit.positions for num_beads ) for simulation_steps )
    
    :param file_name: Output destination for PDB coordinates of minimum energy pose, Default = None
    
    :returns:
    - ensemble ( List() ) - A list of poses that are in the minimum energy ensemble.

    :Example:
    
    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> from cg_openmm.simulation.rep_exch import *
    >>> cgmodel = CGModel()
    >>> replica_energies,replica_positions,replica_state_indices = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions)
    >>> ensemble_size = 5
    >>> file_name = "minimum.pdb"
    >>> minimum_energy_ensemble = get_minimum_energy_ensemble(cgmodel.topology,replica_energies,replica_positions,ensemble_size=ensemble_size,file_name=file_name)
    
    """
    # Get the minimum energy structure sampled during the simulation
    ensemble = []
    ensemble_energies = []
    for replica in range(len(replica_energies)):
        energies = np.array([energy for energy in replica_energies[replica][replica]])
        for energy in range(len(energies)):
            if len(ensemble) < ensemble_size:
                ensemble.append(replica_positions[replica][energy])
                ensemble_energies.append(energies[energy])
            else:
                for comparison in range(len(ensemble_energies)):
                    if energies[energy] < ensemble_energies[comparison]:
                        ensemble_energies[comparison] = energies[energy]
                        ensemble[comparison] = replica_positions[replica][energy]

    if file_name is None:
        index = 1
        for pose in ensemble:
            file = open(str("re_min_" + str(index) + ".pdb"), "w")
            PDBFile.writeFile(topology, pose, file=file)
    else:
        file = open(file_name, "w")
        for pose in ensemble:
            PDBFile.writeFile(topology, pose, file=file)
            
    return ensemble

def plot_replica_exchange_energies(
    replica_energies,
    temperature_list,
    time_interval=1.0*unit.picosecond,
    file_name="rep_ex_ener.png",
    legend=True,
    output_directory=None,
):
    """
    Plot the potential energies for a batch of replica exchange trajectories

    :param replica_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps
    :type replica_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

    :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)]
    :type temperature: List( float * simtk.unit.temperature )

    :param time_interval: interval between energy exchanges.
    :type time_interval: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param file_name: The pathname of the output file for plotting results, default = "replica_exchange_energies.png"
    :type file_name: str

    :param output_directory: Path to which we will write the output from simulation runs, Default = None
    :type output_directory: str

    :param legend: Controls whether a legend is added to the plot
    :type legend: Logical

    ..warning:: If more than 10 replica exchange trajectories are provided as input data, by default, this function will only plot the first 10 thermodynamic states.  These thermodynamic states are chosen based upon their indices, not their instantaneous temperature (ensemble) assignment.

    """

    figure = pyplot.figure(0)

    for replica in range(len(replica_energies)):
        simulation_times = np.array(
            [
                step * time_interval.in_units_of(unit.picosecond)._value 
                for step in range(len(replica_energies[replica][replica]))
            ]
        )
        energies = np.array([float(energy) for energy in replica_energies[replica][replica]])
        pyplot.plot(simulation_times, energies, figure=figure)

    pyplot.xlabel("Simulation Time ( Picoseconds )")
    pyplot.ylabel("Potential Energy ( kJ / mol )")
    pyplot.title("Replica Exchange Simulation")
    if legend:
        if len(temperature_list) > 10:
            pyplot.legend(
                [round(temperature._value, 1) for temperature in temperature_list[0:9]],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title="T (K)",
            )
        else:
            pyplot.legend(
                [round(temperature._value, 1) for temperature in temperature_list],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title="T (K)",
            )
    if output_directory is not None:
        output_file = str(str(output_directory) + "/" + str(file_name))
        pyplot.savefig(output_file, bbox_inches="tight")
    else:
        pyplot.savefig(file_name, bbox_inches="tight")
    pyplot.close()

    return


def plot_replica_exchange_summary(
    replica_states,
    temperature_list,
    time_interval=1.0*unit.picosecond,
    file_name="rep_ex_states.png",
    legend=True,
    output_directory=None,
):
    """
    Plot the thermodynamic state assignments for individual temperature replicas as a function of the simulation time, in order to obtain a visual summary of the replica exchanges from a OpenMM simulation.

    :param replica_states: List of dimension num_replicas X simulation_steps, which gives the thermodynamic state indices for all replicas at all simulation steps
    :type replica_states: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

    :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)]
    :type temperature: List( float * simtk.unit.temperature )

    :param time_interval: interval between energy exchanges.
    :type time_interval: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param file_name: The pathname of the output file for plotting results, default = "replica_exchange_state_transitions.png"
    :type file_name: str

    :param legend: Controls whether a legend is added to the plot
    :type legend: Logical

    :param output_directory: Path to which we will write the output from simulation runs, default = None
    :type output_directory: str

    ..warning:: If more than 10 replica exchange trajectories are provided as input data, by default, this function will only plot the first 10 thermodynamic states.  These thermodynamic states are chosen based upon their indices, not their instantaneous temperature (ensemble) assignment.

    """

    figure = pyplot.figure(1)
    for replica in range(len(replica_states)):
        simulation_times = np.array(
            [
                step * time_interval.in_units_of(unit.picosecond)._value
                for step in range(len(replica_states[replica]))
            ]
        )
        state_indices = np.array([int(round(state)) for state in replica_states[replica]])
        pyplot.plot(simulation_times, state_indices, figure=figure)

    pyplot.xlabel("Simulation Time ( Picoseconds )")
    pyplot.ylabel("Thermodynamic State Index")
    pyplot.title("State Exchange Summary")
    if legend:
        if len(replica_states) > 10:
            pyplot.legend(
                [i for i in range(len(replica_states[0:9]))],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title="Replica Index",
            )
        else:
            pyplot.legend(
                [i for i in range(len(replica_states))],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title="Replica Index",
            )
    if output_directory is not None:
        output_file = str(str(output_directory) + "/" + str(file_name))
        pyplot.savefig(output_file, bbox_inches="tight")
    else:
        pyplot.savefig(file_name, bbox_inches="tight")
    # pyplot.show()
    pyplot.close()

    return
