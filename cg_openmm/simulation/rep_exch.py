import os
import subprocess
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_pdf import PdfPages
from simtk import unit
import openmmtools
from cg_openmm.utilities.util import set_box_vectors, get_box_vectors
from simtk.openmm.app.pdbfile import PDBFile
from simtk.openmm.app.dcdfile import DCDFile
from mdtraj.formats import PDBTrajectoryFile
from mdtraj import Topology
from pymbar import timeseries
import time

from openmmtools.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler
from openmmtools.multistate import ReplicaExchangeAnalyzer

# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

kB = (unit.MOLAR_GAS_CONSTANT_R).in_units_of(unit.kilojoule / (unit.kelvin * unit.mole))

def make_replica_dcd_files(topology, replica_positions, timestep, time_interval, output_dir="output", frame_begin=0, stride=1):
    """
    Make dcd files from replica exchange simulation trajectory data
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

    :param replica_positions: Positions array for the replica exchange data for which we will write dcd files
    :type replica_positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [n_replicas,cgmodel.num_beads,3] ), simtk.unit )

    :param timestep: Time step used in the simulation
    :type timestep: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>` float * simtk.unit
    
    :param time_interval: frequency, in number of time steps, at which positions were recorded
    :type time_interval: int
    
    :param output_directory: Path to which we will write the output (default="output")
    :type output_directory: str
    
    :param frame_begin: Frame at which to start writing the dcd trajectory (default=0)
    :type frame_begin: int
    
    :param stride: advance by this many time intervals when writing dcd trajectories (default=1)
    :type stride: int 
    
    """
    
    file_list = []
    for replica_index in range(len(replica_positions)):
        replica_trajectory = replica_positions[replica_index][frame_begin::stride]
        file_name = os.path.join(output_dir, "replica_" + str(replica_index + 1) + ".dcd")
        file = open(file_name, "wb")
        dcd_file = DCDFile(file, topology, timestep, firstStep=frame_begin,interval=time_interval)
        for positions in replica_trajectory:
            DCDFile.writeModel(dcd_file, positions)
        file.close()
        file_list.append(file_name)
    return file_list
    

def make_replica_pdb_files(topology, replica_positions, output_dir="output", frame_begin=0, stride=1):
    """
    Make PDB files from replica exchange simulation trajectory data
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
    
    :param replica_positions: Positions array for the replica exchange data for which we will write PDB files
    :type replica_positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [n_replicas,cgmodel.num_beads,3] ), simtk.unit )

    :param output_directory: Path to which we will write the output (default="output")
    :type output_directory: str    
    
    :param frame_begin: Frame at which to start writing the pdb trajectory (default=0)
    :type frame_begin: int    
    
    :param stride: advance by this many frames when writing pdb trajectories (default=1)
    :type stride: int
    
    :returns:
        - file_list ( List( str ) ) - A list of names for the files that were written

    :Example:

    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> from cg_openmm.simulation.rep_exch import *
    >>> cgmodel = CGModel()
    >>> replica_energies,replica_positions,replica_state_indices = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions)
    >>> pdb_file_list = make_replica_pdb_files(cgmodel.topology,replica_positions)

    """
    file_list = []
    for replica_index in range(len(replica_positions)):
        replica_trajectory = replica_positions[replica_index][frame_begin::stride]
        file_name = os.path.join(output_dir, "replica_" + str(replica_index + 1) + ".pdb")
        file = open(file_name, "w")
        PDBFile.writeHeader(topology, file=file)
        modelIndex = 1
        for positions in replica_trajectory:
            PDBFile.writeModel(topology, positions, file=file, modelIndex=modelIndex)
        PDBFile.writeFooter(topology, file=file)
        file.close()
        file_list.append(file_name)
    return file_list
    

def make_state_dcd_files(topology, replica_positions, replica_state_indices, timestep, time_interval, output_dir="output", frame_begin=0, stride=1, center=True):
    """
    Make dcd files from replica exchange simulation trajectory data
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

    :param replica_positions: Positions array for the replica exchange data for which we will write dcd files
    :type replica_positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [n_replicas,cgmodel.num_beads,3] ), simtk.unit )

    :param replica_state_indices: The thermodynamic state assignments for all replicas at all (printed) time steps
    :type replica_state_indices: ( np.int64( [number_replicas,number_simulation_steps] ), simtk.unit )     
    
    :param timestep: Time step used in the simulation
    :type timestep: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>` float * simtk.unit
    
    :param time_interval: frequency, in number of time steps, at which positions were recorded
    :type time_interval: int
    
    :param output_directory: Path to which we will write the output (default="output")
    :type output_directory: str
    
    :param frame_begin: Frame at which to start writing the dcd trajectory (default=0)
    :type frame_begin: int
    
    :param stride: advance by this many time intervals when writing dcd trajectories (default=1)
    :type stride: int 
    
    :param center: align the center of mass of each structure in the discontinuous state trajectory (default=True)
    :type center: Boolean
    
    """
    
    file_list = []
    state_positions = np.zeros_like(replica_positions)
    
    for replica_index in range(len(replica_positions)):
        for frame in range(len(replica_state_indices[0,:])):
            # For each frame, assign replica_positions to state positions
            state_positions[replica_state_indices[replica_index,frame],frame,:,:]=replica_positions[replica_index][frame]
        
    for state_index in range(len(replica_positions)):
        state_trajectory = state_positions[state_index][frame_begin::stride]
    
        file_name = os.path.join(output_dir, "state_" + str(state_index + 1) + ".dcd")
        file = open(file_name, "wb")
        dcd_file = DCDFile(file, topology, timestep, firstStep=frame_begin,interval=time_interval)
        
        #***Note: if we have different masses for particle types, need to update this
        if center==True:
            center_x = np.mean(state_trajectory[0,:,0])
            center_y = np.mean(state_trajectory[0,:,1])
            center_z = np.mean(state_trajectory[0,:,2])
        
        for positions in state_trajectory[::stride]:
            if center==True:
                positions[:,0] += (center_x - np.mean(positions[:,0]))
                positions[:,1] += (center_y - np.mean(positions[:,1]))
                positions[:,2] += (center_z - np.mean(positions[:,2]))
                
            # Add the units consistent with replica_energies
            positions *= replica_positions.unit
            DCDFile.writeModel(dcd_file, positions)
        file.close()
        file_list.append(file_name)
    return file_list
    
    
def make_state_pdb_files(topology, replica_positions, replica_state_indices, output_dir="output", frame_begin=0, stride=1, center=True):
    """
    Make PDB files by state from replica exchange simulation trajectory data.
    Note: these are discontinuous trajectories with constant temperature state.
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
    
    :param replica_positions: Positions array for the replica exchange data for which we will write PDB files
    :type replica_positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [n_replicas,cgmodel.num_beads,3] ), simtk.unit )

    :param replica_state_indices: The thermodynamic state assignments for all replicas at all (printed) time steps
    :type replica_state_indices: ( np.int64( [number_replicas,number_simulation_steps] ), simtk.unit ) 
    
    :param output_directory: Path to which we will write the output (default="output")
    :type output_directory: str    
    
    :param frame_begin: Frame at which to start writing the pdb trajectory (default=0)
    :type frame_begin: int    
    
    :param stride: advance by this many frames when writing pdb trajectories (default=1)
    :type stride: int   

    :param center: align the center of mass of each structure in the discontinuous state trajectory (default=True)
    :type center: Boolean
    
    :returns:
        - file_list ( List( str ) ) - A list of names for the files that were written
    """
    file_list = []
    state_positions = np.zeros_like(replica_positions)
    
    for replica_index in range(len(replica_positions)):
        for frame in range(len(replica_state_indices[0,:])):
            # For each frame, assign replica_positions to state positions
            state_positions[replica_state_indices[replica_index,frame],frame,:,:]=replica_positions[replica_index][frame]
        
    for state_index in range(len(replica_positions)):
        state_trajectory = state_positions[state_index]
    
        file_name = os.path.join(output_dir, "state_" + str(state_index + 1) + ".pdb")
        file = open(file_name, "w")

        PDBFile.writeHeader(topology, file=file)
        modelIndex = 1
        
        #***Note: if we have different masses for particle types, need to update this
        if center==True:
            center_x = np.mean(state_trajectory[0,:,0])
            center_y = np.mean(state_trajectory[0,:,1])
            center_z = np.mean(state_trajectory[0,:,2])
        
        for positions in state_trajectory[::stride]:
            if center==True:
                positions[:,0] += (center_x - np.mean(positions[:,0]))
                positions[:,1] += (center_y - np.mean(positions[:,1]))
                positions[:,2] += (center_z - np.mean(positions[:,2]))
                
            # Add the units consistent with replica_energies, such that PDBFile will write in angstroms.
            positions *= replica_positions.unit
            PDBFile.writeModel(topology, positions, file=file, modelIndex=modelIndex)
        PDBFile.writeFooter(topology, file=file)
        file.close()
        file_list.append(file_name)
    return file_list


def process_replica_exchange_data(
    output_data="output.nc", output_directory="output", series_per_page=4, write_data_file=True, detect_equilibration=True, plot_production_only=False
):
    """
    Read replica exchange simulation data.
    
    :param system: OpenMM system object, default = None
    :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_
    
    :param temperature_list: List of temperatures that will be used to define different replicas (thermodynamics states), default = None
    :type temperature_list: List( `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_ * number_replicas )

    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data, default = None
    :type output_data: str

    :param series_per_page: number of data series to plot per pdf page (default=6)
    :type series_per_page: int
    
    :param write_data_file: Option to write a text data file containing the state_energies array (default=True)
    :type write_data_file: Boolean
    
    :param detect_equilibration: Option to determine the frame at which the production region begins (default=True)
    :type detect_equilibration: Boolean
    
    :param plot_production_only: Option to plot only the production region, as determined from pymbar detectEquilibration (default=False)
    :type plot_production_only: Boolean    

    :returns:
        - replica_energies ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.float( [number_replicas,number_simulation_steps] ), simtk.unit ) ) - The potential energies for all replicas at all (printed) time steps
        - replica_positions ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.float( [number_replicas,number_simulation_steps,cgmodel.num_beads,3] ), simtk.unit ) ) - The positions for all replicas at all (printed) time steps
        - replica_state_indices ( np.int64( [number_replicas,number_simulation_steps] ), simtk.unit ) - The thermodynamic state assignments for all replicas at all (printed) time steps
        - production_start ( int - The frame at which the production region begins for all replicas, as determined from pymbar detectEquilibration
        - max_sample_spacing ( int - The number of frames between uncorrelated state energies )
    :Example:

    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> from cg_openmm.simulation.rep_exch import *
    >>> cgmodel = CGModel()
    >>> replica_energies,replica_positions,replica_state_indices = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions)
    >>> replica_energies,replica_positions,replica_state_indices = process_replica_exchange_data(temperature_list=,output_data="output.nc")


    """
    # Read the simulation coordinates for individual temperature replicas
    reporter = MultiStateReporter(output_data, open_mode="r")

    # figure out what the time between output is.
    # We assume all use the same time step (which i think is required)
    mcmove = reporter.read_mcmc_moves()[0]
    time_interval = mcmove.n_steps*mcmove.timestep

    # figure out what the temperature list is
    states = reporter.read_thermodynamic_states()[0]
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)

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
    n_replicas = len(temperature_list)
    for k in range(n_replicas):
        replica_energies[:, k, :] *= beta_k[k] ** (-1)

    total_steps = len(replica_energies[0][0])
    state_energies = np.zeros([n_replicas, total_steps])

    # there must be some better way to do this as list comprehension.
    for step in range(total_steps):
        for state in range(n_replicas):
            state_energies[state, step] = replica_energies[
                np.where(replica_state_indices[:, step] == state)[0], 0, step
            ]

    # can run physical-valication on these state_energies
        
    # Use pymbar timeseries module to detect production period
    # We can also add in the subsampleCorrelatedData routine
    production_start = None
    max_sample_spacing = 1
    
    if detect_equilibration==True:
        t0 = np.zeros((n_replicas))
        subsample_indices = {}
        for state in range(n_replicas):
            t0[state], g, Neff_max = timeseries.detectEquilibration(state_energies[state])
        production_start = int(np.max(t0))
        
        # Choose the most conservative sample spacing
        for state in range(n_replicas):
            subsample_indices[state] = timeseries.subsampleCorrelatedData(
                state_energies[state][production_start:],
                conservative=True,
            )
            if (subsample_indices[state][1]-subsample_indices[state][0]) > max_sample_spacing:
                max_sample_spacing = (subsample_indices[state][1]-subsample_indices[state][0])
                
                
    print("state    mean energies  variance")
    for state in range(n_replicas):
        state_mean = np.mean(state_energies[state,production_start::max_sample_spacing])
        state_std = np.std(state_energies[state,production_start::max_sample_spacing])
        print(
            f"  {state:4d}    {state_mean:10.6f} {state_std:10.6f}"
        )


    replica_positions = np.zeros([n_replicas, total_steps, n_particles, 3])

    if write_data_file == True:
        f = open(os.path.join(output_directory, "replica_energies.dat"), "w")
        for step in range(total_steps):
            f.write(f"{step:10d}")
            sampler_states = reporter.read_sampler_states(iteration=step)
            for replica_index in range(n_replicas):
                replica_positions[replica_index, step, :, :] = sampler_states[replica_index].positions
                f.write(f"{replica_energies[replica_index,replica_index,step]:12.6f}")
            f.write("\n")
        f.close()

    # doing the array operations gets rid of units, convert back to units
    replica_positions = replica_positions * sampler_states[0].positions[0].unit

    if plot_production_only==True:
        plot_replica_exchange_energies(
            state_energies[:,production_start:],
            temperature_list,
            series_per_page,
            time_interval=time_interval,
            time_shift=production_start*time_interval,
        )
        
        plot_replica_exchange_energy_histograms(
            state_energies[:,production_start:],
            temperature_list,
        )

        plot_replica_exchange_summary(
            replica_state_indices[:,production_start:],
            temperature_list,
            series_per_page,
            time_interval=time_interval,
            time_shift=production_start*time_interval,
        )
        
    else:
        plot_replica_exchange_energies(
            state_energies,
            temperature_list,
            series_per_page,
            time_interval=time_interval,
        )
        
        plot_replica_exchange_energy_histograms(
            state_energies,
            temperature_list,
        )

        plot_replica_exchange_summary(
            replica_state_indices,
            temperature_list,
            series_per_page,
            time_interval=time_interval,
        )

    return (replica_energies, replica_positions, replica_state_indices, production_start, max_sample_spacing)


def run_replica_exchange(
    topology,
    system,
    positions,
    total_simulation_time=1.0 * unit.picosecond,
    simulation_time_step=None,
    temperature_list=None,
    friction=1.0 / unit.picosecond,
    minimize=True,
    exchange_frequency=1000,
    output_data="output/output.nc",
):

    """
    Run a OpenMMTools replica exchange simulation using an OpenMM coarse grained model.
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

    :param system: OpenMM System()
    :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

    :param positions: Positions array for the model we would like to test
    :type positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [cgmodel.num_beads,3] ), simtk.unit )

    :param total_simulation_time: Total run time for individual simulations
    :type total_simulation_time: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param simulation_time_step: Simulation integration time step
    :type simulation_time_step: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = None
    :type temperature: List( float * simtk.unit.temperature )

    :param friction: Langevin thermostat friction coefficient, default = 1 / ps
    :type friction: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param minimize: Whether minimization is done before running the simulation
    :type minimize: bool

    :param output_data: Name of NETCDF file where we will write simulation data
    :type output_data: string

    :param exchange_frequency: Number of time steps between replica exchange attempts, Default = None
    :type exchange_frequency: int	

    :param output_data: file to put the output .nc 
    :type output_data: netCDF4 file as generated by OpenMM  

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

    simulation_steps = int(np.floor(total_simulation_time / simulation_time_step))

    exchange_attempts = int(np.floor(simulation_steps / exchange_frequency))

    if temperature_list is None:
        temperature_list = [((300.0 + i) * unit.kelvin) for i in range(-50, 50, 10)]

    num_replicas = len(temperature_list)
    sampler_states = list()
    thermodynamic_states = list()

    # Define thermodynamic states.
    # box_vectors = system.getDefaultPeriodicBoxVectors()
    for temperature in temperature_list:
        thermodynamic_state = openmmtools.states.ThermodynamicState(
            system=system, temperature=temperature
        )
        thermodynamic_states.append(thermodynamic_state)
        sampler_states.append(
            openmmtools.states.SamplerState(positions)
        )  # no box vectors, non-periodic system.

    # Create and configure simulation object.

    move = openmmtools.mcmc.LangevinDynamicsMove(
        timestep=simulation_time_step,
        collision_rate=friction,
        n_steps=exchange_frequency,
        reassign_velocities=False,
    )

    simulation = ReplicaExchangeSampler(
        mcmc_moves=move,
        number_of_iterations=exchange_attempts,
        replica_mixing_scheme='swap-neighbors',
    )

    if os.path.exists(output_data):
        os.remove(output_data)

    reporter = MultiStateReporter(output_data, checkpoint_interval=1)
    simulation.create(thermodynamic_states, sampler_states, reporter)

    if minimize:
        simulation.minimize()

    print("Running replica exchange simulations with OpenMM...")
    print(f"Using a time step of {simulation_time_step}")
    try:
        simulation.run()
    except BaseException:
        print("Replica exchange simulation failed, try verifying your model/simulation settings.")
        exit()


def get_minimum_energy_ensemble(
    topology, replica_energies, replica_positions, ensemble_size=5, file_name=None
):

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
    state_energies,
    temperature_list,
    series_per_page,
    time_interval=1.0 * unit.picosecond,
    time_shift=0.0 * unit.picosecond,    
    file_name="rep_ex_ener.pdf",
    legend=True,
):
    """
    Plot the potential energies for a batch of replica exchange trajectories

    :param state_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps
    :type state_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

    :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)]
    :type temperature: List( float * simtk.unit.temperature )

    :param time_interval: interval between energy exchanges.
    :type time_interval: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param time_shift: amount of time before production period to shift the time axis(default = 0)
    :type time_shift: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_
    
    :param file_name: The pathname of the output file for plotting results, default = "replica_exchange_energies.png"
    :type file_name: str

    :param legend: Controls whether a legend is added to the plot
    :type legend: Logical

    """

    simulation_times = np.array(
        [
            step * time_interval.value_in_unit(unit.picosecond)
            for step in range(len(state_energies[0]))
        ]
    )
    
    simulation_times += time_shift.value_in_unit(unit.picosecond)
    
    # If more than series_per_page replicas, split into separate pages for better visibility
    nmax = series_per_page
    npage = int(np.ceil(len(temperature_list)/nmax))
    
    with PdfPages(file_name) as pdf:
        page_num=1
        plotted_per_page=0
        pyplot.figure()
        for state in range(len(temperature_list)):
            if plotted_per_page <= (nmax):
                pyplot.plot(
                    simulation_times,
                    state_energies[state,:],
                    alpha=0.5,
                    linewidth=1,
                )
                plotted_per_page += 1
                
            if (plotted_per_page >= nmax) or (state==(len(temperature_list)-1)):
                # Save and close previous page
                pyplot.xlabel("Simulation Time ( Picoseconds )")
                pyplot.ylabel("Potential Energy ( kJ / mol )")
                pyplot.title("Replica Exchange Simulation")
                
                if legend:
                    pyplot.legend(
                        [round(temperature.value_in_unit(unit.kelvin), 1) for temperature in temperature_list[(0+(page_num-1)*nmax):(page_num*nmax)]],
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        title="T (K)",
                    )  
                
                pdf.savefig(bbox_inches="tight") # Save current fig to pdf page
                pyplot.close()
                plotted_per_page = 0
                page_num += 1
                
    return
    

def plot_replica_exchange_energy_histograms(
    state_energies,
    temperature_list,
    file_name="rep_ex_ener_hist.pdf",
    legend=True,
):
    """
    Plot the potential energies for a batch of replica exchange trajectories

    :param state_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps
    :type state_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

    :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)]
    :type temperature: List( float * simtk.unit.temperature )

    :param file_name: The pathname of the output file for plotting results, default = "replica_exchange_energies.png"
    :type file_name: str

    :param legend: Controls whether a legend is added to the plot
    :type legend: Logical

    """

    figure = pyplot.figure(figsize=(8.5,11))

    for state in range(len(temperature_list)):
        n_out, bin_edges_out = np.histogram(
            state_energies[state,:],bins=20,density=True,
        )
        
        bin_centers = np.zeros((len(bin_edges_out)-1,1))
        for i in range(len(bin_edges_out)-1):
            bin_centers[i] = (bin_edges_out[i]+bin_edges_out[i+1])/2
        
        pyplot.plot(bin_centers,n_out,'o-',alpha=0.5,linewidth=1,markersize=6)
            

    pyplot.xlabel("Potential Energy ( kJ / mol )")
    pyplot.ylabel("Probability")
    pyplot.title("Replica Exchange Energy Histogram")
    
    if legend:
        pyplot.legend(
            [round(temperature._value, 1) for temperature in temperature_list],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title="T (K)",
        )

    pyplot.savefig(file_name, bbox_inches="tight")
    pyplot.close()

    return
    

def plot_replica_exchange_summary(
    replica_states,
    temperature_list,
    series_per_page,
    time_interval=1.0 * unit.picosecond,
    time_shift=0.0 * unit.picosecond,
    file_name="rep_ex_states.pdf",
    legend=True,
):
    """
    Plot the thermodynamic state assignments for individual temperature replicas as a function of the simulation time, in order to obtain a visual summary of the replica exchanges from a OpenMM simulation.

    :param replica_states: List of dimension num_replicas X simulation_steps, which gives the thermodynamic state indices for all replicas at all simulation steps
    :type replica_states: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

    :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)]
    :type temperature: List( float * simtk.unit.temperature )

    :param time_interval: interval between energy exchanges.
    :type time_interval: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param time_shift: amount of time before production period to shift the time axis(default = 0)
    :type time_shift: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param file_name: The pathname of the output file for plotting results, default = "replica_exchange_state_transitions.png"
    :type file_name: str

    :param legend: Controls whether a legend is added to the plot
    :type legend: Logical

    ..warning:: If more than 10 replica exchange trajectories are provided as input data, by default, this function will only plot the first 10 thermodynamic states.  These thermodynamic states are chosen based upon their indices, not their instantaneous temperature (ensemble) assignment.

    """
    
    simulation_times = np.array(
        [
            step * time_interval.value_in_unit(unit.picosecond)
            for step in range(len(replica_states[0]))
        ]
    )
    
    simulation_times += time_shift.value_in_unit(unit.picosecond)
    
    # If more than series_per_page replicas, split into separate pages for better visibility
    nmax = series_per_page
    npage = int(np.ceil(len(temperature_list)/nmax))
        
    with PdfPages(file_name) as pdf:
        page_num=1
        plotted_per_page=0
        pyplot.figure()
        for replica in range(len(replica_states)):
            state_indices = np.array([int(round(state)) for state in replica_states[replica]])
            
            if plotted_per_page <= (nmax):
                pyplot.plot(
                    simulation_times,
                    state_indices,
                    alpha=0.5,
                    linewidth=1
                )
                plotted_per_page += 1
                
            if (plotted_per_page >= nmax) or (replica==(len(replica_states)-1)):
                # Save and close previous page
                pyplot.xlabel("Simulation Time ( Picoseconds )")
                pyplot.ylabel("Thermodynamic State Index")
                pyplot.title("State Exchange Summary")
                
                if legend:
                    pyplot.legend(
                        [i for i in range((page_num-1)*nmax,page_num*nmax)],
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        title="Replica Index",
                    )
                
                pdf.savefig(bbox_inches="tight") # Save current fig to pdf page
                pyplot.close()
                plotted_per_page = 0
                page_num += 1

    return
