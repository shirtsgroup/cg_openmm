import os
import subprocess
import time

import matplotlib.cm as cm
import matplotlib.pyplot as pyplot
import numpy as np
import openmmtools
from cg_openmm.utilities.util import get_box_vectors, set_box_vectors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from mdtraj import Topology, Trajectory
from mdtraj.formats import PDBTrajectoryFile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from openmm import unit
from openmm.app.dcdfile import DCDFile
from openmm.app.pdbfile import PDBFile
from openmmtools.multistate import (MultiStateReporter, MultiStateSampler,
                                    ReplicaExchangeAnalyzer,
                                    ReplicaExchangeSampler)
from pymbar import timeseries
from scipy.optimize import minimize_scalar
from scipy.special import erf

# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

kB = (unit.MOLAR_GAS_CONSTANT_R).in_units_of(unit.kilojoule / (unit.kelvin * unit.mole))

def make_replica_dcd_files(
    topology, timestep=5*unit.femtosecond, time_interval=200,
    output_dir="output", output_data="output.nc", checkpoint_data="output_checkpoint.nc",
    frame_begin=0, frame_stride=1, center=False):
    """
    Make dcd files from replica exchange simulation trajectory data.
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
    
    :param timestep: Time step used in the simulation (default=5*unit.femtosecond)
    :type timestep: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>` float * simtk.unit
    
    :param time_interval: frequency, in number of time steps, at which positions were recorded (default=200)
    :type time_interval: int
    
    :param output_dir: path to which we will write the output (default='output')
    :type output_dir: str
    
    :param output_data: name of output .nc data file (default='output.nc')
    :type output_data: str    
    
    :param checkpoint_data: name of checkpoint .nc data file (default='output_checkpoint.nc')
    :type checkpoint_data: str   
    
    :param frame_begin: Frame at which to start writing the dcd trajectory (default=0)
    :type frame_begin: int
    
    :param frame_stride: advance by this many time intervals when writing dcd trajectories (default=1)
    :type frame_stride: int
    
    :param center: align all frames in the replica trajectories (default=False)
    :type center: Boolean
    """
    
    file_list = []
    
    output_data_path = os.path.join(output_dir, output_data)
    
    # Get number of replicas:
    reporter = MultiStateReporter(output_data_path, open_mode='r', checkpoint_storage=checkpoint_data)
    states = reporter.read_thermodynamic_states()[0]
    n_replicas=len(states)
    
    sampler_states = reporter.read_sampler_states(iteration=0)
    xunit = sampler_states[0].positions[0].unit
        
    for replica_index in range(n_replicas):
        replica_positions = extract_trajectory(topology, reporter, replica_index=replica_index,
            frame_begin=frame_begin, frame_stride=frame_stride)
    
        n_frames_tot = replica_positions.shape[0]
            
        # Determine simulation time (in ps) for each frame:
        time_delta_ps = (timestep*time_interval).value_in_unit(unit.picosecond)
        traj_times = np.linspace(
            frame_begin*time_delta_ps,
            (frame_begin+frame_stride*(n_frames_tot-1))*time_delta_ps,
            num=n_frames_tot,
        )
    
        file_name = f"{output_dir}/replica_{replica_index+1}.dcd"

        # Trajectories are written in nanometers:
        replica_traj = Trajectory(
            replica_positions,
            Topology.from_openmm(topology),
            time=traj_times,
        )
        
        if center:
            ref_traj = replica_traj[0]
            replica_traj.superpose(ref_traj)
            # This rewrites to replica_traj        
            
        Trajectory.save_dcd(replica_traj,file_name)
    
    reporter.close()
    
    return file_list
    

def make_replica_pdb_files(
    topology, output_dir="output", output_data="output.nc", checkpoint_data="output_checkpoint.nc",
    frame_begin=0, frame_stride=1, center=False):
    """
    Make pdb files from replica exchange simulation trajectory data.
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
    
    :param output_dir: path to which we will write the output (default='output')
    :type output_dir: str
    
    :param output_data: name of output .nc data file (default='output.nc')
    :type output_data: str    
    
    :param checkpoint_data: name of checkpoint .nc data file (default='output_checkpoint.nc')
    :type checkpoint_data: str   
    
    :param frame_begin: Frame at which to start writing the pdb trajectory (default=0)
    :type frame_begin: int    
    
    :param frame_stride: advance by this many frames when writing pdb trajectories (default=1)
    :type frame_stride: int

    :param center: align all frames in the replica trajectories (default=False)
    :type center: Boolean    
    
    :returns:
        - file_list ( List( str ) ) - A list of names for the files that were written
    """
    file_list = []
    
    output_data_path = os.path.join(output_dir, output_data)
    
    # Get number of replicas:
    reporter = MultiStateReporter(output_data_path, open_mode='r', checkpoint_storage=checkpoint_data)
    states = reporter.read_thermodynamic_states()[0]
    n_replicas = len(states)
    
    sampler_states = reporter.read_sampler_states(iteration=0)
    xunit = sampler_states[0].positions[0].unit
    
    for replica_index in range(n_replicas):
        replica_positions = extract_trajectory(topology, reporter, replica_index=replica_index, 
            frame_begin=frame_begin, frame_stride=frame_stride)
    
        file_name = f"{output_dir}/replica_{replica_index+1}.pdb"

        # Trajectories are written in nanometers:
        replica_traj = Trajectory(
            replica_positions,
            Topology.from_openmm(topology),
        )
        
        if center:
            ref_traj = replica_traj[0]
            replica_traj.superpose(ref_traj)
            # This rewrites to replica_traj                
            
        Trajectory.save_pdb(replica_traj,file_name)
        
    reporter.close()
    
    return file_list
    

def make_state_dcd_files(
    topology, timestep=5*unit.femtosecond, time_interval=200,
    output_dir="output", output_data="output.nc", checkpoint_data="output_checkpoint.nc",
    frame_begin=0, frame_stride=1, center=True):
    """
    Make dcd files by state from replica exchange simulation trajectory data.
    Note: these are discontinuous trajectories with constant temperature state.
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
    
    :param timestep: Time step used in the simulation (default=5*unit.femtosecond)
    :type timestep: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>` float * simtk.unit
    
    :param time_interval: frequency, in number of time steps, at which positions were recorded (default=200)
    :type time_interval: int
    
    :param output_dir: path to which we will write the output (default='output')
    :type output_dir: str
    
    :param output_data: name of output .nc data file (default='output.nc')
    :type output_data: str    
    
    :param checkpoint_data: name of checkpoint .nc data file (default='output_checkpoint.nc')
    :type checkpoint_data: str   
    
    :param frame_begin: Frame at which to start writing the dcd trajectory (default=0)
    :type frame_begin: int
    
    :param frame_stride: advance by this many time intervals when writing dcd trajectories (default=1)
    :type frame_stride: int 
    
    :param center: align the center of mass of each structure in the discontinuous state trajectory (default=True)
    :type center: Boolean
    
    """
    
    file_list = []
    
    output_data_path = os.path.join(output_dir, output_data)
    
    # Get number of states:
    reporter = MultiStateReporter(output_data_path, open_mode='r', checkpoint_storage=checkpoint_data)
    states = reporter.read_thermodynamic_states()[0]
    
    sampler_states = reporter.read_sampler_states(iteration=0)
    xunit = sampler_states[0].positions[0].unit
        
    for state_index in range(len(states)):
        state_positions = extract_trajectory(topology, reporter, state_index=state_index,
            frame_begin=frame_begin, frame_stride=frame_stride)
            
        n_frames_tot = state_positions.shape[0]
            
        # Determine simulation time (in ps) for each frame:
        time_delta_ps = (timestep*time_interval).value_in_unit(unit.picosecond)
        traj_times = np.linspace(
            frame_begin*time_delta_ps,
            (frame_begin+frame_stride*(n_frames_tot-1))*time_delta_ps,
            num=n_frames_tot,
        )

        file_name = f"{output_dir}/state_{state_index+1}.dcd"

        # Trajectories are written in nanometers:
        state_traj = Trajectory(
            state_positions,
            Topology.from_openmm(topology),
            time=traj_times,
        )
        
        if center:
            ref_traj = state_traj[0]
            state_traj.superpose(ref_traj)
            # This rewrites to state_traj
            
        Trajectory.save_dcd(state_traj,file_name)
        
    reporter.close()        
        
    return file_list
    
    
def make_state_pdb_files(
    topology, output_dir="output", output_data="output.nc", checkpoint_data="output_checkpoint.nc",
    frame_begin=0, frame_stride=1, center=True):
    """
    Make pdb files by state from replica exchange simulation trajectory data.
    Note: these are discontinuous trajectories with constant temperature state.
    
    :param topology: OpenMM Topology
    :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_
    
    :param output_dir: path to which we will write the output (default='output')
    :type output_dir: str
    
    :param output_data: name of output .nc data file (default='output.nc')
    :type output_data: str    
    
    :param checkpoint_data: name of checkpoint .nc data file (default='output_checkpoint.nc')
    :type checkpoint_data: str   
    
    :param frame_begin: Frame at which to start writing the pdb trajectory (default=0)
    :type frame_begin: int    
    
    :param frame_stride: advance by this many frames when writing pdb trajectories (default=1)
    :type frame_stride: int   

    :param center: align the center of mass of each structure in the discontinuous state trajectory (default=True)
    :type center: Boolean
    
    :returns:
        - file_list ( List( str ) ) - A list of names for the files that were written
    """
    file_list = []
    
    output_data_path = os.path.join(output_dir, output_data)
    
    # Get number of states:
    reporter = MultiStateReporter(output_data_path, open_mode='r', checkpoint_storage=checkpoint_data)
    states = reporter.read_thermodynamic_states()[0]
    
    sampler_states = reporter.read_sampler_states(iteration=0)
    xunit = sampler_states[0].positions[0].unit
    
    for state_index in range(len(states)):
        state_positions = extract_trajectory(topology, reporter, state_index=state_index,
            frame_begin=frame_begin, frame_stride=frame_stride)
    
        file_name = f"{output_dir}/state_{state_index+1}.pdb"
        
        # Trajectories are written in nanometers:
        state_traj = Trajectory(
            state_positions,
            Topology.from_openmm(topology),
        )
        
        if center:
            ref_traj = state_traj[0]
            state_traj.superpose(ref_traj)
            # This rewrites to state_traj
            
        Trajectory.save_pdb(state_traj,file_name)
      
    reporter.close()
      
    return file_list
    
    
def extract_trajectory(
    topology, reporter,
    state_index=None, replica_index=None,
    frame_begin=0, frame_stride=1, frame_end=-1):
    """
    Internal function for extract trajectory (replica or state) from .nc file,
    Based on YANK extract_trajectory code.
    """

    # Get dimensions
    trajectory_storage = reporter._storage_checkpoint  
    n_iterations = reporter.read_last_iteration()
    n_frames = trajectory_storage.variables['positions'].shape[0]
    n_atoms = trajectory_storage.variables['positions'].shape[2]
    
    # Determine frames to extract.
    # Convert negative indices to last indices.
    if frame_begin < 0:
        frame_begin = n_frames + frame_begin
    if frame_end < 0:
        frame_end = n_frames + frame_end + 1
    frame_indices = range(frame_begin, frame_end, frame_stride)
    if len(frame_indices) == 0:
        raise ValueError('No frames selected')
        
    # Determine the number of frames that the trajectory will have.
    if state_index is None:
        n_trajectory_frames = len(frame_indices)        
    else:
        # With SAMS, an iteration can have 0 or more replicas in a given state.
        # Deconvolute state indices.
        state_indices = [None for _ in frame_indices]
        for i, iteration in enumerate(frame_indices):
            replica_indices = reporter._storage_analysis.variables['states'][iteration, :]
            state_indices[i] = np.where(replica_indices == state_index)[0]
        n_trajectory_frames = sum(len(x) for x in state_indices)        
        
    # Initialize positions and box vectors arrays.
    # MDTraj Cython code expects float32 positions.
    positions = np.zeros((n_trajectory_frames, n_atoms, 3), dtype=np.float32)

    # Extract state positions and box vectors.
    if state_index is not None:
        # Extract state positions
        frame_idx = 0
        for i, iteration in enumerate(frame_indices):
            for replica_index in state_indices[i]:
                positions[frame_idx, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                frame_idx += 1

    else:  # Extract replica positions
        for i, iteration in enumerate(frame_indices):
            positions[i, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)

    return positions
    
    
def process_replica_exchange_data(
    output_data="output/output.nc", output_directory="output", series_per_page=4,
    write_data_file=True, plot_production_only=False, print_timing=False,
    equil_nskip=1, frame_begin=0, frame_end=-1,
):
    """
    Read replica exchange simulation data, detect equilibrium and decorrelation time, and plot replica exchange results.
    
    :param output_data: path to output .nc file from replica exchange simulation, (default='output/output.nc')
    :type output_data: str
    
    :param output_directory: path to which output files will be written (default='output')
    :type output_directory: stry

    :param series_per_page: number of replica data series to plot per pdf page (default=4)
    :type series_per_page: int
    
    :param write_data_file: Option to write a text data file containing the state_energies array (default=True)
    :type write_data_file: Boolean
    
    :param plot_production_only: Option to plot only the production region, as determined from pymbar detectEquilibration (default=False)
    :type plot_production_only: Boolean    

    :param equil_nskip: skip this number of frames to sparsify the energy timeseries for pymbar detectEquilibration (default=1) - this is used only when frame_begin=0 and the trajectory has less than 40000 frames.
    :type equil_nskip: Boolean
    
    :param frame_begin: analyze starting from this frame, discarding all prior as equilibration period (default=0)
    :type frame_begin: int
    
    :param frame_end: analyze up to this frame only, discarding the rest (default=-1).
    :type frame_end: int

    :returns:
        - replica_energies ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.float( [number_replicas,number_simulation_steps] ), simtk.unit ) ) - The potential energies for all replicas at all (printed) time steps
        - replica_state_indices ( np.int64( [number_replicas,number_simulation_steps] ), simtk.unit ) - The thermodynamic state assignments for all replicas at all (printed) time steps
        - production_start ( int - The frame at which the production region begins for all replicas, as determined from pymbar detectEquilibration
        - sample_spacing ( int - The number of frames between uncorrelated state energies, estimated using heuristic algorithm )
        - n_transit ( np.float( [number_replicas] ) ) - Number of half-transitions between state 0 and n for each replica
        - mixing_stats ( tuple ( np.float( [number_replicas x number_replicas] ) , np.float( [ number_replicas ] ) , float( statistical inefficiency ) ) ) - transition matrix, corresponding eigenvalues, and statistical inefficiency
    """
    
    t1 = time.perf_counter()
    
    # Read the simulation coordinates for individual temperature replicas
    reporter = MultiStateReporter(output_data, open_mode="r")

    t2 = time.perf_counter()
    if print_timing:
        print(f"open data time: {t2-t1}")
    
    # figure out what the time between output is.
    # We assume all use the same time step (which i think is required)
    
    mcmove = reporter.read_mcmc_moves()[0]
    time_interval = mcmove.n_steps*mcmove.timestep

    t3 = time.perf_counter()
    if print_timing:
        print(f"read_mcmc_moves time: {t3-t2}")
    
    # figure out what the temperature list is
    states = reporter.read_thermodynamic_states()[0]
    
    t4 = time.perf_counter()
    if print_timing:
        print(f"read_thermodynamics_states time: {t4-t3}")
    
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)

    analyzer = ReplicaExchangeAnalyzer(reporter)
    
    t5 = time.perf_counter()
    
    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()
    
    # Truncate output of read_energies() to last frame of interest
    if frame_end > 0:
        # Use frames from frame_begin to frame_end
        replica_energies = replica_energies[:,:,:frame_end]
        unsampled_state_energies = unsampled_state_energies[:,:,:frame_end]
        neighborhoods = neighborhoods[:,:,:frame_end]
        replica_state_indices = replica_state_indices[:,:frame_end]
    
    t6 = time.perf_counter()
    if print_timing:
        print(f"read_energies time: {t6-t5}")

    n_particles = np.shape(reporter.read_sampler_states(iteration=0)[0].positions)[0]
    temps = np.array([temp._value for temp in temperature_list])
    beta_k = 1 / (kB * temps)
    n_replicas = len(temperature_list)
    for k in range(n_replicas):
        replica_energies[:, k, :] *= beta_k[k] ** (-1)

    t7 = time.perf_counter()
    if print_timing:
        print(f"reduce replica energies time: {t7-t6}")
        
    total_steps = len(replica_energies[0][0])
    state_energies = np.zeros([n_replicas, total_steps])

    t8 = time.perf_counter()
    # there must be some better way to do this as list comprehension.
    for step in range(total_steps):
        for state in range(n_replicas):
            state_energies[state, step] = replica_energies[
                np.where(replica_state_indices[:, step] == state)[0], 0, step
            ]
            
    t9 = time.perf_counter()
    if print_timing:
        print(f"assign state energies time: {t9-t8}")

    # can run physical-valication on these state_energies
        
    # Use pymbar timeseries module to detect production period
    
    t10 = time.perf_counter()
    
    # Start of equilibrated data:
    t0 = np.zeros((n_replicas))
    # Statistical inefficiency:
    g = np.zeros((n_replicas))
    
    subsample_indices = {}
    
    # If sufficiently large, discard the first 20000 frames as equilibration period and use 
    # subsampleCorrelatedData to get the energy decorrelation time.
    if total_steps >= 40000 or frame_begin > 0:
        if frame_begin > 0:
            # If specified, use frame_begin as the start of the production region
            production_start=frame_begin
        else:
            # Otherwise, use frame 20000
            production_start=20000
            
        for state in range(n_replicas):
            subsample_indices[state] = timeseries.subsampleCorrelatedData(
                state_energies[state][production_start:],
                conservative=True,
            )
            g[state] = subsample_indices[state][1]-subsample_indices[state][0]
    
    else:
        # For small trajectories, use detectEquilibration
        for state in range(n_replicas):
            t0[state], g[state], Neff_max = timeseries.detectEquilibration(state_energies[state], nskip=equil_nskip)  

            # Choose the latest equil timestep to apply to all states    
            production_start = int(np.max(t0))
    
    # Assume a normal distribution (very rough approximation), and use mean plus
    # the number of standard deviations which leads to (n_replica-1)/n_replica coverage
    # For 12 replicas this should be the mean + 1.7317 standard deviations
    
    # x standard deviations is the solution to (n_replica-1)/n_replica = erf(x/sqrt(2))
    # This is equivalent to a target of 23/24 CDF value 
    
    print(f"Correlation times (frames): {g.astype(int)}")
    
    def erf_fun(x):
        return np.power((erf(x/np.sqrt(2))-(n_replicas-1)/n_replicas),2)
        
    # x must be larger than zero    
    opt_g_results = minimize_scalar(
        erf_fun,
        bounds=(0,10),
        method='bounded',
        )
    
    if not opt_g_results.success:
        print("Error solving for correlation time, exiting...")
        print(f"erf opt results: {opt_g_results}")
        exit()
    
    sample_spacing = int(np.ceil(np.mean(g)+opt_g_results.x*np.std(g)))
    
    t11 = time.perf_counter()
    if print_timing:
        print(f"detect equil and subsampling time: {t11-t10}")
                
    print("state    mean energies  variance")
    for state in range(n_replicas):
        state_mean = np.mean(state_energies[state,production_start::sample_spacing])
        state_std = np.std(state_energies[state,production_start::sample_spacing])
        print(
            f"  {state:4d}    {state_mean:10.6f} {state_std:10.6f}"
        )

    t12 = time.perf_counter()
    
    if write_data_file == True:
        f = open(os.path.join(output_directory, "replica_energies.dat"), "w")
        for step in range(total_steps):
            f.write(f"{step:10d}")
            for replica_index in range(n_replicas):
                f.write(f"{replica_energies[replica_index,replica_index,step]:12.6f}")
            f.write("\n")
        f.close()

    t13 = time.perf_counter()
    if print_timing:
        print(f"Optionally write .dat file: {t13-t12}")
               
    t14 = time.perf_counter()
    
    if plot_production_only==True:
        plot_replica_exchange_energies(
            state_energies[:,production_start:],
            temperature_list,
            series_per_page,
            time_interval=time_interval,
            time_shift=production_start*time_interval,
            file_name=f"{output_directory}/rep_ex_ener.pdf",
        )
        
        plot_replica_exchange_energy_histograms(
            state_energies[:,production_start:],
            temperature_list,
            file_name=f"{output_directory}/rep_ex_ener_hist.pdf",
        )

        plot_replica_exchange_summary(
            replica_state_indices[:,production_start:],
            temperature_list,
            series_per_page,
            time_interval=time_interval,
            time_shift=production_start*time_interval,
            file_name=f"{output_directory}/rep_ex_states.pdf",
        )
        
        plot_replica_state_matrix(
            replica_state_indices[:,production_start:],
            file_name=f"{output_directory}/state_probability_matrix.pdf",
        )
        
    else:
        plot_replica_exchange_energies(
            state_energies,
            temperature_list,
            series_per_page,
            time_interval=time_interval,
            file_name=f"{output_directory}/rep_ex_ener.pdf",
        )
        
        plot_replica_exchange_energy_histograms(
            state_energies,
            temperature_list,
            file_name=f"{output_directory}/rep_ex_ener_hist.pdf",
        )

        plot_replica_exchange_summary(
            replica_state_indices,
            temperature_list,
            series_per_page,
            time_interval=time_interval,
            file_name=f"{output_directory}/rep_ex_states.pdf",
        )
        
        plot_replica_state_matrix(
            replica_state_indices,
            file_name=f"{output_directory}/state_probability_matrix.pdf",
        )
      
    t15 = time.perf_counter()
      
    if print_timing:
        print(f"plotting time: {t15-t14}")
    
    # Analyze replica exchange state transitions
    # For each replica, how many times does the thermodynamic state go between state 0 and state n
    # For consistency with the other mixing statistics, use only the production region here
    
    replica_state_indices_prod = replica_state_indices[:,production_start:]
    
    # Number of one-way transitions from states 0 to n or states n to 0 
    n_transit = np.zeros((n_replicas,1))
    
    # Replica_state_indices is [n_replicas x n_iterations]
    for rep in range(n_replicas):
        last_bound = None
        for i in range(replica_state_indices_prod.shape[1]):
            if replica_state_indices_prod[rep,i] == 0 or replica_state_indices_prod[rep,i] == (n_replicas-1):
                if last_bound is None:
                    # This is the first time state 0 or n is visited
                    pass
                else:
                    if last_bound != replica_state_indices_prod[rep,i]:
                        # This is a completed transition from 0 to n or n to 0
                        n_transit[rep] += 1
                last_bound = replica_state_indices_prod[rep,i]                
                        
    t16 = time.perf_counter()
    
    if print_timing:
        print(f"replica transition analysis: {t16-t15}")
        
    # Compute transition matrix from the analyzer
    mixing_stats = analyzer.generate_mixing_statistics(number_equilibrated=production_start)
    
    t17 = time.perf_counter()
    
    if print_timing:
        print(f"compute transition matrix: {t17-t16}")
        print(f"total time elapsed: {t17-t1}")

    # Close reporter/.nc file:
    reporter.close()

    return (replica_energies, replica_state_indices, production_start, sample_spacing, n_transit, mixing_stats)


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

    print("Running OpenMM replica exchange simulation...")
    print(f"Time step: {simulation_time_step}")
    print(f"Iterations: {exchange_attempts}",flush=True)
    try:
        simulation.run()
    except BaseException:
        print("Replica exchange simulation failed, try verifying your model/simulation settings.")
        exit()
        
    return
        
        
def restart_replica_exchange(
    total_simulation_time,
    simulation_time_step=5*unit.picosecond,
    exchange_frequency=200,
    output_data="output/output.nc",
    ):

    """
    Restart an OpenMMTools replica exchange simulation using an OpenMM coarse grained model and
    output .nc files from the previous segment of the simulation. 

    :param total_simulation_time: Total run time for original + new simulation segments
    :type total_simulation_time: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param simulation_time_step: Simulation integration time step (default=5*unit.picosecond)
    :type simulation_time_step: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param exchange_frequency: Number of time steps between replica exchange attempts (default=200)
    :type exchange_frequency: int

    :param output_data: Path to the NETCDF file for previous segment of simulation - this will be appended to (default="output/output.nc")
    :type output_data: str
    """

    # Compute number of total time steps and iterations:
    simulation_steps = int(np.floor(total_simulation_time / simulation_time_step))
    exchange_attempts = int(np.floor(simulation_steps / exchange_frequency))

    # Load in the reporter from the original simulation:
    reporter = MultiStateReporter(output_data, open_mode="r+")
    simulation = ReplicaExchangeSampler.from_storage(reporter)

    # Compute how many more iterations are needed:
    n_iter_remain = exchange_attempts-simulation.iteration

    print(f"Continuing OpenMM replica exchange simulation from iteration {simulation.iteration}")
    print(f"Time step: {simulation_time_step}")
    print(f"New iterations: {n_iter_remain}",flush=True)

    simulation.extend(n_iterations=n_iter_remain)

    return
   
        
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
    
    # To improve pdf render speed, sparsify data to display less than 2000 data points
    n_xdata = len(simulation_times)
    
    if n_xdata <= 1000:
        plot_stride = 1
    else:
        plot_stride = int(np.floor(n_xdata/1000))
    
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
                    simulation_times[::plot_stride],
                    state_energies[state,::plot_stride],
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
    
    
def plot_replica_state_matrix(
    replica_state_indices,
    file_name='state_probability_matrix.pdf'
    ):
    
    # Plot a matrix of replica vs. state, coloring each box in the grid by normalized frequency 
    # For each replica, histogram the state indices data 
    # Then normalize the data and create [n_replica x n_state] patch graph
    
    n_replicas = replica_state_indices.shape[0]
    
    hist_all = np.zeros((n_replicas, n_replicas))
    
    state_bin_edges = np.linspace(-0.5,n_replicas-0.5,n_replicas+1)
    state_bin_centers = 0.5+state_bin_edges[0:n_replicas]
    
    for rep in range(n_replicas):
        hist_all[rep,:], bin_edges = np.histogram(
            replica_state_indices[rep,:],bins=state_bin_edges,density=True,
        )
        
    # No need for global normalization, since each replica's state probabilities must sum to 1
    
    hist_norm = np.zeros_like(hist_all)
    for rep in range(n_replicas):
        for state in range(n_replicas):
            hist_norm[rep,state] = hist_all[rep,state]/np.max(hist_all[rep,:])    
    
    mean_score = np.mean(hist_norm)
    min_score = np.amin(hist_norm)
    
    ax = pyplot.subplot(111)
    
    cmap=pyplot.get_cmap('nipy_spectral') 
    norm=Normalize(vmin=0,vmax=1) 
    
    ax.imshow(hist_norm,cmap=cmap,norm=norm)
    ax.set_aspect('equal', 'box')
    
    # Append colorbar axis to right side
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="5%",pad=0.20)  
    
    pyplot.colorbar(
        cm.ScalarMappable(cmap=cmap,norm=norm),
        cax=cax,
        label='normalized frequency',
        )
     
    ax.set_xlabel("State")
    ax.set_ylabel("Replica")
    pyplot.suptitle(f"Replica exchange state probabilities\n(Mean: {mean_score:.4f} Min: {min_score:.4f})")  
    
    pyplot.savefig(file_name)
    pyplot.close()    
    
    return hist_all
    
    
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

    """
    
    simulation_times = np.array(
        [
            step * time_interval.value_in_unit(unit.picosecond)
            for step in range(len(replica_states[0]))
        ]
    )
    
    simulation_times += time_shift.value_in_unit(unit.picosecond)
    
    # To improve pdf render speed, sparsify data to display less than 2000 data points
    n_xdata = len(simulation_times)
    
    if n_xdata <= 1000:
        plot_stride = 1
    else:
        plot_stride = int(np.floor(n_xdata/1000))
    
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
                    simulation_times[::plot_stride],
                    state_indices[::plot_stride],
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
