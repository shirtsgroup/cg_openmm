# project.py

import signac
from flow import FlowProject
import os
from simtk import unit
import simtk.openmm as openmm
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
from cg_openmm.simulation.tools import minimize_structure
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.thermo.calc import *
from analyze_foldamers.ensembles.cluster import *
from analyze_foldamers.parameters.bond_distributions import *
from analyze_foldamers.parameters.angle_distributions import *
from openmmtools.cache import global_context_cache
from openmmtools.multistate import ReplicaExchangeSampler
import numpy as np
import pickle
import time
import warnings

# Create signac flow groups:
replica_exchange_group = FlowProject.make_group(name='replica_exchange')
analysis_group = FlowProject.make_group(name='analysis')
cleanup_group = FlowProject.make_group(name='cleanup')

proj_directory = os.getcwd()

# Set up signac flow labels:
@FlowProject.label
def run_replica_exchange_done(job):
    output_directory = os.path.join(job.workspace(),"output")
    output_data = os.path.join(output_directory, "output.nc")
    rep_exch_completed = 0
    if os.path.isfile(output_data):
        # ***Note: this can sometimes fail if the .nc files are currently being written to
        rep_exch_status = ReplicaExchangeSampler.read_status(output_data)
        rep_exch_completed = rep_exch_status.is_completed
    return rep_exch_completed
    
@FlowProject.label
def process_replica_exchange_done(job):
    return job.isfile("analysis_stats_discard_20ns.pkl")
    
@FlowProject.label
def heat_capacity_done(job):
    return job.isfile("opt_T_spacing.pkl")
    
@FlowProject.label    
def CEI_replica_exchange_done(job):
    output_directory = os.path.join(job.workspace(),"output_CEI")
    output_data = os.path.join(output_directory, "output.nc")
    rep_exch_completed = 0
    if os.path.isfile(output_data):
        # ***Note: this can sometimes fail if the .nc files are currently being written to
        rep_exch_status = ReplicaExchangeSampler.read_status(output_data)
        rep_exch_completed = rep_exch_status.is_completed
    return rep_exch_completed
    
@FlowProject.label    
def process_CEI_replica_exchange_done(job):    
    return job.isfile("output_CEI/state_probability_matrix.pdf")
    
@FlowProject.label
def CEI_heat_capacity_done(job):
    return job.isfile("output_CEI/heat_capacity_200ns.pdf")
    
@FlowProject.label
def state_trajectories_created(job):
    return job.isfile("output_CEI/state_12.dcd")
    
@FlowProject.label
def clustering_done(job):
    return job.isfile("output_CEI/native_medoid_min.dcd")
    
def native_contacts_done(job):
    return job.isfile("output_CEI/Q_vs_T_opt.pdf")
    
@FlowProject.label
def ramachandran_done(job):
    return job.isfile("output_CEI/ramachandran.pdf")
    
@FlowProject.label
def bonded_distributions_done(job):
    return job.isfile("output_CEI/bonds_all_states.pdf")
    
@FlowProject.label
def trajectory_cleanup_done(job):
    return job.isfile("output_CEI/heat_capacity_200ns.pdf") and (job.isfile("output_CEI/replica_1.dcd")==0) and (job.isfile("output_CEI/state_1.dcd")==0)
    
@replica_exchange_group
@FlowProject.operation
@FlowProject.post(run_replica_exchange_done)
def signac_run_replica_exchange(job):
    # Run replica exchange simulation for current job parameters 
    
    print(f'job_parameters:')
    print(job.sp)
    
    rep_exch_begin = time.perf_counter()
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    overwrite_files = True  # overwrite files.
    
    global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")    
    
    # Replica exchange simulation settings
    total_simulation_time = 50.0 * unit.nanosecond
    simulation_time_step = 5.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time / simulation_time_step))
    output_data = os.path.join(output_directory, "output.nc")
    number_replicas = job.sp.n_replica
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    exchange_frequency = job.sp.exch_freq  # Number of steps between exchange attempts
    collision_frequency = job.sp.coll_freq/unit.picosecond

    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False    
    
    mass = 100.0 * unit.amu

    # mass and charge are defaults.
    bb = {
        "particle_type_name": "bb",
        "sigma": job.sp.sigma_bb * unit.angstrom,
        "epsilon": job.sp.epsilon_bb * unit.kilojoules_per_mole,
        "mass": mass
    }
        
    sc = {
        "particle_type_name": "sc",
        "sigma": job.sp.sigma_sc * unit.angstrom,
        "epsilon": job.sp.epsilon_sc * unit.kilojoules_per_mole,
        "mass": mass
    }

    # Monomer definition
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }

    sequence = 24 * [A]

    # Bond definitions
    bond_lengths = {"default_bond_length": job.sp.equil_bond_length * unit.nanometer}

    bond_force_constants = {
        "default_bond_force_constant": job.sp.k_bond * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    # Bond angle definitions
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": job.sp.k_angle * unit.kilojoule_per_mole / unit.radian / unit.radian
    }

    equil_bond_angles = {
        "default_equil_bond_angle": job.sp.equil_bond_angle_bb_bb_sc * unit.degrees,
        "bb_bb_bb_equil_bond_angle": job.sp.equil_bond_angle_bb_bb_bb * unit.degrees}

    # torsion angle definitions
    torsion_force_constants = {
        "default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole,
        "bb_bb_bb_bb_torsion_force_constant": job.sp.k_torsion * unit.kilojoule_per_mole}

    # Need to substract 180 degrees from specified torsion for mdtraj consistency
    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": (job.sp.equil_torsion_angle_bb_bb_bb_bb-180) * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "sc_bb_bb_sc_torsion_periodicity": job.sp.torsion_periodicity,
        "bb_bb_bb_bb_torsion_periodicity": job.sp.torsion_periodicity,
        "bb_bb_bb_sc_torsion_periodicity": job.sp.torsion_periodicity,
    }

    # Get initial positions from local file
    pdb_path = os.path.join(proj_directory, f"initial_structure_trial_{job.sp.trial}.pdb")
    positions = PDBFile(pdb_path).getPositions()

    # Build a coarse grained model
    cgmodel = CGModel(
        particle_type_list=[bb, sc],
        bond_lengths=bond_lengths,
        bond_force_constants=bond_force_constants,
        bond_angle_force_constants=bond_angle_force_constants,
        torsion_force_constants=torsion_force_constants,
        equil_bond_angles=equil_bond_angles,
        torsion_phase_angles=torsion_phase_angles,
        torsion_periodicities=torsion_periodicities,
        include_nonbonded_forces=include_nonbonded_forces,
        include_bond_forces=include_bond_forces,
        include_bond_angle_forces=include_bond_angle_forces,
        include_torsion_forces=include_torsion_forces,
        constrain_bonds=constrain_bonds,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )

    # store the cg model so that we can do various analyses.
    cgmodel.export(job.fn("stored_cgmodel.pkl"))

    if not os.path.exists(output_data) or overwrite_files == True:
        run_replica_exchange(
            cgmodel.topology,
            cgmodel.system,
            cgmodel.positions,
            friction=collision_frequency,
            temperature_list=temperature_list,
            simulation_time_step=simulation_time_step,
            total_simulation_time=total_simulation_time,
            exchange_frequency=exchange_frequency,
            output_data=output_data,
        )
    else:
        print("Replica output files exist")
        
    rep_exch_end = time.perf_counter()

    print(f'replica exchange run time: {rep_exch_end-rep_exch_begin}') 
    
    
@replica_exchange_group
@FlowProject.operation
@FlowProject.pre(run_replica_exchange_done)
@FlowProject.post(process_replica_exchange_done)
def signac_process_replica_exchange(job):
    # Process replica exchange data
    analysis_stats = {}

    # Job settings
    output_directory = os.path.join(job.workspace(),"output")
    output_data = os.path.join(output_directory, "output.nc")

    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        write_data_file=False,
        print_timing=True,
        frame_begin=20000,
    )

    analysis_stats["production_start"] = production_start
    analysis_stats["energy_decorrelation"] = sample_spacing
    analysis_stats["n_transit"] = n_transit
    
    # Mixing stats object can't be pickled directly
    analysis_stats["transition_matrix"] = mixing_stats[0]
    analysis_stats["eigenvalues"] = mixing_stats[1]
    analysis_stats["statistical_inefficiency"] = mixing_stats[2]

    pickle_out = open(job.fn("analysis_stats_discard_20ns.pkl"), "wb")
    pickle.dump(analysis_stats, pickle_out)
    pickle_out.close()
    
    # Save transition data to job data file:
    job.data['n_transit'] = n_transit

    
@analysis_group
@FlowProject.operation
@FlowProject.pre(process_replica_exchange_done)
@FlowProject.post(heat_capacity_done)
def signac_calc_heat_capacity(job):
    # Calculate heat capacity curve
    
    print(f'job_parameters:')
    print(job.sp)
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output")
    output_data = os.path.join(output_directory, "output.nc")

    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_discard_20ns.pkl"),"rb"))

    num_intermediate_states = 3
    
    # Read the simulation coordinates for individual temperature replicas                                                                     
    C_v, dC_v, new_temperature_list = get_heat_capacity(
        output_data=output_data,
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        num_intermediate_states=num_intermediate_states,
        plot_file=f"{output_directory}/heat_capacity_50ns.pdf",
    )

    # Save C_v data to data file:
    job.data['C_v'] = C_v
    job.data['dC_v'] = dC_v
    job.data['T_list_C_v'] = new_temperature_list    

    print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0].unit})")
    for i, C in enumerate(C_v):
        print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[i]._value:>10.4f}")
    
    # Compute CEI optimized temperature spacing:
    opt_temperature_list, deltaS_list = get_opt_temperature_list(
        new_temperature_list,
        C_v,
        number_intermediate_states=num_intermediate_states,
        plotfile=f'{output_directory}/Cv_spline_fit.pdf',
        verbose=True,
        )
        
    print("Entropy changes for each temperature interval")
    print(f"Temp1(K) Temp2(K) deltaS({deltaS_list[0].unit})")
    for i in range(len(deltaS_list)):
        print(f"{opt_temperature_list[i]._value:>8.2f}{opt_temperature_list[i+1]._value:>8.2f}{deltaS_list[i]._value:>10.6f}")
        
    pickle_out = open(job.fn("opt_T_spacing.pkl"), "wb")
    pickle.dump(opt_temperature_list, pickle_out)
    pickle_out.close()
    
    
@replica_exchange_group
@FlowProject.operation
@FlowProject.pre(heat_capacity_done)
@FlowProject.post(CEI_replica_exchange_done)
def signac_run_CEI_replica_exchange(job):
    # Run replica exchange simulation for current job parameters  
    
    print(f'job_parameters:')
    print(job.sp)
    
    rep_exch_begin = time.perf_counter()
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output_CEI")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    overwrite_files = True  # overwrite files.
    
    global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")    
    
    # Replica exchange simulation settings
    total_simulation_time = 200.0 * unit.nanosecond
    simulation_time_step = 5.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time / simulation_time_step))
    output_data = os.path.join(output_directory, "output.nc")
    number_replicas = job.sp.n_replica
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    
    # Load in CEI temperature list:
    temperature_list = pickle.load(open(job.fn("opt_T_spacing.pkl"),"rb"))

    exchange_frequency = job.sp.exch_freq  # Number of steps between exchange attempts
    collision_frequency = job.sp.coll_freq/unit.picosecond

    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False    
    
    mass = 100.0 * unit.amu

    # mass and charge are defaults.
    bb = {
        "particle_type_name": "bb",
        "sigma": job.sp.sigma_bb * unit.angstrom,
        "epsilon": job.sp.epsilon_bb * unit.kilojoules_per_mole,
        "mass": mass
    }
        
    sc = {
        "particle_type_name": "sc",
        "sigma": job.sp.sigma_sc * unit.angstrom,
        "epsilon": job.sp.epsilon_sc * unit.kilojoules_per_mole,
        "mass": mass
    }

    # Monomer definition
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }

    sequence = 24 * [A]

    # Bond definitions
    bond_lengths = {"default_bond_length": job.sp.equil_bond_length * unit.nanometer}

    bond_force_constants = {
        "default_bond_force_constant": job.sp.k_bond * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    # Bond angle definitions
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": job.sp.k_angle * unit.kilojoule_per_mole / unit.radian / unit.radian
    }

    equil_bond_angles = {
        "default_equil_bond_angle": job.sp.equil_bond_angle_bb_bb_sc * unit.degrees,
        "bb_bb_bb_equil_bond_angle": job.sp.equil_bond_angle_bb_bb_bb * unit.degrees}

    # torsion angle definitions
    torsion_force_constants = {
        "default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole,
        "bb_bb_bb_bb_torsion_force_constant": job.sp.k_torsion * unit.kilojoule_per_mole}

    # Need to substract 180 degrees from specified torsion for mdtraj consistency
    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": (job.sp.equil_torsion_angle_bb_bb_bb_bb-180) * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "sc_bb_bb_sc_torsion_periodicity": job.sp.torsion_periodicity,
        "bb_bb_bb_bb_torsion_periodicity": job.sp.torsion_periodicity,
        "bb_bb_bb_sc_torsion_periodicity": job.sp.torsion_periodicity,
    }

    # Get initial positions from local file
    pdb_path = os.path.join(proj_directory, f"initial_structure_trial_{job.sp.trial}.pdb")
    positions = PDBFile(pdb_path).getPositions()

    # Build a coarse grained model
    cgmodel = CGModel(
        particle_type_list=[bb, sc],
        bond_lengths=bond_lengths,
        bond_force_constants=bond_force_constants,
        bond_angle_force_constants=bond_angle_force_constants,
        torsion_force_constants=torsion_force_constants,
        equil_bond_angles=equil_bond_angles,
        torsion_phase_angles=torsion_phase_angles,
        torsion_periodicities=torsion_periodicities,
        include_nonbonded_forces=include_nonbonded_forces,
        include_bond_forces=include_bond_forces,
        include_bond_angle_forces=include_bond_angle_forces,
        include_torsion_forces=include_torsion_forces,
        constrain_bonds=constrain_bonds,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )

    # store the cg model so that we can do various analyses.
    cgmodel.export(job.fn("stored_cgmodel.pkl"))

    if not os.path.exists(output_data) or overwrite_files == True:
        run_replica_exchange(
            cgmodel.topology,
            cgmodel.system,
            cgmodel.positions,
            friction=collision_frequency,
            temperature_list=temperature_list,
            simulation_time_step=simulation_time_step,
            total_simulation_time=total_simulation_time,
            exchange_frequency=exchange_frequency,
            output_data=output_data,
        )
    else:
        print("Replica output files exist")
        
    rep_exch_end = time.perf_counter()

    print(f'replica exchange run time: {rep_exch_end-rep_exch_begin}')     
    
    
@replica_exchange_group
@FlowProject.operation
@FlowProject.pre(CEI_replica_exchange_done)
@FlowProject.post(process_CEI_replica_exchange_done)
def signac_process_CEI_replica_exchange(job):
    # Process replica exchange data
    analysis_stats = {}

    # Job settings
    output_directory = os.path.join(job.workspace(),"output_CEI")
    output_data = os.path.join(output_directory, "output.nc")

    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        write_data_file=False,
        print_timing=True,
        frame_begin=20000,
    )

    analysis_stats["production_start"] = production_start
    analysis_stats["energy_decorrelation"] = sample_spacing
    analysis_stats["n_transit"] = n_transit
    
    # Mixing stats object can't be pickled directly
    analysis_stats["transition_matrix"] = mixing_stats[0]
    analysis_stats["eigenvalues"] = mixing_stats[1]
    analysis_stats["statistical_inefficiency"] = mixing_stats[2]

    # We can't modify pickle files on the fly - make a separate one here
    pickle_out = open(job.fn("analysis_stats_CEI_200ns.pkl"), "wb")
    pickle.dump(analysis_stats, pickle_out)
    pickle_out.close()
    
    # Save transition data to data file:
    job.data['n_transit_CEI'] = n_transit

    
@analysis_group
@FlowProject.operation
@FlowProject.pre(process_CEI_replica_exchange_done)
@FlowProject.pre(CEI_replica_exchange_done)
@FlowProject.post(CEI_heat_capacity_done)
def signac_calc_CEI_heat_capacity(job):
    # Calculate heat capacity curve
    
    print(f'job_parameters:')
    print(job.sp)    
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output_CEI")
    output_data = os.path.join(output_directory, "output.nc")

    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_CEI_200ns.pkl"),"rb"))

    # Read the simulation coordinates for individual temperature replicas                                                                     
    C_v, dC_v, new_temperature_list = get_heat_capacity(
        output_data=output_data,
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        num_intermediate_states=3,
        plot_file=f"{output_directory}/heat_capacity_200ns.pdf",
    )

    # Save C_v data to data file:
    job.data['CEI_C_v_300'] = C_v
    job.data['CEI_dC_v_300'] = dC_v
    job.data['CEI_T_list_C_v'] = new_temperature_list    

    print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0].unit})")
    for i, C in enumerate(C_v):
        print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[i]._value:>10.4f}")
     
    
@replica_exchange_group
@FlowProject.operation
@FlowProject.pre(CEI_replica_exchange_done)
@FlowProject.post(state_trajectories_created)
def signac_write_trajectories(job):    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output_CEI")
    output_data = os.path.join(output_directory, "output.nc")

    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))
    
    make_replica_dcd_files(
        cgmodel.topology,
        timestep=5*unit.femtosecond,
        time_interval=job.sp.exch_freq,
        output_dir=output_directory
    )

    make_state_dcd_files(
        cgmodel.topology,
        timestep=5*unit.femtosecond,
        time_interval=job.sp.exch_freq,
        output_dir=output_directory
    )        
    
@analysis_group
@FlowProject.operation
@FlowProject.pre(process_CEI_replica_exchange_done)
@FlowProject.pre(state_trajectories_created)
@FlowProject.post(clustering_done)
def signac_clustering(job):
    # Predict native structure from rmsd clustering:
    
    output_directory = os.path.join(job.workspace(),"output_CEI")
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_CEI_200ns.pkl"),"rb"))
    
    dcd_file_list_rep = []
    number_replicas = job.sp.n_replica
    
    for rep in range(number_replicas):
        dcd_file_list_rep.append(f"{output_directory}/replica_{rep+1}.dcd")
    
    # Cluster all trajectory RMSD data using DBSCAN
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = get_cluster_medoid_positions_DBSCAN(
        file_list=dcd_file_list_rep,
        cgmodel=cgmodel,
        min_samples=50,
        eps=0.10,
        frame_start=analysis_stats["production_start"],
        frame_stride=200, # This can cause memory issues if too small
        filter=True,
        filter_ratio=0.25,
        output_dir=output_directory,
        )
    
    print(f'cluster_sizes: {cluster_sizes}')
    print(f'noise_points: {n_noise}')
    print(f'cluster_rmsd: {cluster_rmsd}')
    print(f'avg_silhouette: {silhouette_avg}')
    
    job.data['cluster_sizes'] = cluster_sizes
    job.data['noise_points'] = n_noise
    job.data['cluster_rmsd'] = cluster_rmsd
    job.data['avg_silhouette'] = silhouette_avg

    # Choose the medoid cluster with the smallest rmsd as the native structure.
    k_min = np.argmin(cluster_rmsd)

    # Minimize energy of native structure
    positions, PE_start, PE_end, simulation = minimize_structure(
        cgmodel,
        medoid_positions[k_min],
        output_file=f"{output_directory}/native_medoid_min.dcd",
    )

    job.data['native_PE'] = PE_start
    job.data['native_PE_min'] = PE_end

    
@analysis_group
@FlowProject.operation
@FlowProject.pre(clustering_done)
@FlowProject.post(native_contacts_done)
def signac_calc_native_contacts(job):
    # Optimize native contact cutoff parameters, and calculate native contact fraction
    print(f'job_parameters:')
    print(job.sp)
    output_directory = os.path.join(job.workspace(),"output_CEI")
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_CEI_200ns.pkl"),"rb"))    
    
    native_structure_file = f"{output_directory}/native_medoid_min.dcd"
    
    dcd_file_list_rep = []
    number_replicas = job.sp.n_replica
    
    for rep in range(number_replicas):
        dcd_file_list_rep.append(f"{output_directory}/replica_{rep+1}.dcd")
    
    (native_contact_cutoff, native_contact_tol, \
    opt_results, Q_expect_results, sigmoid_param_opt, \
    sigmoid_param_cov, contact_type_dict) = optimize_Q_cut(
        cgmodel,
        native_structure_file,
        dcd_file_list_rep,
        output_data=f"{output_directory}/output.nc",
        num_intermediate_states=0,
        frame_begin=analysis_stats["production_start"],
        frame_stride=analysis_stats["energy_decorrelation"],
        plotfile=f"{output_directory}/Q_vs_T_opt.pdf",
        verbose=True,
        )
        
    print(f'Contacts summary: {contact_type_dict}')
    print(f'nc_cut: {native_contact_cutoff}')
    print(f'nc_tol: {native_contact_tol}')
    print(f'{opt_results}')
        
    job.data['nc_cut'] = native_contact_cutoff
    job.data['nc_cut_tol'] = native_contact_tol
    job.data['T_list_Q_expect'] = Q_expect_results['T']
    job.data['Q_expect'] = Q_expect_results['Q']
    job.data['dQ_expect'] = Q_expect_results['dQ']
    
    # Q_folded should be calculated within secondary_structure code,
    # instead of here.
    # job.data['Q_folded'] = 
    
    # Using optimized paramters, compute the free energy of folding
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )
    
    print(f'nc list: {native_contact_list}')
    print(f'nc_distances: {native_contact_distances}')
   
    # Determine native contact fraction of current trajectories:
    Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts(
        cgmodel,
        dcd_file_list_rep,
        native_contact_list,
        native_contact_distances,
        frame_begin=analysis_stats["production_start"],
        native_contact_tol=native_contact_tol,
    )
    
    plot_native_contact_timeseries(
        Q,
        frame_begin=analysis_stats["production_start"],
        time_interval=1*unit.picosecond,
        plot_per_page=3,
        plotfile=f"{output_directory}/Q_vs_time.pdf",
        figure_title="Native contact fraction",
    )
    
    # array_folded_states = np.zeros((len(Q[:,0]),number_replicas))

    # for rep in range(number_replicas):
        # # Classify into folded/unfolded states:
        # for frame in range(len(Q[:,rep])):
            # if Q[frame,rep] >= Q_folded:
                # # Folded
                # array_folded_states[frame,rep] = 0
            # else:
                # # Unfolded
                # array_folded_states[frame,rep] = 1
    
    # job.data['Q_decorrelation'] = decorrelation_time
   
@analysis_group
@FlowProject.operation
@FlowProject.pre(state_trajectories_created)
@FlowProject.post(ramachandran_done)
def signac_ramachandran(job):
    # Make alpha-theta ramachandran plots:
    
    output_directory = os.path.join(job.workspace(),"output")
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_CEI_200ns.pkl"),"rb"))    
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))    
    
    traj_file_list = []
    number_replicas = job.sp.n_replica
    
    for i in range(number_replicas):
        traj_file_list.append(f"{output_directory}/state_{i+1}.dcd")
    
    rama_hist, xedges, yedges = calc_ramachandran(
        cgmodel,
        traj_file_list,
        plotfile=f"{output_directory}/ramachandran.pdf",
        frame_start=analysis_stats["production_start"],
        )
        
        
@analysis_group
@FlowProject.operation
@FlowProject.pre(state_trajectories_created)
@FlowProject.post(bonded_distributions_done)
def signac_bonded_distributions(job):
    # Make alpha-theta ramachandran plots:
    
    output_directory = os.path.join(job.workspace(),"output_CEI")
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_CEI_200ns.pkl"),"rb"))    
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))
    
    traj_file_list = []
    number_replicas = job.sp.n_replica

    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    
    # This temp list is used for labeling the plots:
    temperature_list = pickle.load(open(job.fn("opt_T_spacing.pkl"),"rb"))
    
    for i in range(number_replicas):
        traj_file_list.append(f"{output_directory}/state_{i+1}.dcd")
    
    bond_hist_data = calc_bond_length_distribution(
        cgmodel, traj_file_list, 
        frame_start=analysis_stats["production_start"],
        temperature_list=temperature_list,
        plotfile=f"{output_directory}/bonds_all_states.pdf")
        
    angle_hist_data = calc_bond_angle_distribution(
        cgmodel, traj_file_list,
        frame_start=analysis_stats["production_start"],
        temperature_list=temperature_list,
        plotfile=f"{output_directory}/angles_all_states.pdf")
        
    bond_hist_data = calc_torsion_distribution(
        cgmodel, traj_file_list,
        frame_start=analysis_stats["production_start"],
        temperature_list=temperature_list,
        plotfile=f"{output_directory}/torsions_all_states.pdf")
        
@cleanup_group
@FlowProject.operation
@FlowProject.pre(state_trajectories_created)
@FlowProject.post(trajectory_cleanup_done)
def signac_cleanup_trajectories(job):
    output_directory = os.path.join(job.workspace(),"output_CEI")
    if job.isfile("output/replica_1.dcd"):
        # Clean up replica trajectories:
        for i in range(job.sp.n_replica):
            os.remove(f'{output_directory}/replica_{i+1}.dcd')
            
    if job.isfile("output/state_1.dcd"):
        # Clean up state trajectories:
        for i in range(job.sp.n_replica):
            os.remove(f'{output_directory}/state_{i+1}.dcd')
   
if __name__ == '__main__':
    FlowProject().main()
