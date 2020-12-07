# project.py

import signac
from flow import FlowProject
import os
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
from analyze_foldamers.ensembles.cluster import *
from analyze_foldamers.parameters.bond_distributions import *
from analyze_foldamers.parameters.angle_distributions import *
from openmmtools.cache import global_context_cache
from openmmtools.multistate import ReplicaExchangeSampler
import numpy as np
import simtk.openmm as openmm
import pickle
from cg_openmm.thermo.calc import *

replica_exchange_group = FlowProject.make_group(name='replica_exchange')
analysis_group = FlowProject.make_group(name='analysis')

proj_directory = os.getcwd()

@FlowProject.label
def run_replica_exchange_done(job):
    output_directory = os.path.join(job.workspace(),"output")
    output_data = os.path.join(output_directory, "output.nc")
    rep_exch_completed = 0
    if os.path.isfile(output_data):
        rep_exch_status = ReplicaExchangeSampler.read_status(output_data)
        rep_exch_completed = rep_exch_status.is_completed
    return rep_exch_completed
    
@FlowProject.label
def process_replica_exchange_done(job):
    return job.isfile("output/state_36.dcd")
    
@FlowProject.label
def heat_capacity_done(job):
    return job.isfile("output/heat_capacity.pdf")
    
@FlowProject.label
def state_trajectories_created(job):
    return job.isfile("output/state_1.dcd")
    
@FlowProject.label
def clustering_done(job):
    return job.isfile("output/native_medoid_min.pdb")
    
@FlowProject.label
def ramachandran_done(job):
    return job.isfile("output/ramachandran.pdb")
    
@FlowProject.label
def bonded_distributions_done(job):
    return job.isfile("output/bonds_all_states.pdf")
    
    
@replica_exchange_group
@FlowProject.operation
@FlowProject.post(run_replica_exchange_done)
def signac_run_replica_exchange(job):
    # Run replica exchange simulation for current job parameters
    # equil_bond_angle = job.sp.theta
    # equil_torsion_angle = job.sp.alpha    
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    overwrite_files = True  # overwrite files.
    
    global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")    
    
    # Replica exchange simulation settings
    total_simulation_time = 20.0 * unit.nanosecond
    simulation_time_step = 5.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time / simulation_time_step))
    output_data = os.path.join(output_directory, "output.nc")
    number_replicas = 36
    min_temp = 200.0 * unit.kelvin
    max_temp = 500.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    exchange_frequency = 200  # Number of steps between exchange attempts
    collision_frequency = 5/unit.picosecond

    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False    
    
    mass = 100.0 * unit.amu

    # mass and charge are defaults.
    bb = {
        "particle_type_name": "bb",
        "sigma": job.sp.sigma * unit.angstrom,
        "epsilon": job.sp.epsilon * unit.kilojoules_per_mole,
        "mass": mass
    }
        
    sc = {
        "particle_type_name": "sc",
        "sigma": job.sp.sigma * unit.angstrom,
        "epsilon": job.sp.epsilon * unit.kilojoules_per_mole,
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

    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": job.sp.torsion_phase_angle_bb_bb_bb_bb * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "sc_bb_bb_sc_torsion_periodicity": job.sp.torsion_periodicity,
        "bb_bb_bb_bb_torsion_periodicity": job.sp.torsion_periodicity,
        "bb_bb_bb_sc_torsion_periodicity": job.sp.torsion_periodicity,
    }

    # Get initial positions from local file
    pdb_path = os.path.join(proj_directory, "24mer_1b1s_initial_structure.pdb")
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
        sequence=sequence,
        positions=positions,
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
    replica_energies, replica_positions, replica_states, production_start, sample_spacing = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        write_data_file=False,
        detect_equilibration=True,
    )

    analysis_stats["production_start"] = production_start
    analysis_stats["energy_decorrelation"] = sample_spacing

    pickle_out = open(job.fn("analysis_stats.pkl"), "wb")
    pickle.dump(analysis_stats, pickle_out)
    pickle_out.close()

    make_replica_dcd_files(
        cgmodel.topology,
        replica_positions,
        timestep=5*unit.femtosecond,
        time_interval=200,
        output_dir=output_directory
    )

    make_state_dcd_files(
        cgmodel.topology,
        replica_positions,
        replica_states,
        timestep=5*unit.femtosecond,
        time_interval=200,
        output_dir=output_directory
    )

@analysis_group
@FlowProject.operation
@FlowProject.pre(process_replica_exchange_done)
@FlowProject.post(heat_capacity_done)
def signac_calc_heat_capacity(job):
    # Calculate heat capacity curve
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output")
    output_data = os.path.join(output_directory, "output.nc")
    
    # Replica exchange simulation settings.
    #These must match the simulations that are being analyzed.
    number_replicas = 36
    min_temp = 200.0 * unit.kelvin
    max_temp = 500.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats.pkl"),"rb"))

    # Read the simulation coordinates for individual temperature replicas                                                                     
    C_v, dC_v, new_temperature_list = get_heat_capacity(
        temperature_list,
        output_data=output_data,
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        num_intermediate_states=1,
        plot_file=f"{output_directory}/heat_capacity.pdf",
    )

    # Save C_v data to data file:
    job.data['C_v'] = C_v
    job.data['dC_v'] = dC_v
    job.data['T_list_C_v'] = new_temperature_list    

    print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0].unit})")
    for i, C in enumerate(C_v):
        print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[i]._value:>10.4f}")
 
    
@analysis_group
@FlowProject.operation
@FlowProject.pre(process_replica_exchange_done)
@FlowProject.post(clustering_done)
def signac_clustering(job):
    # Predict native structure from rmsd clustering:
    
    output_directory = os.path.join(job.workspace(),"output")
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats.pkl"),"rb"))
    
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = get_cluster_medoid_positions_DBSCAN(
        file_list=dcd_file_list_rep,
        cgmodel=cgmodel,
        min_samples=50,
        eps=0.1,
        frame_start=analysis_stats["production_start"],
        frame_stride=50,
        frame_end=frame_end,
        filter=True,
        filter_ratio=filter_ratio,
        output_dir=output_directory,
        )
    
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
        output_file=f"{output_directory}/native_medoid_min.pdb",
    )

    job.data['native_positions'] = medoid_positions[k_min]
    job.data['native_positions_min'] = positions
    job.data['native_PE'] = PE_start
    job.data['native_PE_min'] = PE_end
    
    
@analysis_group
@FlowProject.operation
@FlowProject.pre(state_trajectories_created)
@FlowProject.post(ramachandran_done)
def signac_ramachandran(job):
    # Make alpha-theta ramachandran plots:
    
    output_directory = os.path.join(job.workspace(),"output")
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats.pkl"),"rb"))    
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))    
    
    traj_file_list = []
    number_replicas = 36
    
    for i in range(number_replicas):
        traj_file_list.append(f"{output_directory}/state_{rep+1}.dcd")
    
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
    
    output_directory = os.path.join(job.workspace(),"output")
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats.pkl"),"rb"))    
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))    
    
    traj_file_list = []
    number_replicas = 36

    min_temp = 200.0 * unit.kelvin
    max_temp = 500.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
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
        
   
if __name__ == '__main__':
    FlowProject().main()
