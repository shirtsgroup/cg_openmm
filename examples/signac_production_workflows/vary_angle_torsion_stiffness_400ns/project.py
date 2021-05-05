# project.py

import signac
from flow import FlowProject
import os
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
from cg_openmm.simulation.tools import minimize_structure
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.parameters.free_energy import *
from analyze_foldamers.ensembles.cluster import *
from analyze_foldamers.parameters.bond_distributions import *
from analyze_foldamers.parameters.angle_distributions import *
from openmmtools.cache import global_context_cache
from openmmtools.multistate import ReplicaExchangeSampler
import numpy as np
import simtk.openmm as openmm
import pickle
from cg_openmm.thermo.calc import *
import time

replica_exchange_group = FlowProject.make_group(name='replica_exchange')
analysis_group = FlowProject.make_group(name='analysis')
cleanup_group = FlowProject.make_group(name='cleanup')

proj_directory = os.getcwd()

@FlowProject.label
def run_replica_exchange_done(job):
    output_directory = os.path.join(job.workspace(),"output")
    output_data = os.path.join(output_directory, "output.nc")
    rep_exch_completed = 0
    if os.path.isfile(output_data):
        rep_exch_status = ReplicaExchangeSampler.read_status(output_data)
        rep_exch_completed = rep_exch_status.is_completed
    return os.path.isfile(output_data)
    
@FlowProject.label
def process_replica_exchange_done(job):
    return job.isfile("analysis_stats_discard_20ns.pkl")
    
@FlowProject.label
def heat_capacity_done(job):
    output_directory = os.path.join(job.workspace(),"output")
    return job.isfile(f"opt_T_spacing.pkl")
    
@FlowProject.label    
def CEI_replica_exchange_done(job):
    output_directory = os.path.join(job.workspace(),"output_CEI")
    output_data = os.path.join(output_directory, "output.nc")
    rep_exch_completed = 0
    # if os.path.isfile(output_data):
        # rep_exch_status = ReplicaExchangeSampler.read_status(output_data)
        # rep_exch_completed = rep_exch_status.is_completed
    return os.path.isfile(output_data)
    
@FlowProject.label    
def process_CEI_replica_exchange_done(job):    
    return job.isfile(f"output_CEI/state_probability_matrix.pdf")
    
@FlowProject.label
def CEI_heat_capacity_done(job):
    output_directory = os.path.join(job.workspace(),"output_CEI")
    return job.isfile(f"{output_directory}/heat_capacity_200ns.pdf")
    
@FlowProject.label
def boot_CEI_heat_capacity_done(job):
    output_directory = os.path.join(job.workspace(),"output_CEI")
    return job.isfile(f"{output_directory}/heat_capacity_200ns_boot_500_std_ana.pdf")  
    
@FlowProject.label
def state_trajectories_created(job):
    return job.isfile("output_CEI/state_12.dcd")
    
@FlowProject.label
def clustering_done(job):
    return job.isfile("output_CEI/native_medoid_min.dcd")
    
@FlowProject.label    
def native_contacts_done(job):
    return job.isfile("output_CEI/native_contacts_boot200.pdf")
    
@FlowProject.label    
def native_contacts_fixed_tol_done(job):
    return job.isfile("output_CEI/native_contacts_boot200_fixed_tol1_3.pdf")
    
@FlowProject.label
def ramachandran_done(job):
    return job.isfile("output_CEI/ramachandran.pdb")
    
@FlowProject.label
def bonded_distributions_done(job):
    return job.isfile("output_CEI/bonds_all_states_.pdf")
    
@FlowProject.label
def trajectory_cleanup_done(job):
    return job.isfile("output_CEI/heat_capacity_200ns.pdf") and (job.isfile("output_CEI/replica_1.dcd")==0) and (job.isfile("output_CEI/state_1.dcd")==0)
    
@replica_exchange_group
@FlowProject.operation
@FlowProject.post(run_replica_exchange_done)
def signac_run_replica_exchange(job):
    # Run replica exchange simulation for current job parameters 
    
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
        "default_equil_bond_angle": (360-job.sp.equil_bond_angle_bb_bb_bb)/2 * unit.degrees,
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
    
    # Save transition data to data file:
    job.data['n_transit'] = n_transit

    
@analysis_group
@FlowProject.operation
@FlowProject.pre(process_replica_exchange_done)
@FlowProject.post(heat_capacity_done)
def signac_calc_heat_capacity(job):
    # Calculate heat capacity curve
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output")
    output_data = os.path.join(output_directory, "output.nc")

    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_discard_20ns.pkl"),"rb"))

    # Read the simulation coordinates for individual temperature replicas                                                                     
    C_v, dC_v, new_temperature_list = get_heat_capacity(
        output_data=output_data,
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        num_intermediate_states=3,
        plot_file=f"{output_directory}/heat_capacity_g75.pdf",
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
        number_intermediate_states=3,
        plotfile=f'{output_directory}/Cv_spline_fit.pdf',
        verbose=True,
        )
        
    print(f"T({opt_temperature_list[0].unit})")
        
    print("Entropy changes for each temperature interval")
    print(f"Temp1 Temp2 deltaS")
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
    
    rep_exch_begin = time.perf_counter()
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output_CEI")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    overwrite_files = True  # overwrite files.
    
    global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")    
    
    # Replica exchange simulation settings
    total_simulation_time = 400.0 * unit.nanosecond
    simulation_time_step = 5.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time / simulation_time_step))
    output_data = os.path.join(output_directory, "output.nc")
    number_replicas = job.sp.n_replica
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    
    # Load in trajectory stats:
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
        "default_equil_bond_angle": (360-job.sp.equil_bond_angle_bb_bb_bb)/2 * unit.degrees,
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
        frame_begin=300000,
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

    
@replica_exchange_group
@FlowProject.operation
@FlowProject.pre(process_CEI_replica_exchange_done)
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
@FlowProject.pre(CEI_replica_exchange_done)
@FlowProject.post(CEI_heat_capacity_done)
def signac_calc_CEI_heat_capacity(job):
    # Calculate heat capacity curve
    
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
     
     
@analysis_group
@FlowProject.operation
@FlowProject.pre(process_CEI_replica_exchange_done)
@FlowProject.pre(CEI_replica_exchange_done)
@FlowProject.post(boot_CEI_heat_capacity_done)
def signac_calc_CEI_heat_capacity_boot(job):
    # Calculate heat capacity curve
    
    # Job settings
    output_directory = os.path.join(job.workspace(),"output_CEI")
    output_data = os.path.join(output_directory, "output.nc")

    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_CEI_200ns.pkl"),"rb"))

    # Read the simulation coordinates for individual temperature replicas                                                                     
    (new_temperature_list, C_v, dC_v, Tm_value, Tm_uncertainty, 
    Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty) = bootstrap_heat_capacity(
        output_data=output_data,
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        num_intermediate_states=3,
        n_trial_boot=500,
        conf_percent='sigma',
        plot_file=f"{output_directory}/heat_capacity_200ns_boot_500_std_ana.pdf",
    )

    # Save C_v data to data file:
    # Need to convert tuples to numpy arrays
    job.data['CEI_C_v_boot'] = C_v
    job.data['CEI_dC_v_lo_boot'] = dC_v[0]
    job.data['CEI_dC_v_hi_boot'] = dC_v[1]
    job.data['CEI_T_list_C_v_boot'] = new_temperature_list   
    job.data['Tm_boot'] = Tm_value
    job.data['Tm_uncertainty_lo_boot'] = Tm_uncertainty[0]
    job.data['Tm_uncertainty_hi_boot'] = Tm_uncertainty[1]
    job.data['Cv_height_boot'] = Cv_height_value
    job.data['Cv_uncertainty_lo_boot'] = Cv_height_uncertainty[0]
    job.data['Cv_uncertainty_hi_boot'] = Cv_height_uncertainty[1]
    job.data['FWHM_boot'] = FWHM_value
    job.data['FWHM_uncertainty_lo_boot'] = FWHM_uncertainty[0]
    job.data['FWHM_uncertainty_hi_boot'] = FWHM_uncertainty[1]

    print(f"Using bootstrap parameters: n_sample=1000, n_trial=500")
    print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0][0].unit})")
    for i, C in enumerate(C_v):
        print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[0][i]._value:>10.4f} {dC_v[1][i]._value:>10.4f}")
    
    print(f"Tmelt (K)")
    print(f"{Tm_value._value:8.2f} {Tm_uncertainty[0]._value:8.2f} {Tm_uncertainty[1]._value:8.2f}")    

    print(f"Cv_height (kJ/mol/K)")
    print(f"{Cv_height_value._value:8.2f} {Cv_height_uncertainty[0]._value:8.2f} {Cv_height_uncertainty[1]._value:8.2f}")   

    print(f"FWHM (K)")
    print(f"{FWHM_value._value:8.2f} {FWHM_uncertainty[0]._value:8.2f} {FWHM_uncertainty[1]._value:8.2f}")   

    
@analysis_group
@FlowProject.operation
@FlowProject.pre(CEI_replica_exchange_done)
@FlowProject.pre(process_CEI_replica_exchange_done)
@FlowProject.pre(state_trajectories_created)
@FlowProject.post(clustering_done)
def signac_clustering(job):
    # Predict native structure from rmsd clustering:
    print(f'job_parameters:')
    print(job.sp)
    output_directory = os.path.join(job.workspace(),"output_CEI")
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_CEI_200ns.pkl"),"rb"))
    
    dcd_file_list_rep = []
    number_replicas = job.sp.n_replica
    
    for rep in range(number_replicas):
        dcd_file_list_rep.append(f"{output_directory}/replica_{rep+1}.dcd")
    
    (medoid_positions, cluster_sizes, cluster_rmsd, n_noise,
    silhouette_avg, labels, original_indices) = get_cluster_medoid_positions_DBSCAN(
        file_list=dcd_file_list_rep,
        cgmodel=cgmodel,
        min_samples=100,
        eps=0.10,
        frame_start=analysis_stats["production_start"],
        frame_stride=200, # This can cause memory issues if too small
        filter=True,
        filter_ratio=0.25,
        output_dir=output_directory,
        core_points_only=False,
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

    
@analysis_group
@FlowProject.operation
@FlowProject.pre(clustering_done)
@FlowProject.post(native_contacts_fixed_tol_done)
def signac_calc_native_contacts_helix_fixed_tol(job):
    # Used a fixed native contact tolerance, and calculate native contact fraction
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
   
    # Get the native contact list and distances:
    native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
        cgmodel,
        native_structure_file,
        backbone_type_name='bb',
    )
    
    native_contact_tol = 1.3
    
    print(f'Optimal native contacts are i to i+{opt_seq_spacing}')
    print(f'Using fixed tolerance factor of {native_contact_tol}')
    
    # Bootstrap native contacts to get Q_folded and uncertainties
    temp_list, Q_values, Q_uncertainty, sigmoid_results_boot = bootstrap_native_contacts_expectation(
        cgmodel,
        dcd_file_list_rep,
        native_contact_list,
        native_contact_distances,
        output_data=f"{output_directory}/output.nc",
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        native_contact_tol=native_contact_tol,
        num_intermediate_states=3,
        n_trial_boot=200,
        conf_percent='sigma',
        plotfile=f'{output_directory}/native_contacts_boot200_fixed_tol1_3.pdf',
        )
    
    job.data['Q_folded_tol13'] = sigmoid_results_boot['Q_folded_value']
    job.data['Q_folded_uncertainty_tol13'] = sigmoid_results_boot['Q_folded_uncertainty'][1]
    
    job.data['sigmoid_d_tol13'] = sigmoid_results_boot['sigmoid_d_value']
    job.data['sigmoid_d_uncertainty_tol13'] = sigmoid_results_boot['sigmoid_d_uncertainty'][1]
    
    job.data['sigmoid_Q_min_tol13'] = sigmoid_results_boot['sigmoid_Q_min_value']
    job.data['sigmoid_Q_min_uncertainty_tol13'] = sigmoid_results_boot['sigmoid_Q_min_uncertainty'][1]    
    
    job.data['sigmoid_Q_max_tol13'] = sigmoid_results_boot['sigmoid_Q_max_value']
    job.data['sigmoid_Q_max_uncertainty_tol13'] = sigmoid_results_boot['sigmoid_Q_max_uncertainty'][1]    
    
    job.data['sigmoid_Tm_tol13'] = sigmoid_results_boot['sigmoid_Tm_value']
    job.data['sigmoid_Tm_uncertainty_tol13'] = sigmoid_results_boot['sigmoid_Tm_uncertainty'][1]
    
    job.data['Q_expect_boot_tol13'] = Q_values
    job.data['dQ_expect_lo_boot_tol13'] = Q_uncertainty[0]
    job.data['dQ_expect_hi_boot_tol13'] = Q_uncertainty[1]
    
    Q_folded = sigmoid_results_boot['Q_folded_value']
    print(f'Q_folded: {Q_folded}')
    
    # Using optimized paramters, compute the free energy of folding
    # Determine native contact fraction of current trajectories:
    Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts(
        cgmodel,
        dcd_file_list_rep,
        native_contact_list,
        native_contact_distances,
        frame_begin=analysis_stats["production_start"],
        native_contact_tol=native_contact_tol,
    )
    
    job.data['Q_decorrelation_tol13'] = decorrelation_time
    
    plot_native_contact_timeseries(
        Q,
        frame_begin=analysis_stats["production_start"],
        time_interval=1*unit.picosecond,
        plot_per_page=3,
        plotfile=f"{output_directory}/Q_vs_time_tol13.pdf",
        figure_title="Native contact fraction",
    )
    
    # Compute free energy/entropy/enthalpy of folding curves
        
    F_unit = unit.kilojoule / unit.mole 
    S_unit = F_unit / unit.kelvin
    U_unit = F_unit
        
    # From bootstrapping:
    (full_T_list_boot, deltaF_values_boot, deltaF_uncertainty_boot, \
        deltaS_values_boot, deltaS_uncertainty_boot, \
        deltaU_values_boot, deltaU_uncertainty_boot) = bootstrap_free_energy_folding(
        Q,
        Q_folded,
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        output_data=f"{output_directory}/output.nc",
        num_intermediate_states=3,
        n_trial_boot=200,
        conf_percent='sigma',
        plotfile_dir=output_directory,
    )

    
    deltaF_values_boot = deltaF_values_boot['state0_state1'].value_in_unit(F_unit)
    deltaF_uncertainty_boot = deltaF_uncertainty_boot['state0_state1'][1].value_in_unit(F_unit)
    
    job.data['deltaF_values_boot_tol13'] = deltaF_values_boot
    job.data['deltaF_uncertainty_boot_tol13'] = deltaF_uncertainty_boot
    
    deltaS_values_boot = deltaS_values_boot['state0_state1'].value_in_unit(S_unit)
    deltaS_uncertainty_boot = deltaS_uncertainty_boot['state0_state1'][1].value_in_unit(S_unit)
    
    job.data['deltaS_values_boot_tol13'] = deltaS_values_boot
    job.data['deltaS_uncertainty_boot_tol13'] = deltaS_uncertainty_boot    
    
    deltaU_values_boot = deltaU_values_boot['state0_state1'].value_in_unit(U_unit)
    deltaU_uncertainty_boot = deltaU_uncertainty_boot['state0_state1'][1].value_in_unit(U_unit)
    
    job.data['deltaU_values_boot_tol13'] = deltaU_values_boot
    job.data['deltaU_uncertainty_boot_tol13'] = deltaU_uncertainty_boot
    
    print(f"T (K), deltaF (kJ/mol), deltaF_uncertainty (kJ/mol), deltaF_boot, deltaF_uncertainty_boot")
    for i in range(len(full_T_list_boot)):
        print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
            {deltaF_values_boot[i]:>6.8f}, \
            {deltaF_uncertainty_boot[i]:>6.8f}")
            
    print(f"\nT (K), deltaS (kJ/mol/K), deltaS_boot, deltaS_uncertainty_boot")
    for i in range(len(full_T_list_boot)):
        print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
            {deltaS_values_boot[i]:>6.8f}, \
            {deltaS_uncertainty_boot[i]:>6.8f}")
            
    print(f"\nT (K), deltaU (kJ/mol), deltaU_boot, deltaU_uncertainty_boot")
    for i in range(len(full_T_list_boot)):
        print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
            {deltaU_values_boot[i]:>6.8f}, \
            {deltaU_uncertainty_boot[i]:>6.8f}")

   
@analysis_group
@FlowProject.operation
@FlowProject.pre(clustering_done)
@FlowProject.post(native_contacts_done)
def signac_calc_native_contacts_helix_boot(job):
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
    
    # Optimize helical native contacts and tolerance:
    (opt_seq_spacing, native_contact_tol, opt_results, Q_expect_results,
    sigmoid_param_opt, sigmoid_param_cov)  = optimize_Q_tol_helix(
        cgmodel,
        native_structure_file,
        dcd_file_list_rep,
        output_data=f"{output_directory}/output.nc",
        num_intermediate_states=3,
        frame_begin=analysis_stats["production_start"],
        frame_stride=analysis_stats["energy_decorrelation"],
        plotfile=f"{output_directory}/Q_vs_T_opt.pdf",
        verbose=True,
        backbone_type_name='bb',
        brute_step=0.05,
        )
        
    print(f'nc_cut_tol: {native_contact_tol}')
        
    job.data['nc_tol'] = native_contact_tol
    job.data['T_list_Q_expect'] = Q_expect_results['T']
    job.data['Q_expect'] = Q_expect_results['Q']
    job.data['dQ_expect'] = Q_expect_results['dQ']
    
    # Get the native contact list and distances:
    native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
        cgmodel,
        native_structure_file,
        backbone_type_name='bb',
    )
    
    print(f'Optimal native contacts are i to i+{opt_seq_spacing}')
    job.data['opt_seq_helix'] = opt_seq_spacing
    
    # Bootstrap native contacts to get Q_folded and uncertainties
    temp_list, Q_values, Q_uncertainty, sigmoid_results_boot = bootstrap_native_contacts_expectation(
        cgmodel,
        dcd_file_list_rep,
        native_contact_list,
        native_contact_distances,
        output_data=f"{output_directory}/output.nc",
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        native_contact_tol=native_contact_tol,
        num_intermediate_states=3,
        n_trial_boot=200,
        conf_percent='sigma',
        plotfile=f'{output_directory}/native_contacts_boot200.pdf',
        )
    
    job.data['Q_folded'] = sigmoid_results_boot['Q_folded_value']
    job.data['Q_folded_uncertainty'] = sigmoid_results_boot['Q_folded_uncertainty'][1]
    
    job.data['sigmoid_d'] = sigmoid_results_boot['sigmoid_d_value']
    job.data['sigmoid_d_uncertainty'] = sigmoid_results_boot['sigmoid_d_uncertainty'][1]
    
    job.data['sigmoid_Q_min'] = sigmoid_results_boot['sigmoid_Q_min_value']
    job.data['sigmoid_Q_min_uncertainty'] = sigmoid_results_boot['sigmoid_Q_min_uncertainty'][1]    
    
    job.data['sigmoid_Q_max'] = sigmoid_results_boot['sigmoid_Q_max_value']
    job.data['sigmoid_Q_max_uncertainty'] = sigmoid_results_boot['sigmoid_Q_max_uncertainty'][1]    
    
    job.data['sigmoid_Tm'] = sigmoid_results_boot['sigmoid_Tm_value']
    job.data['sigmoid_Tm_uncertainty'] = sigmoid_results_boot['sigmoid_Tm_uncertainty'][1]
    
    job.data['Q_expect_boot'] = Q_values
    job.data['dQ_expect_lo_boot'] = Q_uncertainty[0]
    job.data['dQ_expect_hi_boot'] = Q_uncertainty[1]
    
    Q_folded = sigmoid_results_boot['Q_folded_value']
    print(f'Q_folded: {Q_folded}')
    
    # Using optimized paramters, compute the free energy of folding
    # Determine native contact fraction of current trajectories:
    Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts(
        cgmodel,
        dcd_file_list_rep,
        native_contact_list,
        native_contact_distances,
        frame_begin=analysis_stats["production_start"],
        native_contact_tol=native_contact_tol,
    )
    
    job.data['Q_decorrelation'] = decorrelation_time
    
    plot_native_contact_timeseries(
        Q,
        frame_begin=analysis_stats["production_start"],
        time_interval=1*unit.picosecond,
        plot_per_page=3,
        plotfile=f"{output_directory}/Q_vs_time.pdf",
        figure_title="Native contact fraction",
    )
    
    # Compute free energy/entropy/enthalpy of folding curves
        
    F_unit = unit.kilojoule / unit.mole 
    S_unit = F_unit / unit.kelvin
    U_unit = F_unit
        
    # From bootstrapping:
    (full_T_list_boot, deltaF_values_boot, deltaF_uncertainty_boot, \
        deltaS_values_boot, deltaS_uncertainty_boot, \
        deltaU_values_boot, deltaU_uncertainty_boot) = bootstrap_free_energy_folding(
        Q,
        Q_folded,
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        output_data=f"{output_directory}/output.nc",
        num_intermediate_states=3,
        n_trial_boot=200,
        conf_percent='sigma',
        plotfile_dir=output_directory,
    )

    
    deltaF_values_boot = deltaF_values_boot['state0_state1'].value_in_unit(F_unit)
    deltaF_uncertainty_boot = deltaF_uncertainty_boot['state0_state1'][1].value_in_unit(F_unit)
    
    job.data['deltaF_values_boot'] = deltaF_values_boot
    job.data['deltaF_uncertainty_boot'] = deltaF_uncertainty_boot
    
    deltaS_values_boot = deltaS_values_boot['state0_state1'].value_in_unit(S_unit)
    deltaS_uncertainty_boot = deltaS_uncertainty_boot['state0_state1'][1].value_in_unit(S_unit)
    
    job.data['deltaS_values_boot'] = deltaS_values_boot
    job.data['deltaS_uncertainty_boot'] = deltaS_uncertainty_boot    
    
    deltaU_values_boot = deltaU_values_boot['state0_state1'].value_in_unit(U_unit)
    deltaU_uncertainty_boot = deltaU_uncertainty_boot['state0_state1'][1].value_in_unit(U_unit)
    
    job.data['deltaU_values_boot'] = deltaU_values_boot
    job.data['deltaU_uncertainty_boot'] = deltaU_uncertainty_boot
    
    print(f"T (K), deltaF (kJ/mol), deltaF_uncertainty (kJ/mol), deltaF_boot, deltaF_uncertainty_boot")
    for i in range(len(full_T_list_boot)):
        print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
            {deltaF_values_boot[i]:>6.8f}, \
            {deltaF_uncertainty_boot[i]:>6.8f}")
            
    print(f"\nT (K), deltaS (kJ/mol/K), deltaS_boot, deltaS_uncertainty_boot")
    for i in range(len(full_T_list_boot)):
        print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
            {deltaS_values_boot[i]:>6.8f}, \
            {deltaS_uncertainty_boot[i]:>6.8f}")
            
    print(f"\nT (K), deltaU (kJ/mol), deltaU_boot, deltaU_uncertainty_boot")
    for i in range(len(full_T_list_boot)):
        print(f"{full_T_list_boot[i].value_in_unit(unit.kelvin):>6.4f}, \
            {deltaU_values_boot[i]:>6.8f}, \
            {deltaU_uncertainty_boot[i]:>6.8f}")

   
@analysis_group
@FlowProject.operation
@FlowProject.pre(state_trajectories_created)
@FlowProject.post(ramachandran_done)
def signac_ramachandran(job):
    # Make alpha-theta ramachandran plots:
    
    output_directory = os.path.join(job.workspace(),"output_CEI")
    
    # Load in trajectory stats:
    analysis_stats = pickle.load(open(job.fn("analysis_stats_CEI_200ns.pkl"),"rb"))    
    
    # Load in cgmodel:
    cgmodel = pickle.load(open(job.fn("stored_cgmodel.pkl"),"rb"))  

    # This temp list is used for labeling the plots:
    temperature_list = pickle.load(open(job.fn("opt_T_spacing.pkl"),"rb"))    
    
    traj_file_list = []
    number_replicas = job.sp.n_replica
    
    for i in range(number_replicas):
        traj_file_list.append(f"{output_directory}/state_{i+1}.dcd")
    
    rama_hist, xedges, yedges = calc_ramachandran(
        cgmodel,
        traj_file_list,
        plotfile=f"{output_directory}/ramachandran.pdf",
        frame_start=analysis_stats["production_start"],
        temperature_list=temperature_list,
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
        nbins=180,
        plotfile=f"{output_directory}/angles_all_states.pdf")
        
    torsion_hist_data = calc_torsion_distribution(
        cgmodel, traj_file_list,
        frame_start=analysis_stats["production_start"],
        temperature_list=temperature_list,
        plotfile=f"{output_directory}/torsions_all_states.pdf")
        
    # Save angle hist data to pickle for further analysis/plotting
    pickle_out = open(job.fn("angle_hist_data.pkl"), "wb")
    pickle.dump(angle_hist_data, pickle_out)
    pickle_out.close()
    
    # Save torsion hist data to pickle for further analysis/plotting
    pickle_out = open(job.fn("torsion_hist_data.pkl"), "wb")
    pickle.dump(torsion_hist_data, pickle_out)
    pickle_out.close()
        
        
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
