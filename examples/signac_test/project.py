# project.py

import signac
from flow import FlowProject
import os
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
from openmmtools.cache import global_context_cache
import numpy as np
import simtk.openmm as openmm
import pickle
from cg_openmm.thermo.calc import *

global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")

replica_exchange_group = FlowProject.make_group(name='replica_exchange')
analysis_group = FlowProject.make_group(name='analysis')

proj_directory = os.getcwd()

@FlowProject.label
def run_replica_exchange_done(job):
    return job.isfile("output/output.nc")
    
@FlowProject.label
def process_replica_exchange_done(job):
    return job.isfile("output/rep_ex_states.pdf")
    
@FlowProject.label
def heat_capacity_done(job):
    return job.isfile("output/heat_capacity.pdf")
    
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

    # Replica exchange simulation settings
    total_simulation_time = 0.05 * unit.nanosecond
    simulation_time_step = 10.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time / simulation_time_step))
    output_data = os.path.join(output_directory, "output.nc")
    number_replicas = 36
    min_temp = 100.0 * unit.kelvin
    max_temp = 500.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    exchange_frequency = 100  # Number of steps between exchange attempts
    collision_frequency = 5/unit.picosecond

    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False

    bond_length = 0.2 * unit.nanometer  # reference length unit

    # Particle definitions
    r_min = 1.5 * bond_length  # Lennard-Jones potential r_min
    # Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
    sigma = r_min / (2.0 ** (1.0 / 6.0))
    epsilon = 1.0 * unit.kilojoules_per_mole
    mass = 100.0 * unit.amu

    # mass and charge are defaults.
    bb = {"particle_type_name": "bb", "sigma": sigma, "epsilon": epsilon, "mass": mass}
    sc = {"particle_type_name": "sc", "sigma": sigma, "epsilon": epsilon, "mass": mass}

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
    bond_lengths = {"default_bond_length": bond_length}

    bond_force_constants = {
        "default_bond_force_constant": 1000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    # Bond angle definitions
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 25.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
    }

    equil_bond_angles = {"default_equil_bond_angle": job.sp.theta * unit.degrees}

    # torsion angle definitions
    torsion_force_constants = {
        "default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole,
        "bb_bb_bb_bb_torsion_force_constant": 3 * unit.kilojoule_per_mole}

    equil_torsion_angles = {
        "sc_bb_bb_sc_equil_torsion_angle": 75 * unit.degrees,
        "bb_bb_bb_bb_equil_torsion_angle": job.sp.alpha * unit.degrees,
        "bb_bb_bb_sc_equil_torsion_angle": 75 * unit.degrees,
    }

    torsion_periodicities = {
        "sc_bb_bb_sc_torsion_periodicity": 1,
        "bb_bb_bb_bb_torsion_periodicity": 1,
        "bb_bb_bb_sc_torsion_periodicity": 1,
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
        equil_torsion_angles=equil_torsion_angles,
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
    )

    analysis_stats["production_start"] = production_start
    analysis_stats["energy_decorrelation"] = sample_spacing

    pickle_out = open(job.fn("analysis_stats.pkl"), "wb")
    pickle.dump(analysis_stats, pickle_out)
    pickle_out.close()

    make_replica_dcd_files(
        cgmodel.topology,
        replica_positions,
        timestep=10*unit.femtosecond,
        time_interval=1000,
        output_dir=output_directory
    )

    make_state_dcd_files(
        cgmodel.topology,
        replica_positions,
        replica_states,
        timestep=10*unit.femtosecond,
        time_interval=1000,
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
    min_temp = 100 * unit.kelvin
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

    # This should be written to a file for each job
    print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0].unit})")
    for i, C in enumerate(C_v):
        print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[i]._value:>10.4f}")
    
    
if __name__ == '__main__':
    FlowProject().main()
