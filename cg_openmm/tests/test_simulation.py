"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed
import cg_openmm
from simtk import unit
import pytest
import sys
import os
import simtk.openmm as openmm
from simtk.openmm.app.pdbfile import PDBFile
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.simulation.tools import *
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from cg_openmm.simulation.rep_exch import *
from cg_openmm.utilities import random_builder
from cg_openmm.build.cg_build import build_topology
import numpy as np
from numpy.testing import assert_almost_equal
from openmmtools.cache import global_context_cache
import pickle

def test_cg_openmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cg_openmm" in sys.modules
    
    
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_data')
structures_path = os.path.join(current_path, 'test_structures')
   
def test_minimize_structure_pdb(tmpdir):
    """Test energy minimization structure, with reading/writing pdb files"""
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))    

    native_structure_file=f"{structures_path}/medoid_0.pdb"

    native_traj = md.load(native_structure_file)    
    
    positions = native_traj.xyz[0] * unit.nanometer
    
    # Minimize energy of native structure
    positions, PE_start, PE_end, simulation = minimize_structure(
        cgmodel,
        positions,
        output_file=f"{structures_path}/medoid_min.pdb",
    )
    
    assert PE_end < PE_start
    assert os.path.isfile(f"{structures_path}/medoid_min.pdb")

    
def test_minimize_structure_dcd():
    """Test energy minimization structure, with reading/writing dcd files"""

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))    

    native_structure_file=f"{structures_path}/medoid_0.dcd"

    native_traj = md.load(native_structure_file,top=md.Topology.from_openmm(cgmodel.topology))    
    
    positions = native_traj.xyz[0] * unit.nanometer
    
    # Minimize energy of native structure
    positions, PE_start, PE_end, simulation = minimize_structure(
        cgmodel,
        positions,
        output_file=f"{structures_path}/medoid_min.dcd",
    )
    
    assert PE_end < PE_start
    assert os.path.isfile(f"{structures_path}/medoid_min.dcd")    
    

def test_set_binary_interaction():
    """Regression test for adding binary interaction parameter with customNonbondedForce"""

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))    

    binary_interaction_parameters = {
        "bb_sc_binary_interaction": 0.0}
    
    cgmodel_new = CGModel(
        particle_type_list=cgmodel.particle_type_list,
        bond_lengths=cgmodel.bond_lengths,
        bond_force_constants=cgmodel.bond_force_constants,
        bond_angle_force_constants=cgmodel.bond_angle_force_constants,
        torsion_force_constants=cgmodel.torsion_force_constants,
        equil_bond_angles=cgmodel.equil_bond_angles,
        torsion_periodicities=cgmodel.torsion_periodicities,
        binary_interaction_parameters=binary_interaction_parameters,
        include_nonbonded_forces=cgmodel.include_nonbonded_forces,
        include_bond_forces=cgmodel.include_bond_forces,
        include_bond_angle_forces=cgmodel.include_bond_angle_forces,
        include_torsion_forces=cgmodel.include_torsion_forces,
        constrain_bonds=cgmodel.constrain_bonds,
        sequence=cgmodel.sequence,
        positions=cgmodel.positions,
        monomer_types=cgmodel.monomer_types,
    )
    
    native_structure_file=f"{structures_path}/medoid_0.dcd"

    native_traj = md.load(native_structure_file,top=md.Topology.from_openmm(cgmodel.topology))    
    
    positions = native_traj.xyz[0] * unit.nanometer
    
    # Minimize energy of native structure
    positions, PE_start, PE_end, simulation = minimize_structure(
        cgmodel_new,
        positions,
        output_file=f"{structures_path}/medoid_min.dcd",
    )
    
    # These should be equal to ~3 decimal places (1 Joule/mol)
    PE_start_kappa_off = -47.523193359375
    PE_end_kappa_off = -73.21410369873047
    
    PE_start_kappa_on = PE_start.value_in_unit(unit.kilojoule_per_mole)
    PE_end_kappa_on = PE_end.value_in_unit(unit.kilojoule_per_mole)
    
    assert_almost_equal(PE_start_kappa_on,PE_start_kappa_off,decimal=4)
    assert_almost_equal(PE_end_kappa_on,PE_end_kappa_off,decimal=4)
    
    
def test_run_simulation(tmpdir):
    """Run a short MD simulation of a 24mer 1b1s model"""
    
    # Set output directory
    # In pytest we need to use a temp directory
    # tmpdir is a fixture - hence we need to pass it into test function, not import it
    
    output_directory = tmpdir.mkdir("output")
      
    # OpenMM simulation settings
    print_frequency = 10  # Number of steps to skip when printing output
    total_simulation_time = 1.0 * unit.picosecond
    simulation_time_step = 5.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time/simulation_time_step))
    temperature = 200 * unit.kelvin
    friction = 1.0 / unit.picosecond

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False

    # Bond definitions
    bond_length = 1.5 * unit.angstrom
    bond_lengths = {
        "bb_bb_bond_length": bond_length,
        "bb_sc_bond_length": bond_length,
        "sc_sc_bond_length": bond_length,
    }
    bond_force_constant = 1000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    bond_force_constants = {
        "bb_bb_bond_force_constant": bond_force_constant,
        "bb_sc_bond_force_constant": bond_force_constant,
        "sc_sc_bond_force_constant": bond_force_constant,
    }

    # Particle definitions
    mass = 100.0 * unit.amu
    r_min = 1.5 * bond_length  # Lennard-Jones potential r_min
    # Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
    sigma = r_min / (2.0 ** (1.0 / 6.0))
    epsilon = 0.5 * unit.kilojoule_per_mole
    
    bb = {"particle_type_name": "bb", "sigma": sigma, "epsilon": epsilon, "mass": mass}
    sc = {"particle_type_name": "sc", "sigma": sigma, "epsilon": epsilon, "mass": mass}


    # Bond angle definitions
    bond_angle_force_constant = 100 * unit.kilojoule_per_mole / unit.radian / unit.radian
    bond_angle_force_constants = {
        "bb_bb_bb_bond_angle_force_constant": bond_angle_force_constant,
        "bb_bb_sc_bond_angle_force_constant": bond_angle_force_constant,
    }
    # OpenMM requires angle definitions in units of radians
    bb_bb_bb_equil_bond_angle = 120.0 * unit.degrees
    bb_bb_sc_equil_bond_angle = 120.0 * unit.degrees
    equil_bond_angles = {
        "bb_bb_bb_equil_bond_angle": bb_bb_bb_equil_bond_angle,
        "bb_bb_sc_equil_bond_angle": bb_bb_sc_equil_bond_angle,
    }

    # Torsion angle definitions
    torsion_force_constant = 20.0 * unit.kilojoule_per_mole
    torsion_force_constants = {
        "bb_bb_bb_bb_torsion_force_constant": torsion_force_constant,
        "bb_bb_bb_sc_torsion_force_constant": torsion_force_constant
    }

    bb_bb_bb_bb_torsion_phase_angle = 75.0 * unit.degrees
    bb_bb_bb_sc_torsion_phase_angle = 75.0 * unit.degrees

    torsion_phase_angles = {
        "bb_bb_bb_bb_torsion_phase_angle": bb_bb_bb_bb_torsion_phase_angle,
        "bb_bb_bb_sc_torsion_phase_angle": bb_bb_bb_sc_torsion_phase_angle
    }
    torsion_periodicities = {
        "bb_bb_bb_bb_torsion_periodicity": 3,
        "bb_bb_bb_sc_torsion_periodicity": 3}

    # Monomer definitions
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }
    
    sequence = 24 * [A]
    
    pdb_path = os.path.join(structures_path, "24mer_1b1s_initial_structure.pdb")
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
    
    
    run_simulation(
        cgmodel,
        total_simulation_time,
        simulation_time_step,
        temperature,
        friction=friction,
        print_frequency=print_frequency,
        output_directory=output_directory,
    )
    
    assert os.path.isfile(f"{output_directory}/simulation.dat")
    assert os.path.isfile(f"{output_directory}/simulation.pdb")

    
def test_run_replica_exchange(tmpdir):
    """
    Run a short replica exchange MD simulation of a 24mer 1b1s model
    Test replica exchange processing (write pdb files)
    Test heat capacity analysis code
    Test physical validation code
    """

    global_context_cache.platform = openmm.Platform.getPlatformByName("CPU")
    
    # Set output directory
    # In pytest we need to use a temp directory
    # tmpdir is a fixture - hence we need to pass it into test function, not import it
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange simulation settings
    total_simulation_time = 1.0 * unit.picosecond
    simulation_time_step = 5.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time / simulation_time_step))
    output_data = os.path.join(output_directory, "output.nc")
    number_replicas = 4
    min_temp = 200.0 * unit.kelvin
    max_temp = 300.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    exchange_frequency = 10  # Number of steps between exchange attempts
    
    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False

    # Bond definitions
    bond_length = 1.5 * unit.angstrom
    bond_lengths = {
        "bb_bb_bond_length": bond_length,
        "bb_sc_bond_length": bond_length,
        "sc_sc_bond_length": bond_length,
    }
    bond_force_constant = 1000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    bond_force_constants = {
        "bb_bb_bond_force_constant": bond_force_constant,
        "bb_sc_bond_force_constant": bond_force_constant,
        "sc_sc_bond_force_constant": bond_force_constant,
    }

    # Particle definitions
    mass = 100.0 * unit.amu
    r_min = 1.5 * bond_length  # Lennard-Jones potential r_min
    # Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
    sigma = r_min / (2.0 ** (1.0 / 6.0))
    epsilon = 0.5 * unit.kilojoule_per_mole
    
    bb = {"particle_type_name": "bb", "sigma": sigma, "epsilon": epsilon, "mass": mass}
    sc = {"particle_type_name": "sc", "sigma": sigma, "epsilon": epsilon, "mass": mass}


    # Bond angle definitions
    bond_angle_force_constant = 100 * unit.kilojoule_per_mole / unit.radian / unit.radian
    bond_angle_force_constants = {
        "bb_bb_bb_bond_angle_force_constant": bond_angle_force_constant,
        "bb_bb_sc_bond_angle_force_constant": bond_angle_force_constant,
    }
    # OpenMM requires angle definitions in units of radians
    bb_bb_bb_equil_bond_angle = 120.0 * unit.degrees
    bb_bb_sc_equil_bond_angle = 120.0 * unit.degrees
    equil_bond_angles = {
        "bb_bb_bb_equil_bond_angle": bb_bb_bb_equil_bond_angle,
        "bb_bb_sc_equil_bond_angle": bb_bb_sc_equil_bond_angle,
    }

    # Torsion angle definitions
    torsion_force_constant = 20.0 * unit.kilojoule_per_mole
    torsion_force_constants = {
        "bb_bb_bb_bb_torsion_force_constant": torsion_force_constant,
        "bb_bb_bb_sc_torsion_force_constant": torsion_force_constant
    }

    bb_bb_bb_bb_torsion_phase_angle = 75.0 * unit.degrees
    bb_bb_bb_sc_torsion_phase_angle = 75.0 * unit.degrees

    torsion_phase_angles = {
        "bb_bb_bb_bb_torsion_phase_angle": bb_bb_bb_bb_torsion_phase_angle,
        "bb_bb_bb_sc_torsion_phase_angle": bb_bb_bb_sc_torsion_phase_angle
    }
    torsion_periodicities = {
        "bb_bb_bb_bb_torsion_periodicity": 3,
        "bb_bb_bb_sc_torsion_periodicity": 3}

    # Monomer definitions
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }
    
    sequence = 24 * [A]
    
    pdb_path = os.path.join(structures_path, "24mer_1b1s_initial_structure.pdb")
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
    
    run_replica_exchange(
        cgmodel.topology,
        cgmodel.system,
        cgmodel.positions,
        temperature_list=temperature_list,
        simulation_time_step=simulation_time_step,
        total_simulation_time=total_simulation_time,
        exchange_frequency=exchange_frequency,
        output_data=output_data,
    )
    
    assert os.path.isfile(f"{output_directory}/output.nc")
    
    # Process replica exchange output
    # 1) With detect equilibrium and print_timing:
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        detect_equilibration=True,
        plot_production_only=True,
        print_timing=True,
    )
    
    assert production_start is not None
    
    # 2) With detect equilibrium, equil_nskip
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        detect_equilibration=True,
        plot_production_only=True,
        equil_nskip=2,
    )
    
    assert production_start is not None    
    
    
    # 3) Without detect equilibrium:
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        detect_equilibration=False,
    )
    
    assert production_start is None    
    
    # 4) Without writing .dat file:
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        detect_equilibration=False,
        write_data_file=False,
    )
    
    # Test pdb writer:
    make_replica_pdb_files(
        cgmodel.topology,
        output_dir=output_directory,
    )
        
    make_state_pdb_files(
        cgmodel.topology,
        output_dir=output_directory
    )
    
    assert os.path.isfile(f"{output_directory}/replica_4.pdb")
    assert os.path.isfile(f"{output_directory}/state_4.pdb")
    
    # With non-default frame_begin, stride, no centering:
    make_replica_pdb_files(
        cgmodel.topology,
        frame_begin=10,
        frame_stride=2,
        output_dir=output_directory
    )
    
    make_state_pdb_files(
        cgmodel.topology,
        frame_begin=10,
        frame_stride=2,
        output_dir=output_directory,
        center=False
    )
    
    # Test dcd writer:
    make_replica_dcd_files(
        cgmodel.topology,
        timestep=simulation_time_step,
        time_interval=exchange_frequency,
        output_dir=output_directory
    )
        
    make_state_dcd_files(
        cgmodel.topology,
        timestep=simulation_time_step,
        time_interval=exchange_frequency,
        output_dir=output_directory
    )
    
    assert os.path.isfile(f"{output_directory}/replica_4.dcd")
    assert os.path.isfile(f"{output_directory}/state_4.dcd")
    
    # With non-default frame_begin, stride, no centering:
    make_replica_dcd_files(
        cgmodel.topology,
        timestep=simulation_time_step,
        time_interval=exchange_frequency,
        frame_begin=10,
        frame_stride=2,
        output_dir=output_directory
    )
    
    make_state_dcd_files(
        cgmodel.topology,
        timestep=simulation_time_step,
        time_interval=exchange_frequency,
        frame_begin=10,
        frame_stride=2,
        output_dir=output_directory,
        center=False
    )
    
