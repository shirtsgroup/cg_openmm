"""
Unit and regression test for the cg_openmm package.
"""

import os
import pickle
import sys

# Import package, test suite, and other packages as needed
import cg_openmm
import mdtraj as md
import numpy as np
import openmm
import pytest
from cg_openmm.build.cg_build import build_topology
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
from cg_openmm.simulation.tools import *
from cg_openmm.utilities import random_builder
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from numpy.testing import assert_almost_equal
from openmm import unit
from openmm.app.pdbfile import PDBFile


def test_cg_openmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cg_openmm" in sys.modules
    
    
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path,'test_data')
structures_path = os.path.join(current_path,'test_structures')
   
def test_minimize_structure_pdb(tmpdir):
    """Test energy minimization structure, with reading/writing pdb files"""
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))    

    native_structure_file=f"{structures_path}/medoid_0.pdb"

    native_traj = md.load(native_structure_file)    
    
    positions = native_traj.xyz[0] * unit.nanometer
    
    output_directory = tmpdir.mkdir("output")
    
    # Minimize energy of native structure
    positions, PE_start, PE_end, simulation = minimize_structure(
        cgmodel,
        positions,
        output_file=f"{output_directory}/medoid_min.pdb",
    )
    
    assert PE_end < PE_start
    assert os.path.isfile(f"{output_directory}/medoid_min.pdb")

    
def test_minimize_structure_dcd(tmpdir):
    """Test energy minimization structure, with reading/writing dcd files"""

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))    

    native_structure_file=f"{structures_path}/medoid_0.dcd"

    native_traj = md.load(native_structure_file,top=md.Topology.from_openmm(cgmodel.topology))    
    
    positions = native_traj.xyz[0] * unit.nanometer
    
    output_directory = tmpdir.mkdir("output")
    
    # Minimize energy of native structure
    positions, PE_start, PE_end, simulation = minimize_structure(
        cgmodel,
        positions,
        output_file=f"{output_directory}/medoid_min.dcd",
    )

    assert PE_end < PE_start
    assert os.path.isfile(f"{output_directory}/medoid_min.dcd")    

    
def test_run_simulation_pdb(tmpdir):
    """Run a short MD simulation of a 24mer 1b1s model (pdb trajectory output)"""
    
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
        output_traj='simulation.pdb',
        output_directory=output_directory,
    )
    
    assert os.path.isfile(f"{output_directory}/simulation.dat")
    assert os.path.isfile(f"{output_directory}/simulation.pdb")

    
def test_run_simulation_dcd(tmpdir):
    """Run a short MD simulation of a 24mer 1b1s model (dcd trajectory output)"""
    
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
        output_traj='simulation.dcd',
        output_directory=output_directory,
    )
    
    assert os.path.isfile(f"{output_directory}/simulation.dat")
    assert os.path.isfile(f"{output_directory}/simulation.dcd")    


def test_run_replica_exchange(tmpdir):
    """
    Run a short replica exchange MD simulation of a 24mer 1b1s model
    """

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
    
    
def test_process_replica_exchange_1(tmpdir):
    """
    Test replica exchange processing, trajectory writing with various options
    """

    # Set output directory
    # In pytest we need to use a temp directory
    # tmpdir is a fixture - hence we need to pass it into test function, not import it
    
    output_directory = tmpdir.mkdir("output")    
    
    # Location of previously saved output.nc:
    output_data = os.path.join(data_path, "output.nc")
    
    # Process replica exchange output
    # 1) With plot production only and print_timing:
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        plot_production_only=True,
        print_timing=True,
    )

    
def test_process_replica_exchange_2(tmpdir):
    """
    Test replica exchange processing, trajectory writing with various options
    """

    # Set output directory
    output_directory = tmpdir.mkdir("output")    
    
    # Location of previously saved output.nc:
    output_data = os.path.join(data_path, "output.nc")
        
    # 2) With non-default equil_nskip
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        plot_production_only=True,
        equil_nskip=2,
    )
    
    
def test_process_replica_exchange_3(tmpdir):
    """
    Test replica exchange processing, trajectory writing with various options
    """

    # Set output directory
    output_directory = tmpdir.mkdir("output")    
    
    # Location of previously saved output.nc:
    output_data = os.path.join(data_path, "output.nc")
    
    # 3) With frame_begin used to circumvent detectEquilibration
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        frame_begin=5,
    )   
    
    
def test_process_replica_exchange_4(tmpdir):
    """
    Test replica exchange processing, trajectory writing with various options
    """

    # Set output directory
    output_directory = tmpdir.mkdir("output")    
    
    # Location of previously saved output.nc:
    output_data = os.path.join(data_path, "output.nc")    
    
    # 4) With frame end specified to analyze only the beginning of a trajectory
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        frame_end=25,
    )   
    
    
def test_process_replica_exchange_5(tmpdir):
    """
    Test replica exchange processing, trajectory writing with various options
    """

    # Set output directory
    output_directory = tmpdir.mkdir("output")    
    
    # Location of previously saved output.nc:
    output_data = os.path.join(data_path, "output.nc")    
    
    # 5) Without writing .dat file:
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        write_data_file=False,
    )
    
    
def test_write_state_pdb_traj(tmpdir):
    """
    Test state trajectory writer (pdb):
    """
    
    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))   
    
    make_state_pdb_files(
        cgmodel.topology,
        output_dir=output_directory,
        output_data=output_data,
        center=True,
    )
    
    assert os.path.isfile(f"{output_directory}/state_4.pdb")


def test_write_state_pdb_traj_no_center(tmpdir):
    """
    Test state trajectory writer (pdb):
    """
    
    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))   

    # With non-default frame_begin, stride, no centering:
    make_state_pdb_files(
        cgmodel.topology,
        frame_begin=10,
        frame_stride=2,
        output_dir=output_directory,
        output_data=output_data,
        center=False,
    )
    
    assert os.path.isfile(f"{output_directory}/state_4.pdb")
    

def test_write_state_dcd_traj(tmpdir):
    """
    Test state trajectory writer (dcd):
    """   

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))   

    # For dcd, we also need to set the time intervals:
    exchange_frequency = 10
    timestep = 5 * unit.femtosecond

    make_state_dcd_files(
        cgmodel.topology,
        timestep=timestep,
        time_interval=exchange_frequency,
        output_dir=output_directory,
        output_data=output_data,
        center=True,
    )
    
    assert os.path.isfile(f"{output_directory}/state_4.dcd")
    
    
def test_write_state_dcd_traj_no_center(tmpdir):
    """
    Test state trajectory writer (dcd):
    """   

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))   
    
    # For dcd, we also need to set the time intervals:
    exchange_frequency = 10
    timestep = 5 * unit.femtosecond    
    
    # With non-default frame_begin, stride, no centering:
    make_state_dcd_files(
        cgmodel.topology,
        timestep=timestep,
        time_interval=exchange_frequency,
        frame_begin=10,
        frame_stride=2,
        output_dir=output_directory,
        output_data=output_data,
        center=False
    )
       
    assert os.path.isfile(f"{output_directory}/state_4.dcd")       
    

def test_write_replica_pdb_traj(tmpdir):
    """
    Test replica trajectory writer (pdb):
    """
    
    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))       
    
    make_replica_pdb_files(
        cgmodel.topology,
        output_dir=output_directory,
        output_data=output_data,
        center=True,
    )    
    
    assert os.path.isfile(f"{output_directory}/replica_4.pdb")
    

def test_write_replica_pdb_traj_no_center(tmpdir):
    """
    Test replica trajectory writer (pdb):
    """
    
    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))        
    
    # With non-default frame_begin, stride, no centering:
    make_replica_pdb_files(
        cgmodel.topology,
        frame_begin=10,
        frame_stride=2,
        output_dir=output_directory,
        output_data=output_data,
        center=False,
    )
    
    assert os.path.isfile(f"{output_directory}/replica_4.pdb")
    

def test_write_replica_dcd_traj(tmpdir):
    """
    Test replica trajectory writer (dcd):
    """    

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))   

    # For dcd, we also need to set the time intervals:
    exchange_frequency = 10
    timestep = 5 * unit.femtosecond

    # Test dcd writer:
    make_replica_dcd_files(
        cgmodel.topology,
        timestep=timestep,
        time_interval=exchange_frequency,
        output_dir=output_directory,
        output_data=output_data,
        center=True,
    )
        
    assert os.path.isfile(f"{output_directory}/replica_4.dcd")
    

def test_write_replica_dcd_traj_no_center(tmpdir):
    """
    Test replica trajectory writer (dcd):
    """    

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # For dcd, we also need to set the time intervals:
    exchange_frequency = 10
    timestep = 5 * unit.femtosecond    
    
    # With non-default frame_begin, stride, no centering:
    make_replica_dcd_files(
        cgmodel.topology,
        timestep=timestep,
        time_interval=exchange_frequency,
        frame_begin=10,
        frame_stride=2,
        output_dir=output_directory,
        output_data=output_data,
        center=False,
    )
    
    assert os.path.isfile(f"{output_directory}/replica_4.dcd")


def test_restart_replica_exchange(tmpdir):
    """
    Restart a replica exchange MD simulation of a 24mer 1b1s model
    Test replica exchange processing (write pdb files)
    Test heat capacity analysis code
    Test physical validation code
    """
    
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
    
    exchange_attempts = int(np.floor(total_steps/exchange_frequency))
    
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
    
    # Now, restart the simulation to double the total duration:
    
    restart_replica_exchange(
        2*total_simulation_time,
        simulation_time_step=simulation_time_step,
        exchange_frequency=exchange_frequency,
        output_data=output_data,
    )
    
    # Check that the output files are the correct length:
    replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
    )
    
    assert replica_energies.shape[2] == (exchange_attempts*2 + 1)
    