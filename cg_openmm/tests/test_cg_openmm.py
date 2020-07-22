"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed
import cg_openmm
import foldamers
from simtk import unit
import pytest
import sys
import os
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from cg_openmm.simulation.tools import run_simulation
import numpy as np

def test_cg_openmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cg_openmm" in sys.modules
    

# Test run simulation    
    # check if simulation.pdb file is created with expected number of frames
    # check if simulation.dat file is created with expected lines of output

    
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_structures')
    
    
def test_run_simulation(tmpdir):
    """Run a short MD simulation of a 24mer 1b1s model"""
    
    # Set output directory
    # In pytest we need to use a temp directory
    # tmpdir is a fixture - hence we need to pass it into test function, not import it
    
    output_directory = tmpdir.mkdir("output")
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
      
    # OpenMM simulation settings
    print_frequency = 10  # Number of steps to skip when printing output
    total_simulation_time = 1.0 * unit.picosecond
    simulation_time_step = 5.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time/simulation_time_step))
    temperature = 187.8 * unit.kelvin
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
        "bb_bb_bond_k": bond_force_constant,
        "bb_sc_bond_k": bond_force_constant,
        "sc_sc_bond_k": bond_force_constant,
    }

    # Particle definitions
    mass = 100.0 * unit.amu
    masses = {"backbone_bead_masses": mass, "sidechain_bead_masses": mass}
    r_min = 1.5 * bond_length  # Lennard-Jones potential r_min
    # Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
    sigma = r_min / (2.0 ** (1.0 / 6.0))
    sigmas = {"bb_sigma": sigma, "sc_sigma": sigma}
    epsilon = 0.5 * unit.kilojoule_per_mole
    epsilons = {"bb_eps": epsilon, "sc_eps": epsilon}

    # Bond angle definitions
    bond_angle_force_constant = 100 * unit.kilojoule_per_mole / unit.radian / unit.radian
    bond_angle_force_constants = {
        "bb_bb_bb_angle_k": bond_angle_force_constant,
        "bb_bb_sc_angle_k": bond_angle_force_constant,
    }
    # OpenMM requires angle definitions in units of radians
    bb_bb_bb_equil_bond_angle = 120.0 * unit.degrees
    bb_bb_sc_equil_bond_angle = 120.0 * unit.degrees
    equil_bond_angles = {
        "bb_bb_bb_angle_0": bb_bb_bb_equil_bond_angle,
        "bb_bb_sc_angle_0": bb_bb_sc_equil_bond_angle,
    }

    # Torsion angle definitions
    torsion_force_constant = 20.0 * unit.kilojoule_per_mole
    torsion_force_constants = {
        "bb_bb_bb_bb_torsion_k": torsion_force_constant,
        "bb_bb_bb_sc_torsion_k": torsion_force_constant
    }

    bb_bb_bb_bb_equil_torsion_angle = 78.0 * unit.degrees
    bb_bb_bb_sc_equil_torsion_angle = 78.0 * unit.degrees

    equil_torsion_angles = {
        "bb_bb_bb_bb_torsion_0": bb_bb_bb_bb_equil_torsion_angle,
        "bb_bb_bb_sc_torsion_0": bb_bb_bb_sc_equil_torsion_angle
    }
    torsion_periodicities = {
        "bb_bb_bb_bb_period": 3,
        "bb_bb_bb_sc_period": 3}

    # Monomer definitions
    A = {
        "monomer_name": "A",
        "backbone_length": 1,
        "sidechain_length": 1,
        "sidechain_positions": [0],
        "bond_lengths": bond_lengths,
        "epsilons": epsilons,
        "sigmas": sigmas,
    }
        
    sequence = 24 * [A]
        
    # Get initial positions from local file
    
    positions = PDBFile(pdb_path).getPositions()

    # Build a coarse grained model
    cgmodel = CGModel(
        masses=masses,
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
        monomer_types=[A],
        sequence=sequence,
        constrain_bonds=constrain_bonds,
        positions=positions,
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

