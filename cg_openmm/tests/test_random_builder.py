"""
Unit and regression test for the cg_openmm package.
"""

import os
import sys

# Import package, test suite, and other packages as needed
import cg_openmm
import mdtraj as md
import numpy as np
import openmm
import pytest
from cg_openmm.build.cg_build import build_topology
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.utilities import random_builder
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from openmm import unit
from openmm.app.pdbfile import PDBFile


def test_random_builder(tmpdir):
    """See if the random builder can build a simple 1b1s model"""
    
    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = False
    constrain_bonds = False
    random_positions = True
    
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
    
    sequence = 5 * [A]    
    # Build a coarse grained model
    cgmodel = CGModel(
        particle_type_list=[bb,sc],
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
        sequence=sequence,
        constrain_bonds=constrain_bonds,
        random_positions=random_positions,
        monomer_types=[A],
    )
    
    output_directory = tmpdir.mkdir("output")
    filename = f"{output_directory}/5mer_1b1s_builder_test.pdb"
    
    write_pdbfile_without_topology(cgmodel, filename)
    
    positions = PDBFile(filename).getPositions()
    
    assert len(positions)==10