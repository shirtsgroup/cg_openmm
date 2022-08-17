
import os
import pickle

import numpy as np
import openmm
from cg_openmm.cg_model.cgmodel import CGModel
from openmm import unit
from openmm.app.pdbfile import PDBFile

output_file1 = "stored_cgmodel.pkl"
output_file2 = "stored_cgmodel_per1_3.pkl"
output_file3 = "linear_24mer/stored_cgmodel_24mer_linear.pkl"
output_file4 = "stored_cgmodel_binary_interaction.pkl"

def get_test_cmgodel_1_1_helix(output_file):
    # Generate a cgmodel with 1 backbone bead, 1 sidechain bead per residue,
    # using helix-forming force field parameters.

    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False    

    exclusions = {
        "default_exclusions": [0,0,1],
        }

    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#

    mass = 100.0 * unit.amu

    # Mass and charge are defaults.
    # Backbone particle:
    bb = {
        "particle_type_name": "bb",
        "sigma": 2.25 * unit.angstrom,
        "epsilon": 1.50 * unit.kilojoules_per_mole,
        "mass": mass
    }
        
    # Sidechain particle:
    sc = {
        "particle_type_name": "sc",
        "sigma": 3.50 * unit.angstrom,
        "epsilon": 5.00 * unit.kilojoules_per_mole,
        "mass": mass
    }

    # Monomer definition:
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }

    # Residue sequence:
    sequence = 24 * [A]

    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#

    # Bond definitions:
    bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

    bond_force_constants = {
        "default_bond_force_constant": 15000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#

    # Bond angle definitions:
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 150 * unit.kilojoule_per_mole / unit.radian / unit.radian
    }

    equil_bond_angles = {
        "default_equil_bond_angle": 127.5 * unit.degrees,
        "bb_bb_bb_equil_bond_angle": 105.5 * unit.degrees}

    #-----------------------------#
    # Periodic torsion parameters #
    #-----------------------------#

    # Torsion angle definitions:
    torsion_force_constants = {
        "default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole,
        "bb_bb_bb_bb_torsion_force_constant": 5 * unit.kilojoule_per_mole}

    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": (16.7-180) * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
    }

    # Get initial positions from file:
    pdb_path = "../test_structures/24mer_1b1s_initial_structure.pdb"
    positions = PDBFile(pdb_path).getPositions()

    # Build a coarse grained model:
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )

    # Store the cg model so that we can do various analyses.
    cgmodel.export(output_file)
    
    return cgmodel
    
    
def get_test_cmgodel_1_1_helix_per1_3(output_file):
    # Generate a cgmodel with 1 backbone bead, 1 sidechain bead per residue,
    # and multiple backbone periodic torsion terms (periodicities 1 + 3)

    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False    

    exclusions = {
        "default_exclusions": [0,0,1],
        }

    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#

    mass = 100.0 * unit.amu

    # Mass and charge are defaults.
    # Backbone particle:
    bb = {
        "particle_type_name": "bb",
        "sigma": 2.25 * unit.angstrom,
        "epsilon": 1.50 * unit.kilojoules_per_mole,
        "mass": mass
    }
        
    # Sidechain particle:
    sc = {
        "particle_type_name": "sc",
        "sigma": 3.50 * unit.angstrom,
        "epsilon": 5.00 * unit.kilojoules_per_mole,
        "mass": mass
    }

    # Monomer definition:
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }

    # Residue sequence:
    sequence = 24 * [A]

    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#

    # Bond definitions:
    bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

    bond_force_constants = {
        "default_bond_force_constant": 15000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#

    # Bond angle definitions:
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 150 * unit.kilojoule_per_mole / unit.radian / unit.radian
    }

    equil_bond_angles = {
        "default_equil_bond_angle": 127.5 * unit.degrees,
        "bb_bb_bb_equil_bond_angle": 105.5 * unit.degrees}

    #-----------------------------#
    # Periodic torsion parameters #
    #-----------------------------#

    # Torsion angle definitions:
    torsion_force_constants = {
        "default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole,
        "bb_bb_bb_bb_torsion_force_constant": [3,0] * unit.kilojoule_per_mole}

    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": [(16.7-180),0] * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
        "bb_bb_bb_bb_torsion_periodicity": [1,3],
    }

    # Get initial positions from file:
    pdb_path = "../test_structures/24mer_1b1s_initial_structure.pdb"
    positions = PDBFile(pdb_path).getPositions()

    # Build a coarse grained model:
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )

    # Store the cg model so that we can do various analyses.
    cgmodel.export(output_file)    
    
    return cgmodel
    
    
def get_test_cmgodel_linear(output_file):  
    # Get a cgmodel for a linear backbone-only version of the above helical foldamer.

    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False    

    exclusions = {
        "default_exclusions": [0,0,1],
        }

    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#

    mass = 100.0 * unit.amu

    # Mass and charge are defaults.
    # Backbone particle:
    bb = {
        "particle_type_name": "bb",
        "sigma": 2.25 * unit.angstrom,
        "epsilon": 1.50 * unit.kilojoules_per_mole,
        "mass": mass
    }

    # Monomer definition:
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb,bb],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 1,
    }

    # Residue sequence:
    sequence = 12 * [A]

    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#

    # Bond definitions:
    bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

    bond_force_constants = {
        "default_bond_force_constant": 15000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#

    # Bond angle definitions:
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 250 * unit.kilojoule_per_mole / unit.radian / unit.radian
    }

    equil_bond_angles = {
        "default_equil_bond_angle": 127.5 * unit.degrees,
        "bb_bb_bb_equil_bond_angle": 105.5 * unit.degrees}

    #-----------------------------#
    # Periodic torsion parameters #
    #-----------------------------#

    # Torsion angle definitions:
    torsion_force_constants = {
        "default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole,
        "bb_bb_bb_bb_torsion_force_constant": 10 * unit.kilojoule_per_mole}

    torsion_phase_angles = {
        "bb_bb_bb_bb_torsion_phase_angle": (16.7-180) * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
    }
    
    # Get the initial positions:
    pdb_path = "linear_24mer/initial_structure_24mer_linear.pdb"
    positions = PDBFile(pdb_path).getPositions()

    # Rebuild coarse-grained model with torsions on
    cgmodel = CGModel(
        particle_type_list=[bb],
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )    

    # Store the cg model so that we can do various analyses.
    cgmodel.export(output_file)
    
    return cgmodel
    
    
def get_test_cmgodel_1_1_helix_binary_interaction(output_file):
    # Generate a cgmodel with 1 backbone bead, 1 sidechain bead per residue,
    # using helix-forming force field parameters. Include 1-3 nonbonded
    # interactions and use binary interaction parameters.

    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False    

    exclusions = {
        "default_exclusions": [0,1,1],
        }

    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#

    mass = 100.0 * unit.amu

    # Mass and charge are defaults.
    # Backbone particle:
    bb = {
        "particle_type_name": "bb",
        "sigma": 2.25 * unit.angstrom,
        "epsilon": 1.50 * unit.kilojoules_per_mole,
        "mass": mass
    }
        
    # Sidechain particle:
    sc = {
        "particle_type_name": "sc",
        "sigma": 3.50 * unit.angstrom,
        "epsilon": 5.00 * unit.kilojoules_per_mole,
        "mass": mass
    }

    # Monomer definition:
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }

    # Residue sequence:
    sequence = 24 * [A]


    #-------------------------------#
    # Binary interaction parameters #
    #-------------------------------#

    binary_interaction_parameters = {
        "bb_sc_binary_interaction": 0.5
        }

    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#

    # Bond definitions:
    bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

    bond_force_constants = {
        "default_bond_force_constant": 15000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#

    # Bond angle definitions:
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 150 * unit.kilojoule_per_mole / unit.radian / unit.radian
    }

    equil_bond_angles = {
        "default_equil_bond_angle": 127.5 * unit.degrees,
        "bb_bb_bb_equil_bond_angle": 105.5 * unit.degrees}

    #-----------------------------#
    # Periodic torsion parameters #
    #-----------------------------#

    # Torsion angle definitions:
    torsion_force_constants = {
        "default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole,
        "bb_bb_bb_bb_torsion_force_constant": 5 * unit.kilojoule_per_mole}

    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": (16.7-180) * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
    }

    # Get initial positions from file:
    pdb_path = "../test_structures/24mer_1b1s_initial_structure.pdb"
    positions = PDBFile(pdb_path).getPositions()

    # Build a coarse grained model:
    cgmodel = CGModel(
        particle_type_list=[bb, sc],
        bond_lengths=bond_lengths,
        bond_force_constants=bond_force_constants,
        bond_angle_force_constants=bond_angle_force_constants,
        torsion_force_constants=torsion_force_constants,
        equil_bond_angles=equil_bond_angles,
        torsion_phase_angles=torsion_phase_angles,
        torsion_periodicities=torsion_periodicities,
        binary_interaction_parameters=binary_interaction_parameters,
        include_nonbonded_forces=include_nonbonded_forces,
        include_bond_forces=include_bond_forces,
        include_bond_angle_forces=include_bond_angle_forces,
        include_torsion_forces=include_torsion_forces,
        constrain_bonds=constrain_bonds,
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )

    # Store the cg model so that we can do various analyses.
    cgmodel.export(output_file)
    
    return cgmodel    
    

# Generate the test cgmodels:
cgmodel1 = get_test_cmgodel_1_1_helix(output_file1)
cgmodel2 = get_test_cmgodel_1_1_helix_per1_3(output_file2)
cgmodel3 = get_test_cmgodel_linear(output_file3)
cgmodel4 = get_test_cmgodel_1_1_helix_binary_interaction(output_file4)