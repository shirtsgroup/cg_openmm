
import os
import pickle

import numpy as np
import openmm
from cg_openmm.cg_model.cgmodel import CGModel
import mdtraj as md
from openmm import unit
from openmm.app.pdbfile import PDBFile
from cg_openmm.parameters.secondary_structure import get_native_contacts

output_file1 = "stored_cgmodel.pkl"
output_file2 = "stored_cgmodel_per1_3.pkl"
output_file3 = "linear_24mer/stored_cgmodel_24mer_linear.pkl"
output_file4 = "stored_cgmodel_binary_interaction.pkl"
output_file5 = "stored_cgmodel_helix_3sc_open_triangle_8mer.pkl"

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
    
    
def get_test_cmgodel_3sc_open_triangle_Go(output_file):
    # Generate a cgmodel with 1 backbone bead, 3 sidechain bead per residue,
    # in open triangle configuration.    
    
    n_mono = 8
    
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False    
    
    # Set backbone-backbone exclusions to [0,0,1], all others to [1,1,1]
    exclusions = {
        "default_exclusions": [1,1,1],
        }

    for i in range(n_mono):
        for j in range(i+1,n_mono):
            exclusions[f"b{i}_b{j}_exclusions"] = [0,0,1]
        
    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#        
        
    mass = 100.0 * unit.amu
    
    # All residues get the same sigma, epsilon but different particle type names
    particle_type_list = []
    for i in range(n_mono):
        particle_dict_b = {
            "particle_type_name": f"b{i}",
            "sigma": 2.25 * unit.angstrom,
            "epsilon": 1.5 * unit.kilojoules_per_mole,
            "mass": mass
        }
            
        particle_dict_sa = {
            "particle_type_name": f"sa{i}",
            "sigma": 2.007042724 * unit.angstrom,
            "epsilon": 5.0 * unit.kilojoules_per_mole,
            "mass": mass
        }
        
        particle_dict_sb = {
            "particle_type_name": f"sb{i}",
            "sigma": 2.007042724 * unit.angstrom,
            "epsilon": 5.0 * unit.kilojoules_per_mole,
            "mass": mass
        }       

        particle_dict_sc = {
            "particle_type_name": f"sc{i}",
            "sigma": 2.007042724 * unit.angstrom,
            "epsilon": 5.0 * unit.kilojoules_per_mole,
            "mass": mass
        }        
        
        exec(f'b{i} = particle_dict_b')
        exec(f'sa{i} = particle_dict_sa')
        exec(f'sb{i} = particle_dict_sb')
        exec(f'sc{i} = particle_dict_sc')
        
        particle_type_list.append(eval(f'b{i}'))
        particle_type_list.append(eval(f'sa{i}'))
        particle_type_list.append(eval(f'sb{i}'))
        particle_type_list.append(eval(f'sc{i}'))

    # Monomer definition
    sequence = []
    monomer_types = []
    
    for i in range(n_mono):
        monomer_dict = {
            "monomer_name": f"A{i}",
            "particle_sequence": [eval(f"b{i}"),eval(f"sa{i}"),eval(f"sb{i}"),eval(f"sc{i}")],
            "bond_list": [[0,1],[1,2],[2,3]], # open-form triangle
            "start": 0,
            "end": 0,
        }
        
        exec(f'A{i} = monomer_dict')

        sequence.append(eval(f'A{i}'))
        monomer_types.append(eval(f'A{i}'))
    
    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#
    
    # Bond definitions
    # Set the bb-bb types to be the default, and loop over the bb-sc types:
    bond_lengths = {"default_bond_length": 0.250660083 * unit.nanometer}
    
    for i in range(n_mono):
        bond_lengths[f"b{i}_sa{i}_bond_length"] = 0.238918445 * unit.nanometer
        bond_lengths[f"sa{i}_sb{i}_bond_length"] = 0.225282929 * unit.nanometer
        bond_lengths[f"sa{i}_sc{i}_bond_length"] = 0.225282929 * unit.nanometer
        bond_lengths[f"sb{i}_sc{i}_bond_length"] = 0.225282929 * unit.nanometer

    bond_force_constants = {
        "default_bond_force_constant": 15000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#
    
    # Bond angle definitions
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 0 * unit.kilojoule_per_mole / unit.radian / unit.radian,
    }
    for i in range(n_mono-2):
        bond_angle_force_constants[f"b{i}_b{i+1}_b{i+2}_bond_angle_force_constant"] = 150 * unit.kilojoule_per_mole / unit.radian / unit.radian

    # Set the bb-bb-sc bond angles to be the default, loop over the bb-bb-bb types:
    equil_bond_angles = {}
    for i in range(n_mono-2):
        equil_bond_angles[f"b{i}_b{i+1}_b{i+2}_equil_bond_angle"] = 83.03316 * unit.degrees
        
    #-----------------------------#
    # Periodic torsion parameters #
    #-----------------------------#     
    
    # Torsion angle definitions
    torsion_force_constants = {"default_torsion_force_constant": 0.0 * unit.kilojoule_per_mole}
    for i in range(n_mono-3):
        torsion_force_constants[f"b{i}_b{i+1}_b{i+2}_b{i+3}_torsion_force_constant"] = 0 * unit.kilojoule_per_mole

    # Only backbone types get non-zero force constants, so can set a single default for these:
    torsion_phase_angles = {
        "default_torsion_phase_angle": (33.434975-180) * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
    }

    #------------------------------------------#
    # Binary interaction parameters (Go model) #
    #------------------------------------------#

    native_positions_file = "../test_structures/helix_3sc_open_triangle_8mer.pdb"
    positions = PDBFile(native_positions_file).getPositions()

    # First, build a dummy cgmodel with no binary interaction parameters:
    cgmodel_non_go = CGModel(
        particle_type_list=particle_type_list,
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
        monomer_types=monomer_types,
    )    
    
    native_contact_distance_cutoff = 3.15 * unit.angstrom
    
    # Determine all native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel_non_go,
        native_positions_file,
        native_contact_distance_cutoff,
    )        
    
    # Beads per residue:
    n_per_res = 4

    # Binary interaction parameters:
    # (any not set are assigned kappa=0)
    binary_interaction_parameters = {}    
    
    # Get the particle type name lists:
    type_name_list = []
    for i in range(n_mono):
        type_name_list.append('b')
        type_name_list.append('sa')
        type_name_list.append('sb')
        type_name_list.append('sc')
    
    # The beads get ordered [b0,s0,b1,s1,...,b23,s23]
    for i in range(n_mono*n_per_res):
        for j in range(i+1,n_mono*n_per_res):
            if ([i,j] not in native_contact_list and [j,i] not in native_contact_list):
                # If not in native contact list, scale down the interaction by (1-kappa_go)
                
                type_i = type_name_list[i]
                type_j = type_name_list[j]
                
                resid_i = i//n_per_res
                resid_j = j//n_per_res
                
                binary_interaction_parameters[f"{type_i}{resid_i}_{type_j}{resid_j}_binary_interaction"] = 0.99  

    # Build a coarse grained model
    cgmodel = CGModel(
        particle_type_list=particle_type_list,
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
        monomer_types=monomer_types,
    )

    # store the cg model so that we can do various analyses.
    cgmodel.export(output_file)

# Generate the test cgmodels:
cgmodel1 = get_test_cmgodel_1_1_helix(output_file1)
cgmodel2 = get_test_cmgodel_1_1_helix_per1_3(output_file2)
cgmodel3 = get_test_cmgodel_linear(output_file3)
cgmodel4 = get_test_cmgodel_1_1_helix_binary_interaction(output_file4)
cgmodel5 = get_test_cmgodel_3sc_open_triangle_Go(output_file5)