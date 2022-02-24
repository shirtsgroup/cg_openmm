"""
Unit and regression test for the cg_openmm package.
"""

import os
import pickle
import sys

# Import package, test suite, and other packages as needed
import cg_openmm
import openmm
import pytest
from cg_openmm.cg_model.cgmodel import CGModel
from openmm import unit
from openmm.app.pdbfile import PDBFile

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_structures')
   
@pytest.fixture
def create_cgmodel():
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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
    
    return cgmodel

    
def test_export(tmpdir,create_cgmodel):
    output_directory = tmpdir.mkdir("output")

    cgmodel = create_cgmodel
    cgmodel_path = os.path.join(output_directory, 'saved_cgmodel.pkl')
    cgmodel.export(cgmodel_path)
    
    cgmodel_loaded = pickle.load(open(cgmodel_path, "rb" ))
    
    
def test_get_num_beads(create_cgmodel):
    cgmodel = create_cgmodel
    n_beads = cgmodel.get_num_beads()
    assert n_beads == 48
    

def test_create_particle_list(create_cgmodel):
    cgmodel = create_cgmodel
    particle_list = cgmodel.create_particle_list()
    assert len(particle_list) == 48
    

def test_get_bond_list(create_cgmodel):
    cgmodel = create_cgmodel
    bond_list = cgmodel.get_bond_list()
    assert len(bond_list) == 47
    

def test_get_nonbonded_interaction_list(create_cgmodel):
    cgmodel = create_cgmodel
    nonbond_list = cgmodel.get_nonbonded_interaction_list()
    assert len(nonbond_list) == 1013
    

def test_get_nonbonded_exclusion_list(create_cgmodel):
    cgmodel = create_cgmodel
    nonbond_exclusion_list = cgmodel.get_nonbonded_exclusion_list()
    assert len(nonbond_exclusion_list) == 115
    

def test_get_bond_angle_list(create_cgmodel):
    cgmodel = create_cgmodel
    bond_angle_list = cgmodel.get_bond_angle_list()
    assert len(bond_angle_list) == 68
    

def test_get_torsion_list(create_cgmodel):
    cgmodel = create_cgmodel
    torsion_list = cgmodel.get_torsion_list()
    assert len(torsion_list) == 88
    

def test_get_particle_name(create_cgmodel):
    cgmodel = create_cgmodel
    # By particle index:
    name1 = cgmodel.get_particle_name(1)
    
    # By particle dict:
    dict1 = cgmodel.create_particle_list()[1]
    name1_from_dict = cgmodel.get_particle_name(dict1)
    
    assert name1==name1_from_dict
    assert name1=='sc1'
    

def test_get_particle_index(create_cgmodel):
    cgmodel = create_cgmodel
    dict1 = cgmodel.create_particle_list()[1]
    index1 = cgmodel.get_particle_index(dict1)
    assert index1 == 1
    
    
def test_get_particle_type(create_cgmodel):
    cgmodel = create_cgmodel
    dict1 = cgmodel.create_particle_list()[1]
    type1 = cgmodel.get_particle_type(dict1)
    # type1 is also a dictionary with includes type attributes (mass, charge, etc.)
    
    
def test_get_particle_type_name(create_cgmodel):
    cgmodel = create_cgmodel
    # By particle index:
    type_name1 = cgmodel.get_particle_type_name(1)
    
    # By particle dict:
    dict1 = cgmodel.create_particle_list()[1]
    dict_type_name1 = cgmodel.get_particle_type_name(dict1)
    
    assert type_name1 == dict_type_name1
    assert type_name1 == 'sc'
    
    
def test_get_particle_monomer_type(create_cgmodel):
    # This is for monomer type
    cgmodel = create_cgmodel
    # By particle index:
    mono1 = cgmodel.get_particle_monomer_type(1)
    
    # By particle dict:
    dict1 = cgmodel.create_particle_list()[1]  
    dict_mono1 = cgmodel.get_particle_monomer_type(dict1)
    
    # mono1 is the monomer dictionary
    
    assert mono1 == dict_mono1
    assert mono1['monomer_name'] == 'A'
    
    
def test_get_particle_monomer(create_cgmodel):
    # This is for monomer index
    cgmodel = create_cgmodel
    # By particle index:
    mono1 = cgmodel.get_particle_monomer(1)

    # By particle dict:
    dict1 = cgmodel.create_particle_list()[1]     
    dict_mono1 = cgmodel.get_particle_monomer(dict1)
    
    assert mono1 == dict_mono1
    assert mono1 == 0
    
    
def test_get_particle_mass(create_cgmodel):
    cgmodel = create_cgmodel
    # By particle index:
    mass1 = cgmodel.get_particle_mass(1)
    
    # By particle dict:
    dict1 = cgmodel.create_particle_list()[1]
    mass1_from_dict = cgmodel.get_particle_mass(dict1)
    
    assert mass1==mass1_from_dict
    assert mass1==(100 * unit.amu)
    
    
def test_get_particle_charge(create_cgmodel):
    cgmodel = create_cgmodel
    # By particle index:
    q1 = cgmodel.get_particle_charge(1)
    
    # By particle dict:
    dict1 = cgmodel.create_particle_list()[1]
    q1_from_dict = cgmodel.get_particle_charge(dict1)
    
    assert q1==q1_from_dict
    assert q1==(0 * unit.elementary_charge)
    
    
def test_get_particle_sigma(create_cgmodel):
    cgmodel = create_cgmodel
    # By particle index:
    sig1 = cgmodel.get_particle_sigma(1)
    
    # By particle dict:
    dict1 = cgmodel.create_particle_list()[1]
    sig1_from_dict = cgmodel.get_particle_sigma(dict1)
    
    assert sig1==sig1_from_dict
    assert sig1==(1.5*1.5)/(2.0 ** (1.0 / 6.0)) * unit.angstrom
    
    
def test_get_particle_epsilon(create_cgmodel):
    cgmodel = create_cgmodel
    # By particle index:
    eps1 = cgmodel.get_particle_epsilon(1)
    
    # By particle dict:
    dict1 = cgmodel.create_particle_list()[1]
    eps1_from_dict = cgmodel.get_particle_epsilon(dict1)
    
    assert eps1==eps1_from_dict
    assert eps1== 0.5 * unit.kilojoule_per_mole
    
    
def test_get_bond_length(create_cgmodel):
    cgmodel = create_cgmodel
    bond_list = cgmodel.get_bond_list()
    b0 = cgmodel.get_bond_length([0,1])
    # Particles [0,1], bb-sc
    assert b0 == 1.5 * unit.angstrom
    
    
def test_get_bond_force_constant(create_cgmodel):
    cgmodel = create_cgmodel
    bond_list = cgmodel.get_bond_list()
    k0 = cgmodel.get_bond_force_constant([0,1])
    # Particles [0,1], bb-sc
    assert k0 == 1000 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    
    
def test_get_equil_bond_angle(create_cgmodel):
    cgmodel = create_cgmodel
    angle_list = cgmodel.get_bond_angle_list()   
    # Particles [1,0,2], sc-bb-bb
    theta0 = cgmodel.get_equil_bond_angle([1,0,2])
    assert theta0 == 120.0 * unit.degrees
    
    
def test_get_bond_angle_force_constant(create_cgmodel):
    cgmodel = create_cgmodel
    angle_list = cgmodel.get_bond_angle_list()   
    # Particles [1,0,2], sc-bb-bb
    k0 = cgmodel.get_bond_angle_force_constant([1,0,2])
    assert k0 == 100 * unit.kilojoule_per_mole / unit.radian / unit.radian
    
    
def test_get_torsion_periodicity(create_cgmodel):
    cgmodel = create_cgmodel
    torsion_list = cgmodel.get_torsion_list()
    per0 = cgmodel.get_torsion_periodicity([11,10,8,9])
    # Particles [11,10,8,9], sc-bb-bb-sc
    assert per0 == 1 # default
    
    
def test_get_torsion_force_constant(create_cgmodel):
    cgmodel = create_cgmodel
    torsion_list = cgmodel.get_torsion_list()
    k0 = cgmodel.get_torsion_force_constant([11,10,8,9])
    # Particles [11,10,8,9], sc-bb-bb-sc
    assert k0 == 0 * unit.kilojoule_per_mole # default
    
    
def test_get_torsion_phase_angle(create_cgmodel):
    cgmodel = create_cgmodel
    torsion_list = cgmodel.get_torsion_list()
    phi0 = cgmodel.get_torsion_phase_angle([11,10,8,9])
    # Particles [11,10,8,9], sc-bb-bb-sc
    assert phi0 == 0 * unit.degrees # default
    
    
def test_exclusions_1_1_1():
    # Create a cgmodel with nonbonded forces on all particles

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    exclusions = [1,1,1]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    assert num_exceptions_cgmodel == 0
    
    # Now, check the actual exclusions defined in the OpenMM system
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel   
   
    
def test_exclusions_0_1_1():
    # Create a cgmodel with both angle forces and 1-3 nonbonded interactions

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    exclusions = [0,1,1]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    assert num_exceptions_cgmodel == 47
    
    # Now, check the actual exclusions defined in the OpenMM system
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel
    
    
def test_exclusions_1_0_1():
    # Create a cgmodel with 1-3 pairs excluded, but 1-2 and 1-4 nonbond pairs included

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    exclusions = [1,0,1]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    n_angles = len(cgmodel.get_bond_angle_list())
    
    assert num_exceptions_cgmodel == n_angles
    
    # Now, check the actual exclusions defined in the OpenMM system
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel    
    
    
def test_exclusions_1_1_0():
    # Create a cgmodel with 1-4 pairs excluded, but 1-2 and 1-3 nonbond pairs included

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    exclusions = [1,1,0]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    n_torsions = len(cgmodel.get_torsion_list())
    
    assert num_exceptions_cgmodel == n_torsions
    
    # Now, check the actual exclusions defined in the OpenMM system
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel        
    
    
def test_exclusions_0_0_1():
    # Create a cgmodel with 1-2 and 1-3 nonbonded exclusions (also the default)

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    exclusions = [0,0,1]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    n_bonds = len(cgmodel.get_bond_list())
    n_angles = len(cgmodel.get_bond_angle_list())
    
    assert num_exceptions_cgmodel == n_bonds+n_angles
    
    # Now, check the actual exclusions defined in the OpenMM system
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel    
    
    
def test_exclusions_0_1_0():
    # Create a cgmodel with 1-2 and 1-4 nonbonded exclusions, but 1-3 pairs included

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    exclusions = [0,1,0]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    n_bonds = len(cgmodel.get_bond_list())
    n_torsions = len(cgmodel.get_torsion_list())
    
    assert num_exceptions_cgmodel == n_bonds+n_torsions
    
    # Now, check the actual exclusions defined in the OpenMM system
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel      
    
  
def test_exclusions_1_0_0():
    # Create a cgmodel with 1-3 and 1-4 nonbonded exclusions, but 1-2 pairs included

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    exclusions = [1,0,0]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    n_angles = len(cgmodel.get_bond_angle_list())
    n_torsions = len(cgmodel.get_torsion_list())
    
    assert num_exceptions_cgmodel == n_angles+n_torsions
    
    # Now, check the actual exclusions defined in the OpenMM system
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel       
    
    
def test_exclusions_0_0_0():
    # Create a cgmodel with 1-2, 1-3, and 1-4 nonbonded exclusions
    # (rosetta functional form off)

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    rosetta_functional_form = False
    exclusions = [0,0,0]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        rosetta_functional_form=rosetta_functional_form,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    n_bonds = len(cgmodel.get_bond_list())
    n_angles = len(cgmodel.get_bond_angle_list())
    n_torsions = len(cgmodel.get_torsion_list())
    
    assert num_exceptions_cgmodel == n_bonds+n_angles+n_torsions
    
    # Now, check the actual exclusions defined in the OpenMM system
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel
    

def test_exclusions_0_0_0_rosetta():
    # Create a cgmodel with 1-2, 1-3, and 1-4 nonbonded exclusions
    # (rosetta functional form on, which scales 1-5 nonbonded by 0.2)

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    rosetta_functional_form = True
    exclusions = [0,0,0]

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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        exclusions=exclusions,
        rosetta_functional_form=rosetta_functional_form,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    num_exceptions_cgmodel = len(cgmodel.get_nonbonded_exclusion_list())
    
    # Now, check the actual exclusions defined in the OpenMM system
    # For rosetta_functional_form, this includes 1-5 pairs as well.
    
    # Currently, the cgmodel exceptions don't include the 1-5 scaled
    # terms.
    
    n_15_pairs = 84
    
    for force_index, force in enumerate(cgmodel.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            num_exceptions_openmm = force.getNumExceptions()
            
    assert num_exceptions_openmm == num_exceptions_cgmodel + n_15_pairs
    

def test_sums_periodic_torsions_1():
    # Test cg_model with sums of periodic torsions - test 1
    # Two periodic torsion terms, parameters input as quantities with list values
    # All parameters are set explicitly in lists

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
    torsion_force_constants = {
        "bb_bb_bb_bb_torsion_force_constant": [5,10]*unit.kilojoule_per_mole,
    }

    torsion_phase_angles = {
        "bb_bb_bb_bb_torsion_phase_angle": [0,180]*unit.degrees,
    }
    torsion_periodicities = {
        "bb_bb_bb_bb_torsion_periodicity": [1,3],
    }

    # Monomer definitions
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }
    
    sequence = 24 * [A]
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
    
    # Check the number of periodic torsions terms:
    n_torsion_forces = cgmodel.system.getForces()[3].getNumTorsions()
    assert n_torsion_forces == 109
    
    
def test_sums_periodic_torsions_2():
    # Test cg_model with sums of periodic torsions - test 2
    # Two periodic torsion terms, parameters input as lists of quantities
    # All parameters are set explicitly in lists

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
    torsion_force_constants = {
        "bb_bb_bb_bb_torsion_force_constant": [5*unit.kilojoule_per_mole, 10*unit.kilojoule_per_mole]
    }

    torsion_phase_angles = {
        "bb_bb_bb_bb_torsion_phase_angle": [0*unit.degrees, 180*unit.degrees]
    }
    torsion_periodicities = {
        "bb_bb_bb_bb_torsion_periodicity": [1,3],
    }

    # Monomer definitions
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }
    
    sequence = 24 * [A]
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
    
    # Check the number of periodic torsions terms:
    n_torsion_forces = cgmodel.system.getForces()[3].getNumTorsions()
    assert n_torsion_forces == 109
    
    
def test_sums_periodic_torsions_3():
    # Test cg_model with sums of periodic torsions - test 3
    # Two periodic torsion terms, parameters input as lists of quantities
    # Force constant and phase angle are set implicitly for all periodicities
    # as single element lists of quantities

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
    torsion_force_constants = {
        "bb_bb_bb_bb_torsion_force_constant": [5*unit.kilojoule_per_mole]
    }

    torsion_phase_angles = {
        "bb_bb_bb_bb_torsion_phase_angle": [0*unit.degrees]
    }
    torsion_periodicities = {
        "bb_bb_bb_bb_torsion_periodicity": [1,3],
    }

    # Monomer definitions
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }
    
    sequence = 24 * [A]
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
    
    # Check the number of periodic torsions terms:
    n_torsion_forces = cgmodel.system.getForces()[3].getNumTorsions()
    assert n_torsion_forces == 109


def test_sums_periodic_torsions_4():
    # Test cg_model with sums of periodic torsions - test 4
    # Two periodic torsion terms, parameters input as lists of quantities
    # Force constant and phase angle are set implicitly for all periodicities
    # as single quantities

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
    torsion_force_constants = {
        "bb_bb_bb_bb_torsion_force_constant": 5*unit.kilojoule_per_mole
    }

    torsion_phase_angles = {
        "bb_bb_bb_bb_torsion_phase_angle": 0*unit.degrees
    }
    torsion_periodicities = {
        "bb_bb_bb_bb_torsion_periodicity": [1,3],
    }

    # Monomer definitions
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }
    
    sequence = 24 * [A]
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
    
    # Check the number of periodic torsions terms:
    n_torsion_forces = cgmodel.system.getForces()[3].getNumTorsions()
    assert n_torsion_forces == 109

    
def test_sums_periodic_torsions_5():
    # Test cg_model with sums of periodic torsions - test 5
    # Two periodic torsion terms, parameters input as quantities with list values
    # Parameters are applied to all torsion types using the default input method

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
    torsion_force_constants = {
        "default_torsion_force_constant": [5,10]*unit.kilojoule_per_mole,
    }

    torsion_phase_angles = {
        "default_torsion_phase_angle": [0,180]*unit.degrees,
    }
    torsion_periodicities = {
        "default_torsion_periodicity": [1,3],
    }

    # Monomer definitions
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }
    
    sequence = 24 * [A]
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
    
    # Check the number of periodic torsions terms:
    n_torsion_forces = cgmodel.system.getForces()[3].getNumTorsions()
    assert n_torsion_forces == 176   

    
def test_restricted_angle_potential():
    # Test cg_model with restricted angle potential

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    angle_style='restricted'

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
    bond_angle_force_constant = 50000 * unit.kilojoule_per_mole
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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        angle_style=angle_style,
        constrain_bonds=constrain_bonds,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
    
    
def test_cosine_angle_potential():
    # Test cg_model with cosine angle potential

    # Coarse grained model settings
    include_bond_forces = True
    include_bond_angle_forces = True
    include_nonbonded_forces = True
    include_torsion_forces = True
    constrain_bonds = False
    angle_style='cosine'

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
    bond_angle_force_constant = 50000 * unit.kilojoule_per_mole
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
    
    pdb_path = os.path.join(data_path, "24mer_1b1s_initial_structure.pdb")
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
        angle_style=angle_style,
        constrain_bonds=constrain_bonds,
        positions=positions,
        sequence=sequence,
        monomer_types=[A],
    )
