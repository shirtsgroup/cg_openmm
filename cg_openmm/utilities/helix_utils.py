import os

import matplotlib.pyplot as plt
import numpy as np
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from openmm import unit
from openmm.app.pdbfile import PDBFile
from scipy.optimize import differential_evolution, root_scalar


def dist_unitless(positions_1, positions_2):
    # Distance function:
    return np.sqrt(np.sum(np.power((positions_1 - positions_2),2)))


def get_extended_positions_1sc(r_bb, r_bs, n_particle_bb, theta_bbb):
    """
    Internal function for computing the positions of an extended 1sc model, given bond lengths, backbone angle,
    and number of backbone particles. Assumes particle ordering of bb,sc,bb,sc...
    """
    
    positions_ext = np.zeros((2*n_particle_bb,3))
    
    # Distance between 1-3 neighbor backbone beads:
    theta_bbb_rad = theta_bbb*np.pi/180
    c = 2*r_bb*np.sin(theta_bbb_rad/2)
    
    # Height of triangle defining the b-b-b angle:
    h = r_bb*np.cos(theta_bbb_rad/2)
    
    # TODO: allow for odd numbers of backbone beads:
    j = 0
    for i in range(int(n_particle_bb/2)):
        # First backbone
        positions_ext[j,0] = i*c
        positions_ext[j,1] = 0
        
        # First sidechain
        positions_ext[j+1,0] = i*c
        positions_ext[j+1,1] = -r_bs
        
        # Second backbone
        positions_ext[j+2,0] = i*c + c/2
        positions_ext[j+2,1] = h
        
        # Second sidechain
        positions_ext[j+3,0] = i*c + c/2
        positions_ext[j+3,1] = h+r_bs
        j += 4
        
    return positions_ext        
    
def rotate_coordinates_x(xyz, theta):
    """
    Internal function for rotating 3d coordinates by theta (angle to rotate the triangle in-plane)
    Theta is specified in radians.
    xyz is formatted as (particle, dimension)
    """
    
    R_mat = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)],
        ])

    xyz_rot = np.matmul(R_mat, xyz)
    
    return xyz_rot
    
    
def rotate_coordinates_y(xyz, theta):
    """
    Internal function for rotating 3d coordinates by theta (angle to rotate the triangle in-plane)
    Theta is specified in radians.
    """
    
    R_mat = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
        ])

    xyz_rot = np.matmul(R_mat, xyz)
    
    return xyz_rot    


def rotate_coordinates_z(xyz, theta):
    """
    Internal function for rotating 3d coordinates by theta (angle between adjacent residues)
    Theta is specified in radians.
    """
    
    R_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
        ])

    xyz_rot = np.matmul(R_mat, xyz)    
    
    return xyz_rot


def get_helix_cgmodel(sigma_bb, sigma_sc, epsilon_bb, epsilon_sc, n_particle_bb, exclusions_in):
    """
    Internal function for creating a 1-1 cgmodel to use in helical optimization
    """

    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#

    mass = 100.0 * unit.amu

    # Mass and charge are defaults.
    # Backbone particle:
    bb = {
        "particle_type_name": "bb",
        "sigma": sigma_bb, # angstrom
        "epsilon": epsilon_bb, # kilojoules_per_mole
        "mass": mass
    }
        
    # Sidechain particle:
    sc = {
        "particle_type_name": "sc",
        "sigma": sigma_sc, # angstrom,
        "epsilon": epsilon_sc, # kilojoules_per_mole,
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
    sequence = n_particle_bb * [A]

    # Exclusion rules:
    # If an empty set, apply the default exclusion scheme
    
    if exclusions_in:
        # Apply specified rules:
        exclusions = exclusions_in
    else:
        # Use default scheme
        exclusions = {
            "default_exclusions": [0,0,1],
            }

    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#

    # Bond definitions:
    bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

    bond_force_constants = {
        "default_bond_force_constant": 0 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#

    # Bond angle definitions:
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 0 * unit.kilojoule_per_mole / unit.radian / unit.radian
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
    }

    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": (16.7-180) * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
    }

    # Set initial positions for cgmodel
    # These are arbitrary - we just can't have overlapping beads
    # The particles in cgmodel get ordered as bb,sc,bb,sc...
    positions_init = np.zeros(([2*n_particle_bb,3]))
    
    t0 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t0[i] = i*np.pi/5
        
    xyz_bb = get_helix_backbone_coordinates(1,1,t0)
    
    xyz_sc = np.zeros((n_particle_bb,3))
    
    xyz_sc[:,0] = 2*xyz_bb[:,0]
    xyz_sc[:,1] = 2*xyz_bb[:,1]
    xyz_sc[:,2] = xyz_bb[:,2]
    
    positions_init[0::2] = xyz_bb
    positions_init[1::2] = xyz_sc
    positions_init *= unit.angstrom
    
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
        include_nonbonded_forces=True,
        include_bond_forces=True,
        include_bond_angle_forces=True,
        include_torsion_forces=True,
        constrain_bonds=False,
        exclusions=exclusions,
        positions=positions_init,
        sequence=sequence,
        monomer_types=[A],
    )

    return cgmodel
    
    
def get_helix_cgmodel_2sc_equal(sigma_bb, sigma_sc, epsilon_bb, epsilon_sc, n_particle_bb, exclusions_in):
    """
    Internal function for creating a cgmodel with 1 backbone bead, 2 sidechain beads
    and with 1-3 nonbonded interactions included for bb-sc and sc-sc pairs,
    to use in helical optimization.
    (equal sigma_sc for each sidechain particle)
    """

    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#

    mass = 100.0 * unit.amu

    # Mass and charge are defaults.
    # Backbone particle:
    bb = {
        "particle_type_name": "bb",
        "sigma": sigma_bb, # angstrom
        "epsilon": epsilon_bb, # kilojoules_per_mole
        "mass": mass
    }
        
    # Sidechain particle:
    sc = {
        "particle_type_name": "sc",
        "sigma": sigma_sc, # angstrom,
        "epsilon": epsilon_sc, # kilojoules_per_mole,
        "mass": mass
    }

    # Monomer definition:
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc, sc],
        "bond_list": [[0, 1],[1,2]],
        "start": 0,
        "end": 0,
    }

    # Residue sequence:
    sequence = n_particle_bb * [A]

    # Exclusion rules:
    # If an empty set, apply the default exclusion scheme for 2sc models
    
    if exclusions_in:
        # Apply specified rules:
        exclusions = exclusions_in
    else:
        # Use default scheme
        exclusions = {
            "default_exclusions": [0,1,1],
            "bb_bb_exclusions": [0,0,1],
            }

    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#

    # Bond definitions:
    bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

    bond_force_constants = {
        "default_bond_force_constant": 0 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#

    # Bond angle definitions:
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 0 * unit.kilojoule_per_mole / unit.radian / unit.radian
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
    }

    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": (16.7-180) * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
    }

    # Set initial positions for cgmodel
    # These are arbitrary - we just can't have overlapping beads
    # The particles in cgmodel get ordered as bb,sc,sc,bb,sc,sc...
    positions_init = np.zeros((3*n_particle_bb,3))
    
    t0 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t0[i] = i*np.pi/5
        
    xyz_bb = get_helix_backbone_coordinates(1,1,t0)
    
    j = -1
    for i in range(n_particle_bb):
        j += 1
        positions_init[j,:] = xyz_bb[i,:]
        
        j += 1
        positions_init[j,0] = 2*xyz_bb[i,0]
        positions_init[j,1] = 2*xyz_bb[i,1]
        positions_init[j,2] = xyz_bb[i,2]
        
        j += 1
        positions_init[j,0] = 3*xyz_bb[i,0]
        positions_init[j,1] = 3*xyz_bb[i,1]
        positions_init[j,2] = xyz_bb[i,2]

    positions_init *= unit.angstrom
    
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
        include_nonbonded_forces=True,
        include_bond_forces=True,
        include_bond_angle_forces=True,
        include_torsion_forces=False,
        constrain_bonds=False,
        exclusions=exclusions,
        positions=positions_init,
        sequence=sequence,
        monomer_types=[A],
    )

    return cgmodel        
    
    
def get_helix_cgmodel_2sc_nonequal(sigma_bb, sigma_sc, epsilon_bb, epsilon_sc, n_particle_bb, exclusions_in):
    """
    Internal function for creating a cgmodel with 1 backbone bead, 2 sidechain beads
    and with 1-3 nonbonded interactions included for bb-sc and sc-sc pairs,
    to use in helical optimization.
    (different sigma_sc for each sidechain particle)
    """

    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#

    mass = 100.0 * unit.amu

    # Mass and charge are defaults.
    # Backbone particle:
    bb = {
        "particle_type_name": "bb",
        "sigma": sigma_bb, # angstrom
        "epsilon": epsilon_bb, # kilojoules_per_mole
        "mass": mass
    }
        
    # Sidechain particle:
    # Initially, set the sidechain sigmas equal
    sc1 = {
        "particle_type_name": "sc1",
        "sigma": sigma_sc, # angstrom,
        "epsilon": epsilon_sc, # kilojoules_per_mole,
        "mass": mass
    }
    
    # Sidechain particle:
    sc2 = {
        "particle_type_name": "sc2",
        "sigma": sigma_sc, # angstrom,
        "epsilon": epsilon_sc, # kilojoules_per_mole,
        "mass": mass
    }    

    # Monomer definition:
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc1, sc2],
        "bond_list": [[0, 1],[1,2]],
        "start": 0,
        "end": 0,
    }

    # Residue sequence:
    sequence = n_particle_bb * [A]

    # Exclusion rules:
    # If an empty set, apply the default exclusion scheme for 2sc models
    
    if exclusions_in:
        # Apply specified rules:
        exclusions = exclusions_in
    else:
        # Use default scheme
        exclusions = {
            "default_exclusions": [0,1,1],
            "bb_bb_exclusions": [0,0,1],
            }

    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#

    # Bond definitions:
    bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

    bond_force_constants = {
        "default_bond_force_constant": 0 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#

    # Bond angle definitions:
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 0 * unit.kilojoule_per_mole / unit.radian / unit.radian
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
    }

    torsion_phase_angles = {
        "default_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": (16.7-180) * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
    }

    # Set initial positions for cgmodel
    # These are arbitrary - we just can't have overlapping beads
    # The particles in cgmodel get ordered as bb,sc,sc,bb,sc,sc...
    positions_init = np.zeros((3*n_particle_bb,3))
    
    t0 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t0[i] = i*np.pi/5
        
    xyz_bb = get_helix_backbone_coordinates(1,1,t0)
    
    j = -1
    for i in range(n_particle_bb):
        j += 1
        positions_init[j,:] = xyz_bb[i,:]
        
        j += 1
        positions_init[j,0] = 2*xyz_bb[i,0]
        positions_init[j,1] = 2*xyz_bb[i,1]
        positions_init[j,2] = xyz_bb[i,2]
        
        j += 1
        positions_init[j,0] = 3*xyz_bb[i,0]
        positions_init[j,1] = 3*xyz_bb[i,1]
        positions_init[j,2] = xyz_bb[i,2]

    positions_init *= unit.angstrom
    
    # Build a coarse grained model:
    cgmodel = CGModel(
        particle_type_list=[bb, sc1, sc2],
        bond_lengths=bond_lengths,
        bond_force_constants=bond_force_constants,
        bond_angle_force_constants=bond_angle_force_constants,
        torsion_force_constants=torsion_force_constants,
        equil_bond_angles=equil_bond_angles,
        torsion_phase_angles=torsion_phase_angles,
        torsion_periodicities=torsion_periodicities,
        include_nonbonded_forces=True,
        include_bond_forces=True,
        include_bond_angle_forces=True,
        include_torsion_forces=False,
        constrain_bonds=False,
        exclusions=exclusions,
        positions=positions_init,
        sequence=sequence,
        monomer_types=[A],
    )

    return cgmodel            
    
    
def get_helix_cgmodel_triangle(sigma_bb, sigma_sc, epsilon_bb, epsilon_sc, n_particle_bb, exclusions_in):
    """
    Internal function for creating a cgmodel with 1 backbone bead, 3 sidechain beads in a 
    triangle with 1-3 nonbonded interactions included for bb-sc and sc-sc pairs,
    to use in helical optimization.
    """

    #--------------------------------------------#
    # Particle definitions and oligomer topology #
    #--------------------------------------------#

    mass = 100.0 * unit.amu

    # Mass and charge are defaults.
    # Backbone particle:
    bb = {
        "particle_type_name": "bb",
        "sigma": sigma_bb, # angstrom
        "epsilon": epsilon_bb, # kilojoules_per_mole
        "mass": mass
    }
        
    # Sidechain particle:
    sc = {
        "particle_type_name": "sc",
        "sigma": sigma_sc, # angstrom,
        "epsilon": epsilon_sc, # kilojoules_per_mole,
        "mass": mass
    }

    # Monomer definition:
    A = {
        "monomer_name": "A",
        "particle_sequence": [bb, sc, sc, sc],
        "bond_list": [[0, 1],[1,2],[2,3]],
        "start": 0,
        "end": 0,
    }

    # Residue sequence:
    sequence = n_particle_bb * [A]

    # Exclusion rules:
    # If an empty set, apply the default exclusion scheme for 3sc models
    
    if exclusions_in:
        # Apply specified rules:
        exclusions = exclusions_in
    else:
        # Use default scheme
        exclusions = {
            "default_exclusions": [0,1,1],
            "bb_bb_exclusions": [0,0,1],
            }

    #--------------------------#
    # Harmonic bond parameters #
    #--------------------------#

    # Bond definitions:
    bond_lengths = {"default_bond_length": 2.44 * unit.angstrom}

    bond_force_constants = {
        "default_bond_force_constant": 0 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
    }

    #---------------------------#
    # Harmonic angle parameters #
    #---------------------------#

    # Bond angle definitions:
    bond_angle_force_constants = {
        "default_bond_angle_force_constant": 0 * unit.kilojoule_per_mole / unit.radian / unit.radian
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
    }

    torsion_phase_angles = {
        "sc_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
        "bb_bb_bb_bb_torsion_phase_angle": (16.7-180) * unit.degrees,
        "bb_bb_bb_sc_torsion_phase_angle": 0 * unit.degrees,
    }

    torsion_periodicities = {
        "default_torsion_periodicity": 1,
    }

    # Set initial positions for cgmodel
    # These are arbitrary - we just can't have overlapping beads
    # The particles in cgmodel get ordered as bb,sc,sc,sc,bb,sc,sc,sc...
    positions_init = np.zeros((4*n_particle_bb,3))
    
    t0 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t0[i] = i*np.pi/5
        
    xyz_bb = get_helix_backbone_coordinates(1,1,t0)
    
    j = -1
    for i in range(n_particle_bb):
        j += 1
        positions_init[j,:] = xyz_bb[i,:]
        
        j += 1
        positions_init[j,0] = 2*xyz_bb[i,0]
        positions_init[j,1] = 2*xyz_bb[i,1]
        positions_init[j,2] = xyz_bb[i,2]
        
        j += 1
        positions_init[j,0] = 3*xyz_bb[i,0]
        positions_init[j,1] = 3*xyz_bb[i,1]
        positions_init[j,2] = xyz_bb[i,2]

        j += 1
        positions_init[j,0] = 4*xyz_bb[i,0]
        positions_init[j,1] = 4*xyz_bb[i,1]
        positions_init[j,2] = xyz_bb[i,2]

    positions_init *= unit.angstrom
    
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
        include_nonbonded_forces=True,
        include_bond_forces=True,
        include_bond_angle_forces=True,
        include_torsion_forces=False,
        constrain_bonds=False,
        exclusions=exclusions,
        positions=positions_init,
        sequence=sequence,
        monomer_types=[A],
    )

    return cgmodel    
    

def get_helix_particle_bonded_lists(cgmodel):
    """
    Internal function for getting particle type list and bonded lists
    (1sc model)
    """

    particle_list = cgmodel.create_particle_list()
    
    particle_type_list = []
    
    bb_list = []
    sc_list = []
    
    # Use particle indices rather than the full particle dictionary:
    for par in range(len(particle_list)):
        if cgmodel.get_particle_type_name(par) == 'bb':
            bb_list.append(par)
            particle_type_list.append('bb')
        elif cgmodel.get_particle_type_name(par) == 'sc':
            sc_list.append(par)
            particle_type_list.append('sc')
    
    # Use arrays for particle type indices
    bb_array = np.asarray(bb_list)
    sc_array = np.asarray(sc_list)
    
    # Create bond list for each type:
    bond_list = cgmodel.get_bond_list()
        
    bb_bond_list = []
    sc_bond_list = []
    
    for bond in bond_list:
        type1 = cgmodel.get_particle_type_name(bond[0])
        type2 = cgmodel.get_particle_type_name(bond[1])
        
        if type1 == 'bb' and type2 == 'bb':
            bb_bond_list.append(bond)
        else:
            sc_bond_list.append(bond)
            
    # Create angle list for each type:
    angle_list = cgmodel.get_bond_angle_list()
    
    b_angle_list = []
    s_angle_list = []
    
    for angle in angle_list:
        type1 = cgmodel.get_particle_type_name(angle[0])
        type2 = cgmodel.get_particle_type_name(angle[1])
        type3 = cgmodel.get_particle_type_name(angle[2])
        
        if type1 == 'bb' and type2 == 'bb' and type3 == 'bb':
            b_angle_list.append(angle)
        else:
            s_angle_list.append(angle)
    
    # Create torsion list for each type:
    torsion_list = cgmodel.get_torsion_list()
    
    bbbb_torsion_list = []
    bbbs_torsion_list = []
    sbbs_torsion_list = []
    
    for torsion in torsion_list:
        type1 = cgmodel.get_particle_type_name(torsion[0])
        type2 = cgmodel.get_particle_type_name(torsion[1])
        type3 = cgmodel.get_particle_type_name(torsion[2])
        type4 = cgmodel.get_particle_type_name(torsion[3])
        
        if type1 == 'bb' and type4 == 'bb':
            bbbb_torsion_list.append(torsion)
        elif type1 == 'sc' and type4 == 'sc':
            sbbs_torsion_list.append(torsion)
        else:
            bbbs_torsion_list.append(torsion)    
    
    return (particle_type_list, bb_array, sc_array,
            bb_bond_list, sc_bond_list,
            b_angle_list, s_angle_list,
            bbbb_torsion_list, bbbs_torsion_list, sbbs_torsion_list)
    
    
def get_helix_particle_bonded_lists_2sc(cgmodel):
    """
    Internal function for getting particle type list and bonded lists
    (2sc model)
    """

    particle_list = cgmodel.create_particle_list()
    
    particle_type_list = []
    
    bb_list = []
    sc_list = []
    
    # Use particle indices rather than the full particle dictionary:
    for par in range(len(particle_list)):
        if cgmodel.get_particle_type_name(par) == 'bb':
            bb_list.append(par)
            particle_type_list.append('bb')
        elif cgmodel.get_particle_type_name(par) == 'sc':
            sc_list.append(par)
            particle_type_list.append('sc')
        elif cgmodel.get_particle_type_name(par) == 'sc1':
            sc_list.append(par)
            particle_type_list.append('sc1')   
        elif cgmodel.get_particle_type_name(par) == 'sc2':
            sc_list.append(par)
            particle_type_list.append('sc2')              
    
    # Use arrays for particle type indices
    bb_array = np.asarray(bb_list)
    sc_array = np.asarray(sc_list)
    
    # Create bond list for each type:
    bond_list = cgmodel.get_bond_list()
        
    bb_bond_list = []
    bs_bond_list = []
    ss_bond_list = []
    
    for bond in bond_list:
        type1 = cgmodel.get_particle_type_name(bond[0])
        type2 = cgmodel.get_particle_type_name(bond[1])
        
        if type1 == 'bb' and type2 == 'bb':
            bb_bond_list.append(bond)
        elif (type1 == 'bb' and type2 in ['sc','sc1','sc2']) or \
            (type2 == 'bb' and type1 in ['sc','sc1','sc2']):
            bs_bond_list.append(bond)
        elif type1 in ['sc','sc1','sc2'] and type2 in ['sc','sc1','sc2']:
            ss_bond_list.append(bond)
            
    # Create angle list for each type:
    angle_list = cgmodel.get_bond_angle_list()
    
    bbb_angle_list = []
    bbs_angle_list = []
    bss_angle_list = []
    
    for angle in angle_list:
        type1 = cgmodel.get_particle_type_name(angle[0])
        type2 = cgmodel.get_particle_type_name(angle[1])
        type3 = cgmodel.get_particle_type_name(angle[2])
        
        if type1 == 'bb' and type2 == 'bb' and type3 == 'bb':
            bbb_angle_list.append(angle)
            
        elif (type1 == 'bb' and type2 == 'bb' and type3 in ['sc','sc1','sc2']) or \
            (type1 in ['sc','sc1','sc2'] and type2 == 'bb' and type3 == 'bb'):
            bbs_angle_list.append(angle)
            
        elif (type1 == 'bb' and type2 in ['sc','sc1','sc2'] and type3 in ['sc','sc1','sc2']) or \
            (type1 in ['sc','sc1','sc2'] and type2 in ['sc','sc1','sc2'] and type3 == 'bb'):
            bss_angle_list.append(angle)
    
    # Create torsion list for each type:
    torsion_list = cgmodel.get_torsion_list()
    
    bbbb_torsion_list = []
    bbbs_torsion_list = []
    bbss_torsion_list = []
    sbbs_torsion_list = []
    
    for torsion in torsion_list:
        type1 = cgmodel.get_particle_type_name(torsion[0])
        type2 = cgmodel.get_particle_type_name(torsion[1])
        type3 = cgmodel.get_particle_type_name(torsion[2])
        type4 = cgmodel.get_particle_type_name(torsion[3])
        
        if type1 == 'bb' and type2 == 'bb' and type3 == 'bb' and type4 == 'bb':
            bbbb_torsion_list.append(torsion)
            
        elif (type1 == 'bb' and type2 == 'bb' and type3 == 'bb' and type4 in ['sc','sc1','sc2']) or \
            (type1 in ['sc','sc1','sc2'] and type2 == 'bb' and type3 == 'bb' and type4 == 'bb'):
            bbbs_torsion_list.append(torsion)
          
        elif (type1 == 'bb' and type2 == 'bb' and type3 in ['sc','sc1','sc2'] and type4 in ['sc','sc1','sc2']) or \
            (type1 in ['sc','sc1','sc2'] and type2 in ['sc','sc1','sc2'] and type3 == 'bb' and type4 == 'bb'):
            bbss_torsion_list.append(torsion)        
            
        elif type1 in ['sc','sc1','sc2'] and type2 == 'bb' and type3 == 'bb' and type4 in ['sc','sc1','sc2']:
            sbbs_torsion_list.append(torsion)
    
    return (particle_type_list, bb_array, sc_array,
            bb_bond_list, bs_bond_list, ss_bond_list,
            bbb_angle_list, bbs_angle_list, bss_angle_list,
            bbbb_torsion_list, bbbs_torsion_list, bbss_torsion_list,
            sbbs_torsion_list)        
    
    
def get_helix_particle_bonded_lists_triangle(cgmodel):
    """
    Internal function for getting particle type list and bonded lists (3sc triangle model)
    """

    particle_list = cgmodel.create_particle_list()
    
    particle_type_list = []
    
    bb_list = []
    sc_list = []
    
    # Use particle indices rather than the full particle dictionary:
    for par in range(len(particle_list)):
        if cgmodel.get_particle_type_name(par) == 'bb':
            bb_list.append(par)
            particle_type_list.append('bb')
        elif cgmodel.get_particle_type_name(par) == 'sc':
            sc_list.append(par)
            particle_type_list.append('sc')
    
    # Use arrays for particle type indices
    bb_array = np.asarray(bb_list)
    sc_array = np.asarray(sc_list)
    
    # Create bond list for each type:
    bond_list = cgmodel.get_bond_list()
        
    bb_bond_list = []
    bs_bond_list = []
    ss_bond_list = []
    
    for bond in bond_list:
        type1 = cgmodel.get_particle_type_name(bond[0])
        type2 = cgmodel.get_particle_type_name(bond[1])
        
        if type1 == 'bb' and type2 == 'bb':
            bb_bond_list.append(bond)
        elif type1 != type2:
            bs_bond_list.append(bond)
        elif type1 == 'sc' and type2 == 'sc':
            ss_bond_list.append(bond)
            
    # Create angle list for each type:
    angle_list = cgmodel.get_bond_angle_list()
    
    bbb_angle_list = []
    bbs_angle_list = []
    bss_angle_list = []
    sss_angle_list = []
    
    for angle in angle_list:
        type1 = cgmodel.get_particle_type_name(angle[0])
        type2 = cgmodel.get_particle_type_name(angle[1])
        type3 = cgmodel.get_particle_type_name(angle[2])
        
        if type1 == 'bb' and type2 == 'bb' and type3 == 'bb':
            bbb_angle_list.append(angle)
            
        elif (type1 == 'bb' and type2 == 'bb' and type3 == 'sc') or \
            (type1 == 'sc' and type2 == 'bb' and type3 == 'bb'):
            bbs_angle_list.append(angle)
            
        elif (type1 == 'bb' and type2 == 'sc' and type3 == 'sc') or \
            (type1 == 'sc' and type2 == 'sc' and type3 == 'bb'):
            bss_angle_list.append(angle)             
            
        elif type1 == 'sc' and type2 == 'sc' and type3 == 'sc':
            sss_angle_list.append(angle)
    
    # Create torsion list for each type:
    torsion_list = cgmodel.get_torsion_list()
    
    bbbb_torsion_list = []
    bbbs_torsion_list = []
    bbss_torsion_list = []
    bsss_torsion_list = []
    sbbs_torsion_list = []
    
    for torsion in torsion_list:
        type1 = cgmodel.get_particle_type_name(torsion[0])
        type2 = cgmodel.get_particle_type_name(torsion[1])
        type3 = cgmodel.get_particle_type_name(torsion[2])
        type4 = cgmodel.get_particle_type_name(torsion[3])
        
        if type1 == 'bb' and type2 == 'bb' and type3 == 'bb' and type4 == 'bb':
            bbbb_torsion_list.append(torsion)
            
        elif (type1 == 'bb' and type2 == 'bb' and type3 == 'bb' and type4 == 'sc') or \
            (type1 == 'sc' and type2 == 'bb' and type3 == 'bb' and type4 == 'bb'):
            bbbs_torsion_list.append(torsion)
          
        elif (type1 == 'bb' and type2 == 'bb' and type3 == 'sc' and type4 == 'sc') or \
            (type1 == 'sc' and type2 == 'sc' and type3 == 'bb' and type4 == 'bb'):
            bbss_torsion_list.append(torsion)
          
        elif (type1 == 'bb' and type2 == 'sc' and type3 == 'sc' and type4 == 'sc') or \
            (type1 == 'sc' and type2 == 'sc' and type3 == 'sc' and type4 == 'bb'):
            bsss_torsion_list.append(torsion)            
            
        elif type1 == 'sc' and type2 == 'bb' and type3 == 'bb' and type4 == 'sc':
            sbbs_torsion_list.append(torsion)
    
    return (particle_type_list, bb_array, sc_array,
            bb_bond_list, bs_bond_list, ss_bond_list,
            bbb_angle_list, bbs_angle_list, bss_angle_list, sss_angle_list,
            bbbb_torsion_list, bbbs_torsion_list, bbss_torsion_list,
            bsss_torsion_list, sbbs_torsion_list)    
    
    
def get_helix_backbone_coordinates(r, c, t):
    """
    Internal functon for getting the backbone coordinates of particles along a helix,
    with positions t.
    """
    
    xyz = np.zeros((len(t),3))
    
    xyz[:,0] = r*np.cos(t)
    xyz[:,1] = r*np.sin(t)
    xyz[:,2] = c*t
    
    return xyz
    
    
def get_helix_coordinates_2sc_rotation(r, c, t, r_bs, r_ss, r_eq_bb_sc2, theta, alignment):
    """
    Internal functon for getting the full coordinates of particles along a helix
    (2sc model with bent bb-sc1-sc2 angle)
    
    :param r: helical radius, in Angstrom
    :type r: float
    
    :param c: helical rise parameter, in Angstrom
    :type c: float
    
    :param t: backbone particle spacing, in radians
    :type t: float
    
    :param r_bs: backbone-sidechain bond length, in Angstrom
    :type r_bs: float
    
    :param r_ss: sidechain-sidechain bond length, in Angstrom
    :type r_ss: float
    
    :param r_eq_bb_sc2: equilibrium contact distance for backbone and 2nd sidechain bead, in Angstrom
    :type r_eq_bb_sc2: float
    
    :param theta: angle(s) of rotation for sc2 about bb-sc1 axis, in radians
    :type theta: list(float) 
    
    :param alignment: sidechain alignment scheme - can be 'center' (center of sidechain group is fixed normal to backbone) or 'first' (first bead is normal to backbone)
    :type alignment: str    
    
    :returns:
      - positions - np.array( n_particles_bb*3 x 3 )
    
    """

    n_particle_bb = len(t)
    
    # Get backbone coordinates
    xyz_backbone = get_helix_backbone_coordinates(r,c,t)
    
    ref_orient = {}
    # Type 1 rotation angle:
    ref_orient[1] = np.zeros((3,3))

    # Backbone positions in first residue:
    ref_orient[1][0,:] = xyz_backbone[0,:]
    
    if alignment == 'first':
        # Sidechain bead 1 positions in first residue, normal to backbone:
        ref_orient[1][1,0] = (1+r_bs/r)*xyz_backbone[0,0]
        ref_orient[1][1,1] = (1+r_bs/r)*xyz_backbone[0,1]
        ref_orient[1][1,2] = xyz_backbone[0,2]

        # Place second sidechain particle at optimal local rotation:
        # This theta_sbs tilt angle could also be set as an optimization variable for more degrees of freedom
        if (r_eq_bb_sc2**2 + r_bs**2 - r_ss**2)/(2*r_eq_bb_sc2*r_bs) > 1:
            print(f'arccos(x), x > 1')
            print(f'r_eq_bb_sc2: {r_eq_bb_sc2}')
            print(f'r_bs: {r_bs}')
            print(f'r_ss: {r_ss}')
            print(f'num: {(r_eq_bb_sc2**2 + r_bs**2 - r_ss**2)}')
            print(f'den: {(2*r_eq_bb_sc2*r_bs)}')
            
        elif (r_eq_bb_sc2**2 + r_bs**2 - r_ss**2)/(2*r_eq_bb_sc2*r_bs) < -1:
            print(f'arccos(x), x < -1')
            print(f'r_eq_bb_sc2: {r_eq_bb_sc2}')
            print(f'r_bs: {r_bs}')
            print(f'r_ss: {r_ss}')
            print(f'num: {(r_eq_bb_sc2**2 + r_bs**2 - r_ss**2)}')
            print(f'den: {(2*r_eq_bb_sc2*r_bs)}')
        
        theta_sbs = np.arccos((r_eq_bb_sc2**2 + r_bs**2 - r_ss**2)/(2*r_eq_bb_sc2*r_bs))
    
        # Shift in x from the first backbone particle:
        # This is constant for a tilt angle theta_sbs
        # xshift should always be positive
        xshift = np.abs(r_eq_bb_sc2*np.cos(theta_sbs))
        
        # Shift in y from the first backbone particle: 
        # This needs to be rotated about the y axis by theta:
        yshift = r_eq_bb_sc2*np.sin(theta_sbs)
        
        ref_orient[1][2,0] = xyz_backbone[0,0] + xshift
        ref_orient[1][2,1] = xyz_backbone[0,1] + yshift
        ref_orient[1][2,2] = xyz_backbone[0,2]
        
    elif alignment == 'center':
        # Sidechain group center is normal to backbone:

        # xshift should always be positive and real
        xshift = np.sqrt(r_bs**2 - (r_ss**2)/4)
        yshift = r_ss/2
        
        # sc1 particle:
        ref_orient[1][1,0] = xyz_backbone[0,0] + xshift
        ref_orient[1][1,1] = xyz_backbone[0,1] - yshift
        ref_orient[1][1,2] = xyz_backbone[0,2]
        
        # sc2 particle:
        ref_orient[1][2,0] = xyz_backbone[0,0] + xshift
        ref_orient[1][2,1] = xyz_backbone[0,1] + yshift
        ref_orient[1][2,2] = xyz_backbone[0,2]

    # Here we can have any number of rotation angles:
    n_rotation_angles = len(theta)
    for a in range(1,n_rotation_angles):
        ref_orient[a+1] = ref_orient[1]
    
    # Apply the rotation about x axis - this only changes the positions of sidechain bead 2:
    # The matrix rows here should be x, y, z and columns the particle indices

    ref_orient_rotx = {}
    ref_orient_rotx[1] = np.transpose(rotate_coordinates_x(np.transpose(ref_orient[1]),theta[0]))
    
    for a in range(1,n_rotation_angles):  
        ref_orient_rotx[a+1] = np.transpose(rotate_coordinates_x(np.transpose(ref_orient[1]),theta[a])) 
    
    # Now, rotate the template residue about the z axis to construct the full helix:
    positions = np.zeros((3*n_particle_bb,3))
    
    # distance between backbone beads projected onto a circle:
    dist_bb_xy = np.sqrt(np.sum(np.power((xyz_backbone[0,0:2]-xyz_backbone[1,0:2]),2)))
    
    # helical angle between the two backbone beads projected onto a circle:
    theta_arc = np.arccos(1-dist_bb_xy**2/(2*(r**2))) # radians

    # Vertical rise from one residue to the next:
    z_rise = xyz_backbone[1,2] - xyz_backbone[0,2]    
    
    # Now, assign each residue to a rotation angle index
    rotation_ids = []
    
    i = 0
    while i < n_particle_bb:
        for a in range(n_rotation_angles):
            rotation_ids.append(a)
            i += 1
        
    rotation_ids = rotation_ids[0:n_particle_bb]    
        
    # Now, do the rotation about z axis to construct the helix:
    j = -1
    for i in range(n_particle_bb):
        ref_rotx_curr = eval(f'ref_orient_rotx[rotation_ids[i]+1]')

        ref_orient_rotx_rotz_curr = np.transpose(
            rotate_coordinates_z(np.transpose(ref_rotx_curr),theta_arc*i))
        
        # The z coordinate must be also shifted by the rise/residue
        
        # Backbone
        j += 1
        positions[j,:] = ref_orient_rotx_rotz_curr[0,:]
        positions[j,2] = ref_orient_rotx_rotz_curr[0,2] + z_rise*i
        
        # Sidechain 1
        j += 1
        positions[j,:] = ref_orient_rotx_rotz_curr[1,:]
        positions[j,2] = ref_orient_rotx_rotz_curr[1,2] + z_rise*i
        
        # Sidechain 2
        j += 1
        positions[j,:] = ref_orient_rotx_rotz_curr[2,:]
        positions[j,2] = ref_orient_rotx_rotz_curr[2,2] + z_rise*i
                
    positions *= unit.angstrom

    return positions

    
def get_helix_coordinates_3sc_triangle(r, c, t, r_bs, r_ss, r_eq_bb_sc, theta1, theta2, alignment):
    """
    Internal functon for getting the full coordinates of particles along a helix
    (2sc model with bent bb-sc1-sc2 angle)
    
    :param r: helical radius, in Angstrom
    :type r: float
    
    :param c: helical rise parameter, in Angstrom
    :type c: float
    
    :param t: backbone particle spacing, in radians
    :type t: float
    
    :param r_bs: backbone-sidechain bond length, in Angstrom
    :type r_bs: float
    
    :param r_ss: sidechain-sidechain bond length, in Angstrom
    :type r_ss: float
    
    :param r_eq_bb_sc: equilibrium contact distance for backbone and non-bonded sidechain bead, in Angstrom
    :type r_eq_bb_sc: float    
    
    :param theta1: angle of rotation for sc2 about bb-sc1 axis (even residues), in radians
    :type theta1: float
    
    :param theta2: angle of rotation for sc2 about bb-sc1 axis (odd residues), in radians
    :type theta2: float    
    
    :param alignment: sidechain alignment scheme - can be 'center' (center of sidechain group is fixed normal to backbone) or 'first' (first bead is normal to backbone)
    :type alignment: str    
    
    :returns:
      - positions - np.array( n_particles_bb*3 x 3 )
    
    """    
    n_particle_bb = len(t)
    
    # Get backbone coordinates
    xyz_backbone = get_helix_backbone_coordinates(r,c,t)
    
    ref_orient1 = np.zeros((4,3))

    # Backbone positions in first residue:
    ref_orient1[0,:] = xyz_backbone[0,:]
    
    if alignment == 'first':
        # Sidechain bead 1 positions in first residue, normal to backbone:
        ref_orient1[1,0] = (1+r_bs/r)*xyz_backbone[0,0]
        ref_orient1[1,1] = (1+r_bs/r)*xyz_backbone[0,1]
        ref_orient1[1,2] = xyz_backbone[0,2]
        
        # Due to shift in z for sidechains 2,3, need the distance between backbone
        # and center of line connecting sc2-sc3:
        
        d_bb_sc23 = np.sqrt(r_eq_bb_sc**2-(r_ss**2)/4)
        
        # r_ss distance projected onto the xy plane:
        r_ss_xy = r_ss*np.sqrt(3)/2
    
        if (d_bb_sc23**2 + r_bs**2 - r_ss_xy**2)/(2*d_bb_sc23*r_bs) > 1:
            print(f'arccos(x), x > 1')
            print(f'd_bb_sc23: {d_bb_sc23}')
            print(f'r_bs: {r_bs}')
            print(f'r_ss: {r_ss}')
            print(f'r_ss_xy: {r_ss_xy}')
            print(f'num: {(d_bb_sc23**2 + r_bs**2 - r_ss_xy**2)}')
            print(f'den: {(2*d_bb_sc23*r_bs)}')
            
        elif (d_bb_sc23**2 + r_bs**2 - r_ss_xy**2)/(2*d_bb_sc23*r_bs) < -1:
            print(f'arccos(x), x < -1')
            print(f'd_bb_sc23: {d_bb_sc23}')
            print(f'r_bs: {r_bs}')
            print(f'r_ss: {r_ss}')
            print(f'r_ss_xy: {r_ss_xy}')
            print(f'num: {(d_bb_sc23**2 + r_bs**2 - r_ss_xy**2)}')
            print(f'den: {(2*d_bb_sc23*r_bs)}')
            
        # Angle in xy plane for sidechain-backbone-sidechain        
        # theta_sbs = np.arccos((d_bb_sc23**2 + r_bs**2 - r_ss_xy**2)/(2*d_bb_sc23*r_bs))
    
        # Shift in x from the first backbone particle:
        # This is constant for a tilt angle theta_sbs
        # xshift should always be positive
        xshift = np.sqrt(3/4*r_ss**2-1/4*r_ss**4/r_bs**2)
        
        # Shift in y from the first backbone particle: 
        # This needs to be rotated about the y axis by theta:
        yshift = r_bs - 1/2*r_ss**2/r_bs
        
        # For equilateral triangle model, also shift in z by r_ss/2:
        zshift = r_ss/2
        
        # sc2 particle:
        ref_orient1[2,0] = xyz_backbone[0,0] + xshift
        ref_orient1[2,1] = xyz_backbone[0,1] + yshift
        ref_orient1[2,2] = xyz_backbone[0,2] + zshift
        
        # sc3 particle:
        ref_orient1[3,0] = xyz_backbone[0,0] + xshift
        ref_orient1[3,1] = xyz_backbone[0,1] + yshift
        ref_orient1[3,2] = xyz_backbone[0,2] - zshift
        
        
    elif alignment == 'center':
        # Sidechain group center is normal to backbone:
        
        K = np.sqrt(r_bs**2 - (r_ss**2)/4)  # Distance from backbone bead to center of triangle plane
        L = np.sqrt((r_ss**2)/3)            # Distance from triangle center to lower bead
        M = np.sqrt(L**2 - (r_ss**2)/4)     # Distance from triangle center to top of triangle

        # sc1 particle:
        ref_orient1[1,0] = (1+K/r)*xyz_backbone[0,0]
        ref_orient1[1,1] = (1+K/r)*xyz_backbone[0,1] - r_ss/2
        ref_orient1[1,2] = xyz_backbone[0,2] + M
        
        # sc2 particle:
        ref_orient1[2,0] = (1+K/r)*xyz_backbone[0,0] 
        ref_orient1[2,1] = (1+K/r)*xyz_backbone[0,1] + r_ss/2
        ref_orient1[2,2] = xyz_backbone[0,2] + M

        # sc3 particle:
        ref_orient1[3,0] = (1+K/r)*xyz_backbone[0,0]
        ref_orient1[3,1] = (1+K/r)*xyz_backbone[0,1]
        ref_orient1[3,2] = xyz_backbone[0,2] - L  

    ref_orient2 = ref_orient1
    
    # Apply the rotation about x axis - this only changes the positions of sidechain bead 2:
    # The matrix rows here should be x, y, z and columns the particle indices

    ref_orient1_rotx = np.transpose(rotate_coordinates_x(np.transpose(ref_orient1),theta1))
    ref_orient2_rotx = np.transpose(rotate_coordinates_x(np.transpose(ref_orient2),theta2)) 
    
    # Now, rotate the template residue about the z axis to construct the full helix:
    positions = np.zeros((4*n_particle_bb,3))
    
    # distance between backbone beads projected onto a circle:
    dist_bb_xy = np.sqrt(np.sum(np.power((xyz_backbone[0,0:2]-xyz_backbone[1,0:2]),2)))
    
    # helical angle between the two backbone beads projected onto a circle:
    theta_arc = np.arccos(1-dist_bb_xy**2/(2*(r**2))) # radians

    # Vertical rise from one residue to the next:
    z_rise = xyz_backbone[1,2] - xyz_backbone[0,2]    
    
    j = -1
    for i in range(n_particle_bb):
        if i % 2 == 0:
            # Orientation 1
            ref_orient1_rotx_rotz = np.transpose(rotate_coordinates_z(np.transpose(ref_orient1_rotx),theta_arc*i))
            
            # The z coordinate must be also shifted by the rise/residue
            
            # Backbone
            j += 1
            positions[j,:] = ref_orient1_rotx_rotz[0,:]
            positions[j,2] = ref_orient1_rotx_rotz[0,2] + z_rise*i
            
            # Sidechain 1
            j += 1
            positions[j,:] = ref_orient1_rotx_rotz[1,:]
            positions[j,2] = ref_orient1_rotx_rotz[1,2] + z_rise*i
            
            # Sidechain 2
            j += 1
            positions[j,:] = ref_orient1_rotx_rotz[2,:]
            positions[j,2] = ref_orient1_rotx_rotz[2,2] + z_rise*i
            
            # Sidechain 3
            j += 1
            positions[j,:] = ref_orient1_rotx_rotz[3,:]
            positions[j,2] = ref_orient1_rotx_rotz[3,2] + z_rise*i            
            
        else:
            # Orientation 2
            ref_orient2_rotx_rotz = np.transpose(rotate_coordinates_z(np.transpose(ref_orient2_rotx),theta_arc*i))
            
            # The z coordinate must be also shifted by the rise/residue
            
            # Backbone
            j += 1
            positions[j,:] = ref_orient2_rotx_rotz[0,:]
            positions[j,2] = ref_orient2_rotx_rotz[0,2] + z_rise*i
            
            # Sidechain 1
            j += 1
            positions[j,:] = ref_orient2_rotx_rotz[1,:]
            positions[j,2] = ref_orient2_rotx_rotz[1,2] + z_rise*i
            
            # Sidechain 2
            j += 1
            positions[j,:] = ref_orient2_rotx_rotz[2,:]
            positions[j,2] = ref_orient2_rotx_rotz[2,2] + z_rise*i
            
            # Sidechain 3
            j += 1
            positions[j,:] = ref_orient2_rotx_rotz[3,:]
            positions[j,2] = ref_orient2_rotx_rotz[3,2] + z_rise*i              
                
    positions *= unit.angstrom

    return positions
    
def get_t_from_bond_distance(r, c, bond_dist_bb):
    """
    Internal function for calculating the change in t (from helix parameter equation)
    that corresponds to a backbone bond distance.
    """
    
    # If t1 = 0, then t2=delta_t:
    
    # x1 = r*np.cos(t1) = r
    # x2 = r*np.cos(t2)
    
    # y1 = r*np.sin(t1) = 0
    # y2 = r*np.sin(t2)
    
    # z1 = c*t1 = 0
    # z2 = c*t2
    
    def helical_distance(t):
        return np.sqrt((r*np.cos(t)-r)**2 + (r*np.sin(t))**2 + (c*t)**2) - bond_dist_bb
    
    # Find the positive root
    sol_t = root_scalar(helical_distance,x0=0.1, x1=0.5)
    delta_t = sol_t.root
    
    return delta_t
    
        
def plot_LJ_helix(r, c, t_par, r_eq_bb, r_eq_sc=None, plotfile='LJ_helix.pdf'):
    """
    Internal function for plotting LJ particles on a helix
    
    :param r: radius of helix
    :type r: float
    
    :param c: vertical rise parameter
    :type c: float
    
    :param t_par: particle coordinates, in radians 
    :type t_par: np.array([float])
    
    :param r_eq_bb: equilibrium LJ12-6 distance for backbone particles
    :type r_eq_bb: float
    
    :param r_eq_sc: equilibrium LJ12-6 distance for sidechain particles (default=None)
    :type r_eq_sc: float
    
    :param plotfile: path to file for saving plot (default='LJ_helix.pdf')
    :type plotfile: str
    """
    
    # TODO: add sidechain particles to the 3d plot
    
    # Helix coordinates:
    xyz_helix = get_helix_backbone_coordinates(r,c,np.linspace(0,1.5*max(t_par),num=1001))
    
    # Particle coordinates:
    xyz_par = get_helix_backbone_coordinates(r,c,t_par)
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot helix
    ax.plot(xyz_helix[:,0],xyz_helix[:,1], xyz_helix[:,2],
        color='b', linewidth=2)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    light = LightSource()
    
    # Plot spheres
    for i in range(len(t_par)):
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
        x_sphere = np.cos(u)*np.sin(v)*r_eq_bb/2+xyz_par[i,0]
        y_sphere = np.sin(u)*np.sin(v)*r_eq_bb/2+xyz_par[i,1]
        z_sphere = np.cos(v)*r_eq_bb/2+xyz_par[i,2]
        ax.plot_surface(x_sphere, y_sphere, z_sphere,
            color="r", shade=True, lightsource=light, alpha=0.3)  
     
    # This doesn't work for 3d axes currently.
    #ax.set_aspect('equal')    
        
    plt.savefig(plotfile)
    plt.close() 
    
    return
    
    
def write_helix_pdbfile(coordinates, filename, sidechain):
    """
    Internal function for writing pdb file of optimized helix
    (used for the simple helix optimization only)
    """

    pdb_object = open(filename, "w")
    particle_list = np.arange(0,coordinates.shape[0],1)

    particle_type_list = [] # Type of each particle
    monomer_index_list = [] # Residue number of each bead

    if sidechain:
        mono_id = 0
        for i in range(int(len(particle_list)/2)):
            particle_type_list.append('bb')
            monomer_index_list.append(mono_id)
            mono_id +=1

        mono_id = 0
        for i in range(int(len(particle_list)/2)):
            particle_type_list.append('sc')
            monomer_index_list.append(mono_id)
            mono_id +=1
    else:
        mono_id = 0
        for i in range(len(particle_list)):
            particle_type_list.append('bb')   
            monomer_index_list.append(mono_id)
            mono_id +=1            

    for i in range(len(particle_list)):
        pdb_object.write(
            f"ATOM{particle_list[i]+1:>7d} {particle_type_list[i]:>3s}{1}   A A{monomer_index_list[i]:>4}    "
            f"{coordinates[i][0]:>8.3f}"
            f"{coordinates[i][1]:>8.3f}"
            f"{coordinates[i][2]:>8.3f}"
            f"  1.00  0.00\n"
        )
    pdb_object.write("TER\n")

    # Add bonds:
    bond_list = []

    if sidechain:
        for i in range(int(len(particle_list)/2)-1):
            bond_list.append([i,i+1])
        for i in range(int(len(particle_list)/2)):
            bond_list.append([i,int(i+len(particle_list)/2)])
    else:
        for i in range(len(particle_list)-1):
            bond_list.append([i,i+1])    

    for bond in bond_list:
        pdb_object.write(
            "CONECT"
            + str("{:>5}".format(bond[0] + 1))
            + str("{:>5}".format(bond[1] + 1))
            + "\n"
        )
    pdb_object.write(str("END\n"))

    pdb_object.close()
    return    
    