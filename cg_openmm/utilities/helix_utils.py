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


def get_helix_cgmodel(sigma_bb,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb):
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
        
    xyz_bb = get_helix_coordinates(1,1,t0)
    
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
        positions=positions_init,
        sequence=sequence,
        monomer_types=[A],
    )

    return cgmodel
    

def get_helix_particle_bonded_lists(cgmodel):
    """
    Internal function for getting particle type list and bonded lists
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
    
    
def get_helix_coordinates(r,c,t):
    """
    Internal functon for getting the coordinates of particles along a helix,
    with positions t.
    """
    
    xyz = np.zeros((len(t),3))
    
    xyz[:,0] = r*np.cos(t)
    xyz[:,1] = r*np.sin(t)
    xyz[:,2] = c*t
    
    return xyz
    
    
def get_t_from_bond_distance(r,c,bond_dist_bb):
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
    
        
def plot_LJ_helix(r,c,t_par,r_eq_bb,r_eq_sc=None, plotfile='LJ_helix.pdf'):
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
    xyz_helix = get_helix_coordinates(r,c,np.linspace(0,1.5*max(t_par),num=1001))
    
    # Particle coordinates:
    xyz_par = get_helix_coordinates(r,c,t_par)
    
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
