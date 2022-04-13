import os

import mdtraj as md
import numpy as np
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.utilities.helix_utils import *
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from openmm import LangevinIntegrator, unit
from openmm.app import Simulation
from openmm.app.pdbfile import PDBFile
from scipy.optimize import differential_evolution, root_scalar


def optimize_helix_LJ_parameters(radius, pitch, n_particle_bb,
    bond_dist_bb=None, bond_dist_sc=None,DE_popsize=50,
    pdbfile='LJ_helix_openmm_energy.pdb', plotfile='LJ_helix_openmm_energy.pdf'):
    """
    Optimize backbone and sidechain particle parameters along a helix with specified radius and
    pitch, with equal spacing of backbone particles and sidechain beads normal to the helix.
    
    :param radius: fixed helical radius
    :type radius: Quantity
    
    :param pitch: fixed helical pitch (c*2*pi)
    :type pitch: Quantity
    
    :param n_particle_bb: Number of backbone particles to model
    :type n_particle_bb: int
    
    :param bond_dist_bb: bond distance for bb-bb bonds. If None, bond distance will also be optimized.
    :type bond_dist_bb: Quantity
    
    :param bond_dist_sc: bond distance for bb-sc bonds. If None, bond distance will also be optimized.
    :type bond_dist_sc: Quantity

    :param DE_popsize: population size to use in SciPy differential_evolution solver (default=50)
    :type DE_popsize: int

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix_openmm_energy.pdb')
    :type pdbfile: str
    
    :param plotfile: Path to pdf file for plotting the helical equations and particle positions (default='LJ_helix_openmm_energy.pdf')
    :type plotfile: str
    
    :returns:
      - opt_sol - Results from scipy.optimize (dict)
      - geometry - Dictionary containing key geometric parameters of the optimized helix
    """
    
    r_unit = radius.unit
    # Use angstrom for writing pdb file:
    radius = radius.value_in_unit(unit.angstrom)
    pitch = pitch.value_in_unit(unit.angstrom)
    
    r = radius
    c = pitch/(2*np.pi) # Helical rise parameter
    
    # t_delta is related to the specified bond distance - this must be computed at each iteration
    
    # Here we need to create a cgmodel
    
    # Set initial epsilon parameters
    epsilon_bb = 1.0 * unit.kilojoule_per_mole
    epsilon_sc = 1.0 * unit.kilojoule_per_mole
    
    # Set initial sigma parameters
    sigma_bb = 1.0 * unit.angstrom
    sigma_sc = 1.0 * unit.angstrom
    
    cgmodel = get_helix_cgmodel(sigma_bb,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb)
    
    # Get particle type lists and bonded lists:
    (particle_type_list, bb_array, sc_array, bb_bond_list, sc_bond_list,
    b_angle_list, s_angle_list, bbbb_torsion_list, bbbs_torsion_list,
    sbbs_torsion_list) = get_helix_particle_bonded_lists(cgmodel)
    
    # Set up Simulation object beforehand:
    simulation_time_step = 5.0 * unit.femtosecond
    friction = 0.0 / unit.picosecond
    integrator = LangevinIntegrator(
        0.0 * unit.kelvin, friction, simulation_time_step.in_units_of(unit.picosecond)
    )
    simulation = Simulation(cgmodel.topology, cgmodel.system, integrator)    
    
    if bond_dist_bb is None and bond_dist_sc is None:
    
        #-----------------------------#
        # Unconstrained bonds version #
        #-----------------------------#
    
        # Set sidechain bonds to the backbone bond distance for now
        # Vary t spacing, sigma_bb, sigma_sc

        params = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb)
    
        # Set optimization bounds [t, sigma_bb, sigma_sc]:
        # Use a minimium of 3 residues/turn
        bounds = [(0.01,2*np.pi/3),(r/50,15*r),(r/50,15*r)]

        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        t_delta_opt = opt_sol.x[0]
        sigma_bb_opt = opt_sol.x[1]
        sigma_sc_opt = opt_sol.x[2]
           
        t_par = np.zeros(n_particle_bb)
        for i in range(n_particle_bb):
            t_par[i] = i*t_delta_opt
            
    else:
    
        #---------------------------#
        # Constrained bonds version #
        #---------------------------#
    
        # For now, we have to specify both bb-bb and bb-sc bond distances
        
        bond_dist_bb = bond_dist_bb.value_in_unit(unit.angstrom)
        bond_dist_sc = bond_dist_sc.value_in_unit(unit.angstrom)
    
        params = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, bond_dist_bb, bond_dist_sc)
    
        # Set optimization bounds [sigma_bb, sigma_sc]:
        bounds = [(r/50,15*r),(r/50,15*r)]   
        
        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ_constrained,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        sigma_bb_opt = opt_sol.x[0]
        sigma_sc_opt = opt_sol.x[1]
           
        # Compute particle spacing based on bond constraints
        t_delta_opt = get_t_from_bond_distance(r,c,bond_dist_bb)
        if t_delta_opt < 0:
            print(t_delta_opt)
            t_delta_opt *= -1
        
        t_par = np.zeros(n_particle_bb)
        for i in range(n_particle_bb):
            t_par[i] = i*t_delta_opt    
        
        
    # Equilibrium LJ distance (for visual representation)
    r_eq_bb = sigma_bb_opt*np.power(2,(1/6))
    r_eq_sc = sigma_sc_opt*np.power(2,(1/6))
    
    # Get particle positions:
    xyz = get_helix_coordinates(r,c,t_par)
    
    # Place sidechain particles normal to helix
    if bond_dist_sc == None:
        # Use optimized bond length from first two backbone beads:
        r_bs = dist_unitless(xyz[0,:],xyz[1,:])
    else:
        # Use specified bb-sc bond distance:
        r_bs = bond_dist_sc
    
    side_xyz = np.zeros((n_particle_bb,3))

    side_xyz[:,0] = (1+r_bs/r)*xyz[:,0]
    side_xyz[:,1] = (1+r_bs/r)*xyz[:,1]
    side_xyz[:,2] = xyz[:,2]
    
    # Now, set the backbone and sidechain positions to the correct bead indices:
    positions = np.zeros((2*n_particle_bb,3))
    
    # This assumes that the backbone and sidechain beads are ordered from end-to-end
    positions[bb_array] = xyz
    positions[sc_array] = side_xyz
    
    # Write pdb file
    cgmodel.positions = positions * unit.angstrom
    write_pdbfile_without_topology(cgmodel, pdbfile)    
    
    # Also write dcd file (better precision)
    dcdfile = pdbfile[:-3]+'dcd'
    dcdtraj = md.Trajectory(
        xyz=positions,
        topology=md.Topology.from_openmm(cgmodel.topology),
    )
    md.Trajectory.save_dcd(dcdtraj,dcdfile)    
    
    # Store key geometric parameters
    geometry = {}
    
    geometry['sigma_bb'] = (sigma_bb_opt*unit.angstrom).in_units_of(r_unit)
    geometry['sigma_sc'] = (sigma_sc_opt*unit.angstrom).in_units_of(r_unit)
    
    # Add back units:
    geometry['helical_radius'] = r * r_unit
    geometry['particle_spacing'] = t_delta_opt * unit.radian
    geometry['pitch'] = (2*np.pi*c) * r_unit
        
    # Load dcd file into mdtraj
    traj = md.load(dcdfile,top=md.Topology.from_openmm(cgmodel.topology))
    
    # Get bb-bb bond distance
    geometry['bb_bb_distance'] = (dist_unitless(positions[bb_bond_list[0][0],:],positions[bb_bond_list[0][1],:]) * unit.angstrom).in_units_of(r_unit)
    geometry['bb_sc_distance'] = (dist_unitless(positions[sc_bond_list[0][0],:],positions[sc_bond_list[0][1],:]) * unit.angstrom).in_units_of(r_unit)
    
    # Get bb-bb-bb angle
    angle_indices = np.array([[b_angle_list[0][0], b_angle_list[0][1], b_angle_list[0][2]]])
    geometry['bb_bb_bb_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-sc angle
    angle_indices = np.array([[s_angle_list[0][0], s_angle_list[0][1], s_angle_list[0][2]]])
    geometry['bb_bb_sc_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-bb-bb torsion
    dihedral_indices = np.array([[bbbb_torsion_list[0][0], bbbb_torsion_list[0][1], bbbb_torsion_list[0][2], bbbb_torsion_list[0][3]]])
    geometry['bb_bb_bb_bb_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get sc-bb-bb-sc torsion
    dihedral_indices = np.array([[sbbs_torsion_list[0][0], sbbs_torsion_list[0][1], sbbs_torsion_list[0][2], sbbs_torsion_list[0][3]]])
    geometry['sc_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-bb-sc torsion
    dihedral_indices = np.array([[bbbs_torsion_list[0][0], bbbs_torsion_list[0][1], bbbs_torsion_list[0][2], bbbs_torsion_list[0][3]]])
    geometry['bb_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]

    # Plot helix:
    if plotfile is not None:
        plot_LJ_helix(r,c,t_par,r_eq_bb,r_eq_sc=r_eq_sc,plotfile=plotfile)
    
    return opt_sol, geometry


def optimize_helix_LJ_parameters_triangle_sidechain(radius, pitch, n_particle_bb, sigma_bb,
    bond_dist_bb, DE_popsize=200, pdbfile='LJ_helix_openmm_energy.pdb',
    plotfile='LJ_helix_openmm_energy.pdf'):
    """
    With specified radius, pitch, backbone sigma, and backbone-backbone bond length, 
    optimize the sidechain sigma and backbone-sidechain bond length in a helix with
    1 backbone bead and 3 sidechain beads in an equilateral triangle whose center is
    normal to the backbone bead, and plane of the triangle runs parallel to the helical axis.
    For now, the orientations of the triangles are alternating between 2 sidechain beads at the top,
    and 2 sidechain beads at the bottom.
    
    :param radius: fixed helical radius
    :type radius: Quantity
    
    :param pitch: fixed helical pitch (c*2*pi)
    :type pitch: Quantity
    
    :param n_particle_bb: Number of backbone particles to model
    :type n_particle_bb: int
    
    :param sigma_bb: LJ sigma parameter for backbone beads
    :type sigma_bb: Quantity
    
    :param bond_dist_bb: bond distance for bb-bb bonds.
    :type bond_dist_bb: Quantity

    :param constrain_bb_sc_bond_distance: Option to constrain bb-sc bond distance to the same as sc-sc bond distance. If False, bb_sc will be optimized.
    :type constrain_bb_sc_bond_distance: bool

    :param DE_popsize: population size to use in SciPy differential_evolution solver (default=50)
    :type DE_popsize: int

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix_openmm_energy.pdb')
    :type pdbfile: str
    
    :returns:
      - opt_sol - Results from scipy.optimize (dict)
      - geometry - Dictionary containing key geometric parameters of the optimized helix
    """
    
    r_unit = radius.unit
    # Use angstrom for writing pdb file:
    radius = radius.value_in_unit(unit.angstrom)
    pitch = pitch.value_in_unit(unit.angstrom)
    
    r = radius
    c = pitch/(2*np.pi) # Helical rise parameter
    
    # Here we need to create a cgmodel
    
    # Set initial epsilon parameters
    epsilon_bb = 1.0 * unit.kilojoule_per_mole
    epsilon_sc = 1.0 * unit.kilojoule_per_mole
    
    # Set initial sigma parameters
    sigma_sc = 1.0 * unit.angstrom
    
    cgmodel = get_helix_cgmodel_triangle(sigma_bb,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb)
    
    # Get particle type lists and bonded lists:
    (particle_type_list, bb_array, sc_array,
    bb_bond_list, bs_bond_list, ss_bond_list,
    bbb_angle_list, bbs_angle_list, bss_angle_list, sss_angle_list,
    bbbb_torsion_list, bbbs_torsion_list, bbss_torsion_list,
    bsss_torsion_list, sbbs_torsion_list) = get_helix_particle_bonded_lists_triangle(cgmodel)
    
    # Set up Simulation object beforehand:
    simulation_time_step = 5.0 * unit.femtosecond
    friction = 0.0 / unit.picosecond
    integrator = LangevinIntegrator(
        0.0 * unit.kelvin, friction, simulation_time_step.in_units_of(unit.picosecond)
    )
    simulation = Simulation(cgmodel.topology, cgmodel.system, integrator)    
    
    bond_dist_bb = bond_dist_bb.value_in_unit(unit.angstrom)

    params = (simulation, particle_type_list, r, c, n_particle_bb, bond_dist_bb)

    # Set optimization bounds [r_bs, sigma_sc, theta1, theta2]:
    bounds = [(r/50,5*r),(r/50,5*r),(0,2*np.pi/3),(0,2*np.pi/3)]
    
    opt_sol = differential_evolution(
        compute_helix_openmm_energy_vary_LJ_triangle,
        bounds,
        args=params,
        polish=True,
        popsize=DE_popsize,
    )
    
    r_bs = opt_sol.x[0]
    sigma_sc_opt = opt_sol.x[1]
    r_ss = sigma_sc_opt*np.power(2,(1/6))
    
    # Angles of rotation in-plane (x axis) for triangle templates 1 and 2
    theta1 = opt_sol.x[2]
    theta2 = opt_sol.x[3]
       
    # Compute particle spacing based on bond constraints
    t_delta_opt = get_t_from_bond_distance(r,c,bond_dist_bb)
    if t_delta_opt < 0:
        print(t_delta_opt)
        t_delta_opt *= -1
    
    t_par = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t_par[i] = i*t_delta_opt  

    # Equilibrium LJ distance (for visual representation)
    r_eq_bb = sigma_bb.value_in_unit(unit.angstrom)*np.power(2,(1/6))
    r_eq_sc = sigma_sc_opt*np.power(2,(1/6))
    
    # Get particle positions:
    xyz = get_helix_coordinates(r,c,t_par)
    
    # Place sidechain triangle normal to helix
    positions = np.zeros((4*n_particle_bb,3))

    # From the law of cosines:
    if r_bs > r_ss/2:
        K = np.sqrt(r_bs**2 - (r_ss**2)/4)  # Distance from backbone bead to center of triangle plane
    else:
        K = np.sqrt(r_ss**2 - (r_bs**2)/4)
        
    L = np.sqrt((r_ss**2)/3)                # Distance from triangle center to lower bead (orientation 1)
    M = np.sqrt(L**2 - (r_ss**2)/4)         # Distance from triangle center to top of triangle (orientation 1) 

    # To get the coordinates of the triangle, we need to rotate the coordinates 
    # by the angle separating each residue.
    
    # distance between backbone beads projected onto a circle:
    dist_bb_xy = np.sqrt(np.sum(np.power((xyz[0,0:2]-xyz[1,0:2]),2)))
    
    # helical angle between the two backbone beads projected onto a circle:
    theta_arc = np.arccos(1-dist_bb_xy**2/(2*(r**2))) # radians

    # loop over all residues, rotating the reference triangles about the z axis by theta_arc:
    
    # Orientation 1:
    tri_ref_orient1 = np.zeros((3,3))
    
    tri_ref_orient1[0,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient1[0,1] = (1+K/r)*xyz[0,1] - r_ss/2
    tri_ref_orient1[0,2] = xyz[0,2] + M
    
    tri_ref_orient1[1,0] = (1+K/r)*xyz[0,0] 
    tri_ref_orient1[1,1] = (1+K/r)*xyz[0,1] + r_ss/2
    tri_ref_orient1[1,2] = xyz[0,2] + M

    tri_ref_orient1[2,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient1[2,1] = (1+K/r)*xyz[0,1]
    tri_ref_orient1[2,2] = xyz[0,2] - L  
    
    # Apply the rotation in x:
    tri_ref_no_rotate1 = tri_ref_orient1
    
    tri_ref_orient1[0,:] = rotate_coordinates_x(tri_ref_orient1[0,:],theta1)
    tri_ref_orient1[1,:] = rotate_coordinates_x(tri_ref_orient1[1,:],theta1)
    tri_ref_orient1[2,:] = rotate_coordinates_x(tri_ref_orient1[2,:],theta1)
    
    # Orientation 2:
    tri_ref_orient2 = np.zeros((3,3))
    
    tri_ref_orient2[0,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient2[0,1] = (1+K/r)*xyz[0,1] - r_ss/2
    tri_ref_orient2[0,2] = xyz[0,2] - M
    
    tri_ref_orient2[1,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient2[1,1] = (1+K/r)*xyz[0,1] + r_ss/2
    tri_ref_orient2[1,2] = xyz[0,2] - M

    tri_ref_orient2[2,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient2[2,1] = (1+K/r)*xyz[0,1]
    tri_ref_orient2[2,2] = xyz[0,2] + L
        
    # Apply the rotation in x:
    tri_ref_no_rotate2 = tri_ref_orient2
    
    tri_ref_orient2[0,:] = rotate_coordinates_x(tri_ref_orient2[0,:],theta2)
    tri_ref_orient2[1,:] = rotate_coordinates_x(tri_ref_orient2[1,:],theta2)
    tri_ref_orient2[2,:] = rotate_coordinates_x(tri_ref_orient2[2,:],theta2)        
        
    z_rise = xyz[1,2] - xyz[0,2]    
        
    j = -1
    for i in range(n_particle_bb):
        if i % 2 == 0:
            # Orientation 1:
            j += 1
            positions[j] = xyz[i]
            
            j += 1
            triangle_xyz_a = rotate_coordinates_z(tri_ref_orient1[0,:],theta_arc*i)
            triangle_xyz_b = rotate_coordinates_z(tri_ref_orient1[1,:],theta_arc*i)
            triangle_xyz_c = rotate_coordinates_z(tri_ref_orient1[2,:],theta_arc*i)
            
            # Use only the x and y from the rotated reference:
            positions[j] = triangle_xyz_a
            positions[j,2] = triangle_xyz_a[2] + z_rise*i
            
            j += 1
            positions[j] = triangle_xyz_b
            positions[j,2] = triangle_xyz_b[2] + z_rise*i
            
            j += 1
            positions[j] = triangle_xyz_c
            positions[j,2] = triangle_xyz_c[2] + z_rise*i
            
            
        else:
            # Orientation 2:
            j += 1
            positions[j] = xyz[i]
            
            j += 1
            triangle_xyz_a = rotate_coordinates_z(tri_ref_orient2[0,:],theta_arc*i)
            triangle_xyz_b = rotate_coordinates_z(tri_ref_orient2[1,:],theta_arc*i)
            triangle_xyz_c = rotate_coordinates_z(tri_ref_orient2[2,:],theta_arc*i)
            
            # Use only the x and y from the rotated reference:
            positions[j] = triangle_xyz_a
            positions[j,2] = triangle_xyz_a[2] + z_rise*i
            
            j += 1
            positions[j] = triangle_xyz_b
            positions[j,2] = triangle_xyz_b[2] + z_rise*i
            
            j += 1
            positions[j] = triangle_xyz_c
            positions[j,2] = triangle_xyz_c[2] + z_rise*i
            
    
    # Write pdb file
    cgmodel.positions = positions * unit.angstrom
    write_pdbfile_without_topology(cgmodel, pdbfile)
    
    # Also write dcd file (better precision)
    dcdfile = pdbfile[:-3]+'dcd'
    dcdtraj = md.Trajectory(
        xyz=positions,
        topology=md.Topology.from_openmm(cgmodel.topology),
    )
    md.Trajectory.save_dcd(dcdtraj,dcdfile)    
    
    # Store key geometric parameters
    geometry = {}
    
    geometry['sigma_bb'] = sigma_bb.in_units_of(r_unit)
    geometry['sigma_sc'] = (sigma_sc_opt*unit.angstrom).in_units_of(r_unit)
    
    # Add back units:
    geometry['helical_radius'] = r * r_unit
    geometry['particle_spacing'] = t_delta_opt * unit.radian
    geometry['pitch'] = (2*np.pi*c) * r_unit
        
    # Load dcd file into mdtraj
    traj = md.load(dcdfile,top=md.Topology.from_openmm(cgmodel.topology))
    
    # Get bb-bb bond distance
    geometry['bb_bb_distance'] = (dist_unitless(positions[bb_bond_list[0][0],:],positions[bb_bond_list[0][1],:]) * unit.angstrom).in_units_of(r_unit)
    geometry['bb_sc_distance'] = (dist_unitless(positions[bs_bond_list[0][0],:],positions[bs_bond_list[0][1],:]) * unit.angstrom).in_units_of(r_unit)
    geometry['sc_sc_distance'] = (dist_unitless(positions[ss_bond_list[0][0],:],positions[ss_bond_list[0][1],:]) * unit.angstrom).in_units_of(r_unit) 
    
    # Get bb-bb-bb angle
    angle_indices = np.array([[bbb_angle_list[0][0], bbb_angle_list[0][1], bbb_angle_list[0][2]]])
    geometry['bb_bb_bb_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-sc angle (orientation 1)
    angle_indices = np.array([[bbs_angle_list[0][0], bbs_angle_list[0][1], bbs_angle_list[0][2]]])
    geometry['bb_bb_sc_angle_type1'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-sc angle (orientation 2)
    angle_indices = np.array([[bbs_angle_list[1][0], bbs_angle_list[1][1], bbs_angle_list[1][2]]])
    geometry['bb_bb_sc_angle_type2'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]    
    
    # Get bb-sc-sc angle (orientation 1)
    angle_indices = np.array([[bss_angle_list[0][0], bss_angle_list[0][1], bss_angle_list[0][2]]])
    geometry['bb_sc_sc_angle_type1'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]    
    
    # Get bb-sc-sc angle (orientation 2)
    angle_indices = np.array([[bss_angle_list[1][0], bss_angle_list[1][1], bss_angle_list[1][2]]])
    geometry['bb_sc_sc_angle_type2'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]      
    
    # Get sc-sc-sc angle (orientation 1)
    angle_indices = np.array([[sss_angle_list[0][0], sss_angle_list[0][1], sss_angle_list[0][2]]])
    geometry['sc_sc_sc_angle_type1'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]  
    
    # Get sc-sc-sc angle (orientation 2)
    angle_indices = np.array([[sss_angle_list[1][0], sss_angle_list[1][1], sss_angle_list[1][2]]])
    geometry['sc_sc_sc_angle_type2'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]      
    
    # Get bb-bb-bb-bb torsion
    dihedral_indices = np.array([[bbbb_torsion_list[0][0], bbbb_torsion_list[0][1], bbbb_torsion_list[0][2], bbbb_torsion_list[0][3]]])
    geometry['bb_bb_bb_bb_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-bb-sc torsion (orientation 1)
    dihedral_indices = np.array([[bbbs_torsion_list[0][0], bbbs_torsion_list[0][1], bbbs_torsion_list[0][2], bbbs_torsion_list[0][3]]])
    geometry['bb_bb_bb_sc_angle_type1'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]    
    
    # Get bb-bb-bb-sc torsion (orientation 2)
    dihedral_indices = np.array([[bbbs_torsion_list[1][0], bbbs_torsion_list[1][1], bbbs_torsion_list[1][2], bbbs_torsion_list[1][3]]])
    geometry['bb_bb_bb_sc_angle_type2'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]       
    
    # Get bb-bb-sc-sc torsion (orientation 1)
    dihedral_indices = np.array([[bbss_torsion_list[0][0], bbss_torsion_list[0][1], bbss_torsion_list[0][2], bbss_torsion_list[0][3]]])
    geometry['bb_bb_sc_sc_angle_type1'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]    
    
    # Get bb-bb-sc-sc torsion (orientation 2)
    dihedral_indices = np.array([[bbss_torsion_list[1][0], bbss_torsion_list[1][1], bbss_torsion_list[1][2], bbss_torsion_list[1][3]]])
    geometry['bb_bb_sc_sc_angle_type2'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]  

    # Get bb-sc-sc-sc torsion (orientation 1)
    dihedral_indices = np.array([[bsss_torsion_list[0][0], bsss_torsion_list[0][1], bsss_torsion_list[0][2], bsss_torsion_list[0][3]]])
    geometry['bb_sc_sc_sc_angle_type1'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]   
    
    # Get bb-sc-sc-sc torsion (orientation 2)
    dihedral_indices = np.array([[bsss_torsion_list[1][0], bsss_torsion_list[1][1], bsss_torsion_list[1][2], bsss_torsion_list[1][3]]])
    geometry['bb_sc_sc_sc_angle_type2'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]      
    
    # Get sc-bb-bb-sc torsion
    dihedral_indices = np.array([[sbbs_torsion_list[0][0], sbbs_torsion_list[0][1], sbbs_torsion_list[0][2], sbbs_torsion_list[0][3]]])
    geometry['sc_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]

    # # Plot helix:
    # if plotfile is not None:
        # plot_LJ_helix(r,c,t_par,r_eq_bb,r_eq_sc=r_eq_sc,plotfile=plotfile)
    
    return opt_sol, geometry


def compute_helix_openmm_energy_vary_LJ(geo, simulation, bb_array, sc_array, 
    particle_type_list, r, c, n_particle_bb):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    """
    
    # Particle spacing (radians)
    t_delta = geo[0]
    
    # Backbone sigma parameter
    sigma_bb = geo[1] * unit.angstrom
    
    # Sidechain sigma parameter
    sigma_sc = geo[2] * unit.angstrom
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_coordinates(r,c,t1)
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated. In the cgmodel, all force constants
    # are zero.
        
    # Place sidechain particles normal to helix with same bond length as bb_bb
    r_bs = dist_unitless(xyz[0,:],xyz[1,:])
    side_xyz = np.zeros((n_particle_bb,3))
    
    side_xyz[:,0] = (1+r_bs/r)*xyz[:,0]
    side_xyz[:,1] = (1+r_bs/r)*xyz[:,1]
    side_xyz[:,2] = xyz[:,2]
    
    # Now, set the backbone and sidechain positions to the correct bead indices:
    positions = np.zeros((2*n_particle_bb,3))
    
    # This assumes that the backbone and sidechain beads are ordered from end-to-end
    positions[bb_array] = xyz
    positions[sc_array] = side_xyz
    
    positions *= unit.angstrom
    
    # Update the nonbonded parameters:
    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            for particle_index in range(len(particle_type_list)):
                (q,sigma_old,eps) = force.getParticleParameters(particle_index)
                
                # Only need to change the sigma values here:
                if particle_type_list[particle_index] == 'bb':
                    force.setParticleParameters(particle_index,q,sigma_bb,eps)
                else:
                    force.setParticleParameters(particle_index,q,sigma_sc,eps)
                force.updateParametersInContext(simulation.context)
            
    # Update the positions:        
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
        
    return U_helix    
    

def rotate_coordinates_z(xyz,theta):
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
    
    
def rotate_coordinates_x(xyz,theta):
    """
    Internal function for rotating 3d coordinates by theta (angle to rotate the triangle in-plane)
    Theta is specified in radians.
    """
    
    R_mat = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)],
        ])

    xyz_rot = np.matmul(R_mat, xyz)
    
    return xyz_rot    
    
    
def compute_helix_openmm_energy_vary_LJ_constrained(
    geo, simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, bond_dist_bb, bond_dist_sc):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    """
    
    # Backbone sigma parameter
    sigma_bb = geo[0] * unit.angstrom
    
    # Sidechain sigma parameter
    sigma_sc = geo[1] * unit.angstrom
    
    # Particle spacing (radians)
    t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_coordinates(r,c,t1)
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated. In the cgmodel, all force constants
    # are zero.
        
    # Place sidechain particles normal to helix with same bond length as bb_bb
    r_bs = bond_dist_sc
    side_xyz = np.zeros((n_particle_bb,3))
    
    side_xyz[:,0] = (1+r_bs/r)*xyz[:,0]
    side_xyz[:,1] = (1+r_bs/r)*xyz[:,1]
    side_xyz[:,2] = xyz[:,2]
    
    # Now, set the backbone and sidechain positions to the correct bead indices:
    positions = np.zeros((2*n_particle_bb,3))
    
    # This assumes that the backbone and sidechain beads are ordered from end-to-end
    positions[bb_array] = xyz
    positions[sc_array] = side_xyz
    
    positions *= unit.angstrom
    
    # Update the nonbonded parameters:
    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            for particle_index in range(len(particle_type_list)):
                (q,sigma_old,eps) = force.getParticleParameters(particle_index)
                
                # Only need to change the sigma values here:
                if particle_type_list[particle_index] == 'bb':
                    force.setParticleParameters(particle_index,q,sigma_bb,eps)
                else:
                    force.setParticleParameters(particle_index,q,sigma_sc,eps)
                force.updateParametersInContext(simulation.context)
            
    # Update the positions:        
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
        
    return U_helix    
    
        
def compute_helix_openmm_energy_vary_LJ_triangle(
    geo, simulation, particle_type_list, r, c, n_particle_bb, bond_dist_bb):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    """
    
    # Backbone-sidechain bond length:
    r_bs = geo[0]
    
    # Sidechain sigma parameter
    sigma_sc = geo[1] * unit.angstrom
    
    # Angle of rotation in x for each triangle reference:
    theta1 = geo[2]
    theta2 = geo[3]
    
    # sidechain-sidechain bond length:
    r_ss = sigma_sc.value_in_unit(unit.angstrom)*np.power(2,(1/6))
    
    # Particle spacing (radians)
    t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_coordinates(r,c,t1)
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated. In the cgmodel, all force constants
    # are zero.
        
    # Place sidechain triangle normal to helix
    positions = np.zeros((4*n_particle_bb,3))

    # From the law of cosines:
    if r_bs > r_ss/2:
        K = np.sqrt(r_bs**2 - (r_ss**2)/4) 
    else:
        K = np.sqrt(r_ss**2 - (r_bs**2)/4)
        
    L = np.sqrt((r_ss**2)/3)
    M = np.sqrt(L**2 - (r_ss**2)/4)
    
    # To get the coordinates of the triangle, we need to rotate the coordinates 
    # by the angle separating each residue.
    
    # distance between backbone beads projected onto a circle:
    dist_bb_xy = np.sqrt(np.sum(np.power((xyz[0,0:2]-xyz[1,0:2]),2)))
    
    # helical angle between the two backbone beads projected onto a circle:
    theta_arc = np.arccos(1-dist_bb_xy**2/(2*(r**2))) # radians
    
    # loop over all residues, rotating the reference triangles about the z axis by theta_arc:
    
    # Orientation 1:
    tri_ref_orient1 = np.zeros((3,3))
    
    tri_ref_orient1[0,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient1[0,1] = (1+K/r)*xyz[0,1] - r_ss/2
    tri_ref_orient1[0,2] = xyz[0,2] + M
    
    tri_ref_orient1[1,0] = (1+K/r)*xyz[0,0] 
    tri_ref_orient1[1,1] = (1+K/r)*xyz[0,1] + r_ss/2
    tri_ref_orient1[1,2] = xyz[0,2] + M

    tri_ref_orient1[2,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient1[2,1] = (1+K/r)*xyz[0,1]
    tri_ref_orient1[2,2] = xyz[0,2] - L  
    
    # Apply the rotation in x:
    tri_ref_no_rotate1 = tri_ref_orient1
    
    tri_ref_orient1[0,:] = rotate_coordinates_x(tri_ref_orient1[0,:],theta1)
    tri_ref_orient1[1,:] = rotate_coordinates_x(tri_ref_orient1[1,:],theta1)
    tri_ref_orient1[2,:] = rotate_coordinates_x(tri_ref_orient1[2,:],theta1)    
    
    # Orientation 2:
    tri_ref_orient2 = np.zeros((3,3))
    
    tri_ref_orient2[0,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient2[0,1] = (1+K/r)*xyz[0,1] - r_ss/2
    tri_ref_orient2[0,2] = xyz[0,2] - M
    
    tri_ref_orient2[1,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient2[1,1] = (1+K/r)*xyz[0,1] + r_ss/2
    tri_ref_orient2[1,2] = xyz[0,2] - M

    tri_ref_orient2[2,0] = (1+K/r)*xyz[0,0]
    tri_ref_orient2[2,1] = (1+K/r)*xyz[0,1]
    tri_ref_orient2[2,2] = xyz[0,2] + L  
    
    # Apply the rotation in x:
    tri_ref_no_rotate2 = tri_ref_orient2
    
    tri_ref_orient2[0,:] = rotate_coordinates_x(tri_ref_orient2[0,:],theta2)
    tri_ref_orient2[1,:] = rotate_coordinates_x(tri_ref_orient2[1,:],theta2)
    tri_ref_orient2[2,:] = rotate_coordinates_x(tri_ref_orient2[2,:],theta2)  

    z_rise = xyz[1,2] - xyz[0,2]
        
    j = -1
    for i in range(n_particle_bb):
        if i % 2 == 0:
            # Orientation 1:
            j += 1
            positions[j] = xyz[i]
            
            j += 1
            triangle_xyz_a = rotate_coordinates_z(tri_ref_orient1[0,:],theta_arc*i)
            triangle_xyz_b = rotate_coordinates_z(tri_ref_orient1[1,:],theta_arc*i)
            triangle_xyz_c = rotate_coordinates_z(tri_ref_orient1[2,:],theta_arc*i)
            
            # Use only the x and y from the rotated reference:
            positions[j] = triangle_xyz_a
            positions[j,2] = triangle_xyz_a[2] + z_rise*i
            
            j += 1
            positions[j] = triangle_xyz_b
            positions[j,2] = triangle_xyz_b[2] + z_rise*i
            
            j += 1
            positions[j] = triangle_xyz_c
            positions[j,2] = triangle_xyz_c[2] + z_rise*i
            
            
        else:
            # Orientation 2:
            j += 1
            positions[j] = xyz[i]
            
            j += 1
            triangle_xyz_a = rotate_coordinates_z(tri_ref_orient2[0,:],theta_arc*i)
            triangle_xyz_b = rotate_coordinates_z(tri_ref_orient2[1,:],theta_arc*i)
            triangle_xyz_c = rotate_coordinates_z(tri_ref_orient2[2,:],theta_arc*i)
            
            # Use only the x and y from the rotated reference:
            positions[j] = triangle_xyz_a
            positions[j,2] = triangle_xyz_a[2] + z_rise*i
            
            j += 1
            positions[j] = triangle_xyz_b
            positions[j,2] = triangle_xyz_b[2] + z_rise*i
            
            j += 1
            positions[j] = triangle_xyz_c
            positions[j,2] = triangle_xyz_c[2] + z_rise*i
    
    positions *= unit.angstrom
    
    # Update the nonbonded parameters:
    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            for particle_index in range(len(particle_type_list)):
                (q,sigma_old,eps) = force.getParticleParameters(particle_index)
                
                # Only need to change the sigma_sc values here:
                force.setParticleParameters(particle_index,q,sigma_sc,eps)
                force.updateParametersInContext(simulation.context)
            
    # Update the positions:        
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
   
    return U_helix            