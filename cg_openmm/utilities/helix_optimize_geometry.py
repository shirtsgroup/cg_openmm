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


def optimize_helix_simple(n_particle_bb, sigma, epsilon, sidechain=True, DE_popsize=50,
    pdbfile='LJ_helix.pdb', plotfile='LJ_helix.pdf'):
    """
    Optimize backbone particle positions along a helix and helical radius, vertical rise,
    with equal spacing of particles.
    
    :param n_particle_bb: Number of backbone particles to model
    :type n_particle_bb: int
    
    :param sigma: Lennard-Jones 12-6 sigma parameter
    :type sigma: Quantity
    
    :param epsilon: Lennard-Jones 12-6 epsilon parameter
    :type epsilon: Quantity
    
    :param sidechain: Option to include sidechain particles in a 1b1s model (default=True)
    :type sidechain: bool
    
    :param DE_popsize: population size to use in SciPy differential_evolution solver (default=50)
    :type DE_popsize: int    
    
    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix.pdb')
    :type pdbfile: str
    
    :param plotfile: Path to pdf file for plotting the helical equations and particle positions (default='LJ_helix.pdf')
    :type plotfile: str
    
    :returns:
      - opt_sol - Results from scipy.optimize (dict)
      - geometry - Dictionary containing key geometric parameters of the optimized helix
    """
    
    sigma_unit = sigma.unit
    # Use angstrom for writing pdb file:
    sigma = sigma.value_in_unit(unit.angstrom)
    
    eps_unit = epsilon.unit
    epsilon = epsilon.value_in_unit(unit.kilojoule_per_mole)
    
    # Set optimization bounds [t_delta, r, c]:
    bounds = [(0.01,np.pi),(sigma/50,15*sigma),(sigma/50,2*sigma)]
    
    params = (sigma, epsilon, n_particle_bb, sidechain)
    
    opt_sol = differential_evolution(
        compute_LJ_helix_energy_simple, bounds, args=params, polish=True, popsize=DE_popsize)
    
    t_delta_opt = opt_sol.x[0]
    r_opt = opt_sol.x[1]
    c_opt = opt_sol.x[2]
    
    t_par = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t_par[i] = i*t_delta_opt       
        
    # Equilibrium LJ distance    
    r_eq = sigma*np.power(2,(1/6))  
    
    # Get particle positions:
    xyz_par = get_helix_coordinates(r_opt,c_opt,t_par)
    
    if sidechain:
        # Place sidechain particles normal to helix with same bond length as bb_bb
        r_bs = dist_unitless(xyz_par[0,:],xyz_par[1,:])
        side_xyz = np.zeros((n_particle_bb,3))

        side_xyz[:,0] = (1+r_bs/r_opt)*xyz_par[:,0]
        side_xyz[:,1] = (1+r_bs/r_opt)*xyz_par[:,1]
        side_xyz[:,2] = xyz_par[:,2]

        xyz_all = np.zeros((2*n_particle_bb,3))
        xyz_all[:n_particle_bb,:] = xyz_par
        xyz_all[n_particle_bb:,:] = side_xyz

        xyz_par = xyz_all
    
    # Store key geometric parameters
    geometry = {}
    
    geometry['helical_radius'] = (r_opt*unit.angstrom).in_units_of(sigma_unit)
    geometry['particle_spacing'] = t_delta_opt * unit.radian
    geometry['pitch'] = (2*np.pi*c_opt*unit.angstrom).in_units_of(sigma_unit)
    
    # Write pdb file
    write_helix_pdbfile(xyz_par, pdbfile, sidechain)
        
    # Load pdb file into mdtraj
    traj = md.load(pdbfile)
    
    # Get bb-bb bond distance
    geometry['bb_bb_distance'] = (dist_unitless(xyz_par[0,:],xyz_par[1,:]) * unit.angstrom).in_units_of(sigma_unit)
    
    # Get bb-bb-bb angle
    angle_indices = np.array([[0,1,2]])
    dihedral_indices = np.array([[0,1,2,3]])
    
    geometry['bb_bb_bb_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-bb-bb torsion
    geometry['bb_bb_bb_bb_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    if sidechain:
        # Get bb-bb-sc angle
        angle_indices = np.array([[0,1,1+int(xyz_par.shape[0]/2)]])
        geometry['bb_bb_sc_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
        
        # Get sc-bb-bb-sc torsion
        dihedral_indices = np.array([[int(xyz_par.shape[0]/2),0,1,1+int(xyz_par.shape[0]/2)]])
        geometry['sc_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
        
        # Get bb-bb-bb-sc torsion
        dihedral_indices = np.array([[0,1,2,2+int(xyz_par.shape[0]/2)]])
        geometry['bb_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Plot helix:
    if plotfile is not None:
        plot_LJ_helix(r_opt,c_opt,t_par,r_eq,plotfile=plotfile)
    
    return opt_sol, geometry
    
   
def optimize_helix_openmm_energy(n_particle_bb, sigma_bb, sigma_sc, epsilon_bb, epsilon_sc,
    bond_dist_bb=None, bond_dist_sc=None,DE_popsize=50,
    pdbfile='LJ_helix_openmm_energy.pdb', plotfile='LJ_helix_openmm_energy.pdf'):
    """
    Optimize backbone particle positions along a helix and helical radius, vertical rise,
    with equal spacing of particles. Assumes a 1-1 model with sidechain beads normal to helix.
    
    :param n_particle_bb: Number of backbone particles to model
    :type n_particle_bb: int
    
    :param sigma_bb: Lennard-Jones 12-6 sigma parameter of backbone bead
    :type sigma_bb: Quantity
    
    :param sigma_sc: Lennard-Jones 12-6 sigma parameter of sidechain bead
    :type sigma_sc: Quantity
    
    :param epsilon_bb: Lennard-Jones 12-6 epsilon parameter of backbone bead
    :type epsilon_bb: Quantity
    
    :param epsilon_sc: Lennard-Jones 12-6 epsilon parameter of sidechain bead
    :type epsilon_sc: Quantity
    
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
    
    sigma_unit = sigma_bb.unit
    # Use angstrom for writing pdb file:
    sigma_bb = sigma_bb.in_units_of(unit.angstrom)
    sigma_sc = sigma_sc.in_units_of(unit.angstrom)
    
    eps_unit = epsilon_bb.unit
    epsilon_bb = epsilon_bb.in_units_of(unit.kilojoules_per_mole)
    epsilon_sc = epsilon_sc.in_units_of(unit.kilojoules_per_mole)
    
    # t_delta is related to the specified bond distance - this must be computed at each iteration
    
    # Here we need to create a cgmodel
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
    
        params = (simulation, bb_array, sc_array, n_particle_bb)
    
        # Set optimization bounds [t, r, c]:
        # Bounds cannot have units attached.
        sigma_bb_val = sigma_bb.value_in_unit(unit.angstrom)
        sigma_sc_val = sigma_sc.value_in_unit(unit.angstrom)
        
        if sigma_bb > sigma_sc:
            bounds = [(0.01,np.pi),(sigma_bb_val/50,15*sigma_bb_val),(sigma_bb_val/50,2*sigma_bb_val)]
        else:
            bounds = [(0.01,np.pi),(sigma_sc_val/50,15*sigma_sc_val),(sigma_sc_val/50,2*sigma_sc_val)]
        
        opt_sol = differential_evolution(
            compute_LJ_helix_openmm_energy,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        t_delta_opt = opt_sol.x[0]
        r_opt = opt_sol.x[1]
        c_opt = opt_sol.x[2]
           
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
    
        params = (simulation, bb_array, sc_array, n_particle_bb, bond_dist_bb, bond_dist_sc)
    
        # Set optimization bounds [r, c]:
        # Bounds cannot have units attached.
        sigma_bb_val = sigma_bb.value_in_unit(unit.angstrom)
        sigma_sc_val = sigma_sc.value_in_unit(unit.angstrom)
        
        if sigma_bb > sigma_sc:
            bounds = [(sigma_bb_val/50,15*sigma_bb_val),(sigma_bb_val/50,2*sigma_bb_val)]
        else:
            bounds = [(sigma_sc_val/50,15*sigma_sc_val),(sigma_sc_val/50,2*sigma_sc_val)]
        
        opt_sol = differential_evolution(
            compute_LJ_helix_openmm_energy_constrained,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        r_opt = opt_sol.x[0]
        c_opt = opt_sol.x[1]
           
        # Compute particle spacing based on bond constraints
        t_delta_opt = get_t_from_bond_distance(r_opt,c_opt,bond_dist_bb)
        if t_delta_opt < 0:
            print(t_delta_opt)
            t_delta_opt *= -1
        
        t_par = np.zeros(n_particle_bb)
        for i in range(n_particle_bb):
            t_par[i] = i*t_delta_opt    
        
        
    # Equilibrium LJ distance (for visual representation)
    r_eq_bb = sigma_bb_val*np.power(2,(1/6))
    r_eq_sc = sigma_sc_val*np.power(2,(1/6))
    
    # Get particle positions:
    xyz_par = get_helix_coordinates(r_opt,c_opt,t_par)
    
    # Place sidechain particles normal to helix
    if bond_dist_sc == None:
        # Use optimized bond length from first two backbone beads:
        r_bs = dist_unitless(xyz_par[0,:],xyz_par[1,:])
    else:
        # Use specified bb-sc bond distance:
        r_bs = bond_dist_sc
    
    side_xyz = np.zeros((n_particle_bb,3))

    side_xyz[:,0] = (1+r_bs/r_opt)*xyz_par[:,0]
    side_xyz[:,1] = (1+r_bs/r_opt)*xyz_par[:,1]
    side_xyz[:,2] = xyz_par[:,2]
    
    # Now, set the backbone and sidechain positions to the correct bead indices:
    positions = np.zeros((2*n_particle_bb,3))
    
    # This assumes that the backbone and sidechain beads are ordered from end-to-end
    positions[bb_array] = xyz_par
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
    
    geometry['helical_radius'] = (r_opt*unit.angstrom).in_units_of(sigma_unit)
    geometry['particle_spacing'] = t_delta_opt * unit.radian
    geometry['pitch'] = (2*np.pi*c_opt*unit.angstrom).in_units_of(sigma_unit)
        
    # Load dcd file into mdtraj
    traj = md.load(dcdfile,top=md.Topology.from_openmm(cgmodel.topology))
    
    # Get bb-bb bond distance
    geometry['bb_bb_distance'] = (dist_unitless(positions[bb_bond_list[0][0],:],positions[bb_bond_list[0][1],:]) * unit.angstrom).in_units_of(sigma_unit)
    geometry['bb_sc_distance'] = (dist_unitless(positions[sc_bond_list[0][0],:],positions[sc_bond_list[0][1],:]) * unit.angstrom).in_units_of(sigma_unit)
    
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

    if plotfile is not None:
        plot_LJ_helix(r_opt,c_opt,t_par,r_eq_bb,r_eq_sc=r_eq_sc,plotfile=plotfile)
    
    return opt_sol, geometry       
    
   
def compute_LJ_helix_energy_simple(geo, sigma, epsilon, n_particle_bb, sidechain):
    """
    Internal function for computing energy of Lennard-Jones 12-6 helix
    """
    
    # Particle spacing (radians)
    t_delta = geo[0]
    
    # Helical radius (units of sigma)
    r = geo[1]
    
    # Vertical rise parameter (units of sigma)
    c = geo[2]
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_coordinates(r,c,t1)    
        
    # Add any sidechain beads
    if sidechain:
        # Place sidechain particles normal to helix with same bond length as bb_bb
        r_bs = dist_unitless(xyz[0,:],xyz[1,:])
        side_xyz = np.zeros((n_particle_bb,3))
        
        side_xyz[:,0] = (1+r_bs/r)*xyz[:,0]
        side_xyz[:,1] = (1+r_bs/r)*xyz[:,1]
        side_xyz[:,2] = xyz[:,2]
        
        xyz_all = np.zeros((2*n_particle_bb,3))
        xyz_all[:n_particle_bb,:] = xyz
        xyz_all[n_particle_bb:,:] = side_xyz
        
        xyz = xyz_all
        
    U_helix = 0    
    for i in range(xyz.shape[0]):
        for j in range(i+1,xyz.shape[0]):
            U_helix += 4*epsilon*(np.power((sigma/dist_unitless(xyz[i,:],xyz[j,:])),12) - \
                np.power((sigma/dist_unitless(xyz[i,:],xyz[j,:])),6)) 
                
    return U_helix
    
    
def compute_LJ_helix_openmm_energy(geo, simulation, bb_array, sc_array, n_particle_bb):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    """
    
    # Particle spacing (radians)
    t_delta = geo[0]
    
    # Helical radius (units of sigma)
    r = geo[1]
    
    # Vertical rise parameter (units of sigma)
    c = geo[2]
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_coordinates(r,c,t1)
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated.
        
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
    
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()   
    U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
        
    return U_helix    


def compute_LJ_helix_openmm_energy_constrained(
    geo, simulation, bb_array, sc_array, n_particle_bb, bond_dist_bb, bond_dist_sc):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    """
    
    # Helical radius (units of sigma)
    r = geo[0]
    
    # Vertical rise parameter (units of sigma)
    c = geo[1]
    
    # Particle spacing (radians)
    t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_coordinates(r,c,t1)
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated.
        
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
    
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()   
    U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
        
    return U_helix        
    
