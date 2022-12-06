import os

import mdtraj as md
import numpy as np
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.utilities.helix_utils import *
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from openmm import LangevinIntegrator, unit
from openmm.app import Simulation
from openmm.app.pdbfile import PDBFile
from scipy.optimize import differential_evolution, root_scalar, brute, LinearConstraint


def optimize_helix_LJ_parameters(radius, pitch, n_particle_bb, epsilon_sc=None,
    bond_dist_bb=None, bond_dist_sc=None, equal_bonds=True, sigma_bb=None, DE_popsize=50,
    pdbfile='LJ_helix_openmm_energy.pdb', plotfile='LJ_helix_openmm_energy.pdf', exclusions={}):
    """
    1sc model: Optimize backbone and sidechain particle parameters along a helix with specified radius and
    pitch, with equal spacing of backbone particles and sidechain beads normal to the helix.
    
    :param radius: fixed helical radius
    :type radius: Quantity
    
    :param pitch: fixed helical pitch (c*2*pi)
    :type pitch: Quantity
    
    :param n_particle_bb: Number of backbone particles to model
    :type n_particle_bb: int
    
    :param bond_dist_bb: bond distance for bb-bb bonds. If None, bond distance will also be optimized. (default=None)
    :type bond_dist_bb: Quantity
    
    :param bond_dist_sc: bond distance for bb-sc bonds. If None, bond distance will also be optimized. If 'LJ', the equilibrium nonbonded distance will be used instead. (default=None).
    :type bond_dist_sc: Quantity
    
    :param equal_bonds: option to constrain bb-sc bond distance to equal bb-bb bond distance. If True, any specified bond distances are ignored. (default=True)
    :type equal_bonds: bool    
    
    :param sigma_bb: sigma LJ parameter for backbone. If None, sigma_bb will also be optimized. (default=None)
    :type sigma_bb: Quantity

    :param DE_popsize: population size to use in SciPy differential_evolution solver (default=50)
    :type DE_popsize: int

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix_openmm_energy.pdb')
    :type pdbfile: str
    
    :param plotfile: Path to pdf file for plotting the helical equations and particle positions (default='LJ_helix_openmm_energy.pdf')
    :type plotfile: str
    
    :param exclusions: pass cg_openmm exclusion rules to the cgmodel (by default [0,0,1] is applied to all pair types)
    :type exclusions: dict    
    
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
    if epsilon_sc is None:
        epsilon_sc = 1.0 * unit.kilojoule_per_mole
    
    # Set initial sigma parameters
    sigma_bb_init = 1.0 * unit.angstrom
    sigma_sc_init = 1.0 * unit.angstrom
    
    cgmodel = get_helix_cgmodel(sigma_bb_init,sigma_sc_init,epsilon_bb,epsilon_sc,n_particle_bb,exclusions)
    
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
    
    if (bond_dist_bb is None and bond_dist_sc is None) or equal_bonds:
    
        #-----------------------------#
        # Unconstrained bonds version #
        #-----------------------------#
    
        # Set sidechain bonds to the backbone bond distance for now
        # Vary t spacing, sigma_bb, sigma_sc

        params = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, equal_bonds)

        if equal_bonds:
            # Set optimization bounds [t, sigma_bb, sigma_sc]:
            # Use a minimium of 3 residues/turn

            # TODO: add case where we specific sigma_bb and not bond lengths
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
            
        else:
            # Set optimization bounds [t, sigma_bb, sigma_sc, r_bs]:
            # Use a minimium of 3 residues/turn
            bounds = [(0.01,2*np.pi/3),(r/50,15*r),(r/50,15*r),(r/50,15*r)]
            
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
            r_bs_opt = opt_sol.x[3]
            
    elif bond_dist_bb is not None and bond_dist_sc is None:
    
        #----------------------------#
        # 1 constrained bond version #
        #----------------------------#
    
        bond_dist_bb = bond_dist_bb.value_in_unit(unit.angstrom)
    
        params = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, bond_dist_bb, bond_dist_sc, sigma_bb)

        if sigma_bb is None:
            # Set optimization bounds [sigma_bb, sigma_sc, r_bs]: 
            bounds = [(r/50,15*r),(r/50,15*r),(r/50,15*r)]
        else:
            # Set optimization bounds [sigma_sc, r_bs]: 
            bounds = [(r/50,15*r),(r/50,15*r)]
        
        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ_constrained,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        if sigma_bb is None:
            sigma_bb_opt = opt_sol.x[0]
            sigma_sc_opt = opt_sol.x[1]
            r_bs_opt = opt_sol.x[2]
        else:
            sigma_bb_opt = sigma_bb.value_in_unit(unit.angstrom)
            sigma_sc_opt = opt_sol.x[0]
            r_bs_opt = opt_sol.x[1]
        
        # Compute particle spacing based on bond constraints
        t_delta_opt = get_t_from_bond_distance(r,c,bond_dist_bb)
        if t_delta_opt < 0:
            print(t_delta_opt)
            t_delta_opt *= -1 


    elif bond_dist_bb is None and bond_dist_sc is not None and bond_dist_sc != 'LJ':
    
        #----------------------------#
        # 1 constrained bond version #
        #----------------------------#
    
        bond_dist_sc = bond_dist_sc.value_in_unit(unit.angstrom)
    
        params = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, bond_dist_bb, bond_dist_sc, sigma_bb)
    
        # Set optimization bounds [sigma_bb, sigma_sc, t]: 
        bounds = [(r/50,15*r),(r/50,15*r),(r/50,15*r)]
        
        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ_constrained,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        sigma_bb_opt = opt_sol.x[0]
        sigma_sc_opt = opt_sol.x[1]
        t_delta_opt = opt_sol.x[2]        

    elif bond_dist_bb is not None and bond_dist_sc is not None and bond_dist_sc != 'LJ':
    
        #-----------------------------#
        # 2 constrained bonds version #
        #-----------------------------#
        
        bond_dist_bb = bond_dist_bb.value_in_unit(unit.angstrom)
        bond_dist_sc = bond_dist_sc.value_in_unit(unit.angstrom)
    
        params = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, bond_dist_bb, bond_dist_sc, sigma_bb)
    
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
            
    elif bond_dist_bb is not None and bond_dist_sc == 'LJ':
        # TODO: add the last case with bond_dist_bb is None and bond_dist_sc == 'LJ'
        
        #----------------------------#
        #   LJ distance bb-sc bonds  #
        #----------------------------#
    
        bond_dist_bb = bond_dist_bb.value_in_unit(unit.angstrom)
    
        params = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, bond_dist_bb, bond_dist_sc, sigma_bb)

        if sigma_bb is None:
            # Set optimization bounds [sigma_bb, sigma_sc]: 
            bounds = [(r/50,15*r),(r/50,15*r)]
        else:
            # Set optimization bounds [sigma_sc]:
            bounds = [(r/50,15*r)]
        
        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ_constrained,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        if sigma_bb is None:
            sigma_bb_opt = opt_sol.x[0]
            sigma_sc_opt = opt_sol.x[1]
            r_bs_opt = (sigma_bb_opt+sigma_sc_opt)/2*np.power(2,(1/6))
        else:
            sigma_bb_opt = sigma_bb.value_in_unit(unit.angstrom)
            sigma_sc_opt = opt_sol.x[0]
            r_bs_opt = r_bs_opt = (sigma_bb_opt+sigma_sc_opt)/2*np.power(2,(1/6))
        
        # Compute particle spacing based on bond constraints
        t_delta_opt = get_t_from_bond_distance(r,c,bond_dist_bb)
        if t_delta_opt < 0:
            print(t_delta_opt)
            t_delta_opt *= -1 
            
    # Determine backbone particle parametric coordinates:
    t_par = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t_par[i] = i*t_delta_opt    
        
    # Equilibrium LJ distance (for visual representation)
    r_eq_bb = sigma_bb_opt*np.power(2,(1/6))
    r_eq_sc = sigma_sc_opt*np.power(2,(1/6))
    
    # Get particle positions:
    xyz = get_helix_backbone_coordinates(r,c,t_par)
    
    # Place sidechain particles normal to helix
    r_bb = dist_unitless(xyz[0,:],xyz[1,:])
    
    if equal_bonds:
        # Use optimized bond length from first two backbone beads:
        r_bs = r_bb
    elif bond_dist_sc == 'LJ':
        r_bs = r_bs_opt
    else:
        if bond_dist_sc is not None:
            # Use specified bb-sc bond distance:
            r_bs = bond_dist_sc
        else:
            # Use optimized bb-sc bond distance:
            r_bs = r_bs_opt
    
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


def optimize_helix_LJ_parameters_energy_diff(radius, pitch, n_particle_bb,
    equal_bonds=True, brute_step_rad=0.1, brute_step_r_bs=0.1, DE_popsize=50,
    pdbfile='LJ_helix_openmm_energy.pdb', plotfile='LJ_helix_openmm_energy.pdf', exclusions={}):
    """
    1sc model: Optimize backbone and sidechain particle parameters along a helix with specified radius and
    pitch, with equal spacing of backbone particles and sidechain beads normal to the helix. Objective
    function is the energy difference beween helical and extended state, where the extended state also
    has equilibrium angles and bond distances but no helical backbone contacts. Performs a brute force
    outer optimization over particle spacing (bond distances), and an inner differential evolution 
    optimization over the sigma_bb and sigma_sc parameters. 
    
    Note: finishing function for brute is turned off to avoid costly fine-tuning of the solution
    
    :param radius: fixed helical radius
    :type radius: Quantity
    
    :param pitch: fixed helical pitch (c*2*pi)
    :type pitch: Quantity
    
    :param n_particle_bb: Number of backbone particles to model
    :type n_particle_bb: int
    
    :param equal_bonds: option to constrain bb-sc bond distance to equal bb-bb bond distance
    :type equal_bonds: bool
    
    :param brute_step_rad: step size in radians for outer brute force optimization in backbone particle spacing (default=0.1)
    :type brute_step_rad: float
    
    :param brute_step_r_bs: step size in angstrom for outer brute force optimization bb-sc bond distance (default=0.1)
    :type brute_step_r_bs: float    
    
    :param DE_popsize: population size to use in SciPy differential_evolution solver (default=50)
    :type DE_popsize: int

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix_openmm_energy.pdb')
    :type pdbfile: str
    
    :param plotfile: Path to pdf file for plotting the helical equations and particle positions (default='LJ_helix_openmm_energy.pdf')
    :type plotfile: str
    
    :param exclusions: pass cg_openmm exclusion rules to the cgmodel (by default [0,0,1] is applied to all pair types)
    :type exclusions: dict
    
    :returns:
      - geometry - Dictionary containing key geometric parameters of the optimized helix
      - opt_results_outer - Results from scipy.optimize.brute for outer loop bond distances (1D array)
      - opt_results_inner - Results from scipy.optimize.differential_evolution for optimal inner loop sigmas (dict)
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
    
    cgmodel = get_helix_cgmodel(sigma_bb,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb,exclusions)
    
    # Get particle type lists and bonded lists:
    (particle_type_list, bb_array, sc_array, bb_bond_list, sc_bond_list,
    b_angle_list, s_angle_list, bbbb_torsion_list, bbbs_torsion_list,
    sbbs_torsion_list) = get_helix_particle_bonded_lists(cgmodel)
    
    # Get the non-excluded interaction pairs:
    nonbonded_interaction_list = cgmodel.get_nonbonded_interaction_list()
    
    # Easier to separate into bb-bb, bb-sc, and sc-sc interaction lists:
    bb_interaction_list = []
    bs_interaction_list = []
    ss_interaction_list = []
    
    for pair in nonbonded_interaction_list:
        if pair[0] in bb_array and pair[1] in bb_array:
            bb_interaction_list.append(pair)
        elif pair[0] in sc_array and pair[1] in sc_array:
            ss_interaction_list.append(pair)
        else:
            bs_interaction_list.append(pair)
    
    # Set up Simulation object beforehand:
    simulation_time_step = 5.0 * unit.femtosecond
    friction = 0.0 / unit.picosecond
    integrator = LangevinIntegrator(
        0.0 * unit.kelvin, friction, simulation_time_step.in_units_of(unit.picosecond)
    )
    simulation = Simulation(cgmodel.topology, cgmodel.system, integrator)    

    if equal_bonds:
        # bb-sc bond lengths are equal to bb-bb bond lengths 
        # Vary t_spacing in outer function, vary sigma_bb, sigma_sc in inner function
    
        # Set bounds for particle spacing:
        min_rad = 2*np.pi/12 # 12 res/turn
        max_rad = 2*np.pi/3  #  3 res/turn
        brute_range = [slice(min_rad,max_rad,brute_step_rad)]

        params_outer = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, DE_popsize,
            bb_interaction_list, bs_interaction_list, ss_interaction_list)

        # Run brute force minimization over the particle spacing, with sigma_bb, sigma_sc optimized in an inner function:
        opt_results_outer = brute(compute_energy_diff_1sc, brute_range, args=params_outer, finish=None)

        t_delta_opt = opt_results_outer

        # Rerun the final inner optimization to retreive the sigma_bb, sigma_sc:
        opt_results_inner = compute_energy_diff_1sc(t_delta_opt, simulation, bb_array, sc_array,
            particle_type_list, r, c, n_particle_bb, DE_popsize,
            bb_interaction_list, bs_interaction_list, ss_interaction_list, return_inner=True)
            
        sigma_bb_opt = opt_results_inner.x[0]
        sigma_sc_opt = opt_results_inner.x[1]
        
    else:
        # bb-sc bond lengths indepedent from bb-bb bond lengths
        # Vary t_spacing, r_bs in outer function, vary sigma_bb, sigma_sc in inner function
        
        # Set bounds for particle spacing (radians):
        min_rad = 2*np.pi/12 # 12 res/turn
        max_rad = 2*np.pi/3  #  3 res/turn
        
        # bounds for r_bs bond distance (angstrom):
        min_r_bs = r/10
        max_r_bs = 3*r
        
        brute_range = (slice(min_rad,max_rad,brute_step_rad), slice(min_r_bs,max_r_bs,brute_step_r_bs))

        params_outer = (simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, DE_popsize,
            bb_interaction_list, bs_interaction_list, ss_interaction_list)

        # Run brute force minimization over the particle spacing, with sigma_bb, sigma_sc optimized in an inner function:
        opt_results_outer = brute(compute_energy_diff_1sc, brute_range, args=params_outer, finish=None)

        t_delta_opt = opt_results_outer[0]
        r_bs_opt = opt_results_outer[1]

        # Rerun the final inner optimization to retreive the sigma_bb, sigma_sc:
        opt_results_inner = compute_energy_diff_1sc(t_delta_opt, simulation, bb_array, sc_array,
            particle_type_list, r, c, n_particle_bb, DE_popsize,
            bb_interaction_list, bs_interaction_list, ss_interaction_list, return_inner=True)
            
        sigma_bb_opt = opt_results_inner.x[0]
        sigma_sc_opt = opt_results_inner.x[1]
        
        
    t_par = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t_par[i] = i*t_delta_opt    
    
    # Get particle positions:
    xyz = get_helix_backbone_coordinates(r,c,t_par)
    
    # Place sidechain particles normal to helix
    r_bb = dist_unitless(xyz[0,:],xyz[1,:])
    
    if equal_bonds:
        # Use optimized bond length from first two backbone beads:
        r_bs = r_bb
    else:
        r_bs = r_bs_opt
    
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

    # Equilibrium LJ distance (for visual representation)
    r_eq_bb = sigma_bb_opt*np.power(2,(1/6))
    r_eq_sc = sigma_sc_opt*np.power(2,(1/6))

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

    # Alse write out the extended conformation
    positions_extended = get_extended_positions_1sc(
        r_bb, r_bs, n_particle_bb, geometry['bb_bb_bb_angle'], 
        )

    # Write pdb file
    cgmodel.positions = positions_extended * unit.angstrom
    write_pdbfile_without_topology(cgmodel, pdbfile[:-4]+'_ext.pdb')    
        
    # Plot helix:
    if plotfile is not None:
        plot_LJ_helix(r,c,t_par,r_eq_bb,r_eq_sc=r_eq_sc,plotfile=plotfile)
    
    return geometry, opt_results_outer, opt_results_inner


def optimize_helix_LJ_parameters_2sc(radius, pitch, n_particle_bb, sigma_bb,
    bond_dist_bb, equal_sc=True, DE_popsize=200, pdbfile='LJ_helix_2sc_opt.pdb',
    exclusions={}):
    """
    With specified radius, pitch, backbone sigma, and backbone-backbone bond length, 
    optimize the sidechain sigma and bb-sc, sc-sc bond lengths in a helix with
    1 backbone bead and 2 sidechain beads extending normal to the backbone.
    
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

    :param equal_sc: Option for keeping the sigma_sc of each sidechain bead the same. If False, each sigma_sc will be varied independently. (default=True)
    :type equal_sc: bool
    
    :param DE_popsize: population size to use in SciPy differential_evolution solver (default=50)
    :type DE_popsize: int

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix_2sc_opt.pdb')
    :type pdbfile: str
    
    :param exclusions: pass cg_openmm exclusion rules to the cgmodel (by default [0,0,1] is applied to bb-bb, [0,1,1] to bb-sc, sc-sc)
    :type exclusions: dict    
    
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
    
    if equal_sc:
        cgmodel = get_helix_cgmodel_2sc_equal(sigma_bb,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb,exclusions)
    else:
        # Need to set 2 independent particle types for sc1, sc2
        cgmodel = get_helix_cgmodel_2sc_nonequal(sigma_bb,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb,exclusions)
    
    # Get particle type lists and bonded lists:
    # (sc1 and sc2 have the same bonded type here)
    (particle_type_list, bb_array, sc_array,
    bb_bond_list, bs_bond_list, ss_bond_list,
    bbb_angle_list, bbs_angle_list, bss_angle_list,
    bbbb_torsion_list, bbbs_torsion_list, bbss_torsion_list,
    sbbs_torsion_list)  = get_helix_particle_bonded_lists_2sc(cgmodel)
    
    # Set up Simulation object beforehand:
    simulation_time_step = 5.0 * unit.femtosecond
    friction = 0.0 / unit.picosecond
    integrator = LangevinIntegrator(
        0.0 * unit.kelvin, friction, simulation_time_step.in_units_of(unit.picosecond)
    )
    simulation = Simulation(cgmodel.topology, cgmodel.system, integrator)    
    
    bond_dist_bb = bond_dist_bb.value_in_unit(unit.angstrom)

    params = (simulation, particle_type_list, r, c, n_particle_bb, bond_dist_bb)

    if equal_sc:
        #-----------------------#
        # Equal sidechain sigma #
        #-----------------------#
        
        # Set optimization bounds [r_bs, r_ss, sigma_sc]:
        bounds = [(r/50,5*r),(r/50,5*r),(r/50,10*r)]
        
        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ_2sc_equal,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        r_bs = opt_sol.x[0]
        r_ss = opt_sol.x[1]
        
        sigma_sc1_opt = opt_sol.x[2]
        sigma_sc2_opt = sigma_sc1_opt
    
    else:
        #---------------------------#
        # Non-equal sidechain sigma #
        #---------------------------#
        
        # Set optimization bounds [r_bs, r_ss, sigma_sc1, sigma_sc2]:
        bounds = [(r/50,5*r),(r/50,5*r),(r/50,10*r),(r/50,10*r)]
        
        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ_2sc_nonequal,
            bounds,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        r_bs = opt_sol.x[0]
        r_ss = opt_sol.x[1]
        
        sigma_sc1_opt = opt_sol.x[2]
        sigma_sc2_opt = opt_sol.x[3]
           
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
    r_eq_sc1 = sigma_sc1_opt*np.power(2,(1/6))
    r_eq_sc2 = sigma_sc2_opt*np.power(2,(1/6))
    
    # Get particle positions:
    xyz = get_helix_backbone_coordinates(r,c,t_par)

    # Place sidechain particles normal to helix:
    side_xyz1 = np.zeros((n_particle_bb,3))
    side_xyz2 = np.zeros_like(side_xyz1)
    
    side_xyz1[:,0] = (1+r_bs/r)*xyz[:,0]
    side_xyz1[:,1] = (1+r_bs/r)*xyz[:,1]
    side_xyz1[:,2] = xyz[:,2]
    
    side_xyz2[:,0] = (1+(r_bs+r_ss)/r)*xyz[:,0]
    side_xyz2[:,1] = (1+(r_bs+r_ss)/r)*xyz[:,1]
    side_xyz2[:,2] = xyz[:,2]      
    
    # Now, set the backbone and sidechain positions to the correct bead indices:
    positions = np.zeros((3*n_particle_bb,3))
    
    j = -1
    for i in range(n_particle_bb):
        j += 1
        positions[j] = xyz[i]
        
        j += 1
        positions[j] = side_xyz1[i]
        
        j += 1
        positions[j] = side_xyz2[i]
            
    positions *= unit.angstrom
    
    # Write pdb file
    cgmodel.positions = positions
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
    geometry['sigma_sc1'] = (sigma_sc1_opt*unit.angstrom).in_units_of(r_unit)
    geometry['sigma_sc2'] = (sigma_sc2_opt*unit.angstrom).in_units_of(r_unit) 
    
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
    
    # Get bb-bb-sc angle
    angle_indices = np.array([[bbs_angle_list[0][0], bbs_angle_list[0][1], bbs_angle_list[0][2]]])
    geometry['bb_bb_sc_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-sc-sc angle
    angle_indices = np.array([[bss_angle_list[0][0], bss_angle_list[0][1], bss_angle_list[0][2]]])
    geometry['bb_sc_sc_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]         
    
    # Get bb-bb-bb-bb torsion
    dihedral_indices = np.array([[bbbb_torsion_list[0][0], bbbb_torsion_list[0][1], bbbb_torsion_list[0][2], bbbb_torsion_list[0][3]]])
    geometry['bb_bb_bb_bb_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-bb-sc torsion
    dihedral_indices = np.array([[bbbs_torsion_list[0][0], bbbs_torsion_list[0][1], bbbs_torsion_list[0][2], bbbs_torsion_list[0][3]]])
    geometry['bb_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]     
    
    # Get bb-bb-sc-sc torsion
    dihedral_indices = np.array([[bbss_torsion_list[0][0], bbss_torsion_list[0][1], bbss_torsion_list[0][2], bbss_torsion_list[0][3]]])
    geometry['bb_bb_sc_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]    

    # Get sc-bb-bb-sc torsion
    dihedral_indices = np.array([[sbbs_torsion_list[0][0], sbbs_torsion_list[0][1], sbbs_torsion_list[0][2], sbbs_torsion_list[0][3]]])
    geometry['sc_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    return opt_sol, geometry


def optimize_helix_LJ_parameters_2sc_rotation(radius, pitch, n_particle_bb, sigma_bb,
    bond_dist_bb, equal_sc=True, DE_popsize=200, pdbfile='LJ_helix_2sc_opt_rotation.pdb',
    n_rotation_angles=2, alignment='center', exclusions={}):
    """
    With specified radius, pitch, backbone sigma, and backbone-backbone bond length, 
    optimize the sidechain sigma, bb-sc, sc-sc bond lengths, and rotation angle of the 
    sidechain particle about the 
    
    :param radius: fixed helical radius
    :type radius: Quantity
    
    :param pitch: fixed helical pitch (c*2*pi)
    :type pitch: Quantity
    
    :param n_particle_bb: Number of backbone particles to model
    :type n_particle_bb: int
    
    :param sigma_bb: LJ sigma parameter for backbone beads. If None, sigma_bb and backbone particle spacing will also be optimized. 
    :type sigma_bb: Quantity
    
    :param bond_dist_bb: bond distance for bb-bb bonds. If None, sigma_bb and backbone particle spacing will also be optimized.
    :type bond_dist_bb: Quantity

    :param equal_sc: Option for keeping the sigma_sc of each sidechain bead the same. If False, each sigma_sc will be varied independently. (default=True)
    :type equal_sc: bool
    
    :param DE_popsize: population size to use in SciPy differential_evolution solver (default=50)
    :type DE_popsize: int

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix_2sc_opt.pdb')
    :type pdbfile: str
    
    :param n_rotation_angles: number of independent sidechain rotation angles. For example 2 treats odd and even residue rotations independently.
    :type n_rotation_angles: str
    
    :param alignment: sidechain alignment scheme - can be 'center' (center of sidechain group is fixed normal to backbone) or 'first' (first bead is normal to backbone)
    :type alignment: str    
    
    :param exclusions: pass cg_openmm exclusion rules to the cgmodel (by default [0,0,1] is applied to bb-bb, [0,1,1] to bb-sc, sc-sc)
    :type exclusions: dict    
    
    :returns:
      - opt_sol - Results from scipy.optimize (dict)
      - geometry - Dictionary containing key geometric parameters of the optimized helix
    """
    
    # Check input:
    if alignment not in ['center','first']:
        print(f'Error: invalid alignment parameter {alignment}')
        exit()
        
    if alignment == 'center' and not equal_sc:
        print(f'Error: center sidechain group alignment not compatible with non-equal sigma_sc') 
        exit()
        
    if n_rotation_angles < 1:
        print(f'n_rotation_angles must be at least 1')
        exit()
    
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
    if sigma_bb is None:
        sigma_bb_init = 1.0 * unit.angstrom
    else:
        sigma_bb_init = sigma_bb
    sigma_sc = 1.0 * unit.angstrom
    
    if bond_dist_bb is None and sigma_bb is not None:
        print(f'Warning: setting sigma_bb to None for variable backbone optimization')
        sigma_bb = None
    
    if bond_dist_bb is not None and sigma_bb is None:
        print(f'Warning: setting bond_dist_bb to None for variable backbone optimization')
        bond_dist_bb = None
    
    if equal_sc:
        cgmodel = get_helix_cgmodel_2sc_equal(sigma_bb_init,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb,exclusions)
    else:
        # Need to set 2 independent particle types for sc1, sc2
        cgmodel = get_helix_cgmodel_2sc_nonequal(sigma_bb_init,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb,exclusions)
    
    # Get particle type lists and bonded lists:
    # (sc1 and sc2 have the same bonded type here)
    (particle_type_list, bb_array, sc_array,
    bb_bond_list, bs_bond_list, ss_bond_list,
    bbb_angle_list, bbs_angle_list, bss_angle_list,
    bbbb_torsion_list, bbbs_torsion_list, bbss_torsion_list,
    sbbs_torsion_list)  = get_helix_particle_bonded_lists_2sc(cgmodel)
    
    # Set up Simulation object beforehand:
    simulation_time_step = 5.0 * unit.femtosecond
    friction = 0.0 / unit.picosecond
    integrator = LangevinIntegrator(
        0.0 * unit.kelvin, friction, simulation_time_step.in_units_of(unit.picosecond)
    )
    simulation = Simulation(cgmodel.topology, cgmodel.system, integrator)    
    
    if bond_dist_bb is not None:
        bond_dist_bb = bond_dist_bb.value_in_unit(unit.angstrom)

    if equal_sc:
        #-----------------------#
        # Equal sidechain sigma #
        #-----------------------#
        
        # Here we also pass sigma_bb to calculate the contact distance for backbone-sc2:
        params = (simulation, particle_type_list, r, c, n_particle_bb, bond_dist_bb, sigma_bb, alignment)
        
        # We can also add the tilt angle defining the cone that sc2 rotates around,
        # and make two alternating rotation theta angles
        
        # Set optimization bounds [sigma_sc, theta1, theta2]:
        if n_rotation_angles > 1:
            # Allow for n independent rotation of residue sequences
            
            # Due to symmetry, we can fix the rotation angle bounds to (-pi/2,pi/2).
            # This keeps the backbone-sc1 bond on the left side for all residues
            
            bounds = []
            
            if sigma_bb is None:
                # Also optimize the backbone parameters:
                bounds.append((r/10,10*r))  # sigma_bb
                bounds.append((0.01,2*np.pi/3))  # particle spacing in radians 
            
            bounds.append((r/10,10*r))  # sigma_sc
            
            if alignment == 'center':
                for a in range(n_rotation_angles):
                    bounds.append((-np.pi/2,+np.pi/2))  # Rotation angles

            elif alignment == 'first':
                # Due to end-to-end symmetry, we can set one range to span pi
                bounds.append((-np.pi/2,+np.pi/2))
                
                # The rest of the angles should have full range of rotation:
                for a in range(n_rotation_angles-1):
                    bounds.append((-np.pi,+np.pi))

            # No constraints if r_ss is fixed by sigma_sc
            linear_constraint = ()
            
            # If r_ss is independent variable:
            # (Corresponds to [r_ss, sigma_sc, theta1, theta2)
            # linear_constraint = LinearConstraint(
                # [-1, np.power(2,(1/6)), 0, 0],
                # [-np.power(2,(1/6))*sigma_bb.value_in_unit(unit.angstrom)],
                # [np.inf]
                # )
                
        else:
            # Uniform rotation across all residues
            if sigma_bb is None:
                bounds = [(0.01,2*np.pi/3),(r/10,10*r),(r/10,10*r),(-np.pi/2,+np.pi/2)]
                # Due to end-to-end symmetry, we can fix the rotation angle bounds to (-pi/2,pi/2)
            else:
                bounds = [(r/10,10*r),(-np.pi/2,+np.pi/2)]
            # No constraints if r_ss is fixed by sigma_sc
            linear_constraint = ()
            
        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ_2sc_rotation_equal,
            bounds,
            constraints=linear_constraint,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )
        
        if sigma_bb is None:
            t_delta = opt_sol.x[0]
            sigma_bb_opt = opt_sol.x[1] * unit.angstrom
        
            sigma_sc1_opt = opt_sol.x[2]
            sigma_sc2_opt = sigma_sc1_opt
            
        else:
            t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
            sigma_bb_opt = sigma_bb
            
            sigma_sc1_opt = opt_sol.x[0]
            sigma_sc2_opt = sigma_sc1_opt
        
        theta_opt = []
        
        if alignment == 'center' and n_rotation_angles == 2:
            
            res_per_turn = 2*np.pi/t_delta
            n_seq = np.floor(res_per_turn).astype(int)
            m_seq = np.ceil(res_per_turn).astype(int)
            
            if sigma_bb is None:
                theta1 = opt_sol.x[3]
                theta2 = opt_sol.x[4]
            else:
                theta1 = opt_sol.x[1]
                theta2 = opt_sol.x[2]
            i = 0
            while i < n_particle_bb:
                for a in range(n_seq):
                    theta_opt.append(theta1)
                    i += 1
                for b in range(m_seq):
                    theta_opt.append(theta2)
                    i += 1
                    
            theta_opt = theta_opt[0:n_particle_bb]
        
        else:
            if sigma_bb is None:
                for a in range(n_rotation_angles):
                    theta_opt.append(opt_sol.x[a+3])
            else:
                for a in range(n_rotation_angles):
                    theta_opt.append(opt_sol.x[a+1])
    else:
        #---------------------------#
        # Non-equal sidechain sigma #
        #---------------------------#
        # ***TODO: add in the variable backbone option for non-equal sigma_sc
        # Here we also pass sigma_bb to calculate the contact distance for backbone-sc2:
        params = (simulation, particle_type_list, r, c, n_particle_bb, bond_dist_bb, sigma_bb, alignment)
        
        # Here we also set r_bs=r_eq_bb_sc1, and the other contact distance will be fixed at r_eq_bb_sc2
        
        if n_rotation_angles > 1:
            # Set optimization bounds [sigma_sc1, sigma_sc2, theta1, theta2]:
            bounds = []
            bounds.append((r/10,10*r))  # sigma_sc1
            bounds.append((r/10,10*r))  # sigma_sc2
            
            # For sigma_sc2 >= sigma_sc1:
            constraint_var_list = []
            constraint_var_list.append(-1)
            constraint_var_list.append(1)
            
            if alignment == 'center':
                for a in range(n_rotation_angles):
                    bounds.append((-np.pi/2,+np.pi/2))  # Rotation angles
                    constraint_var_list.append(0)
                    
            elif alignment == 'first':
                # Due to end-to-end symmetry, we can set one range to span pi
                bounds.append((-np.pi/2,+np.pi/2))
                constraint_var_list.append(0)
                
                # The rest of the angles should have full range of rotation:
                for a in range(n_rotation_angles-1):
                    bounds.append((-np.pi,+np.pi))
                    constraint_var_list.append(0)
        
            # If we fix r_ss from sigma_sc, there are no constraints:
            # linear_constraint = ()
        
            # If we constrain sigma_sc2 >= sigma_sc1, we have the following:
            linear_constraint = LinearConstraint(
                constraint_var_list,
                [0],
                [np.inf],
                )
            
            # If r_ss is independent variable:
            # (Corresponds to [r_ss, sigma_sc1, sigma_sc2, theta1, theta2)
            # linear_constraint = LinearConstraint(
                # [[-1, np.power(2,(-5/6)), np.power(2,(-5/6)), 0, 0], [1, -np.power(2,(-5/6)), np.power(2,(-5/6)), 0, 0], [1, np.power(2,(-5/6)), -np.power(2,(-5/6)), 0, 0]],
                # [-np.power(2,(1/6))*sigma_bb.value_in_unit(unit.angstrom), 0, 0],
                # [np.inf, np.inf, np.inf]
                # )
        else:
            # Uniform rotation across all residues:
            # Set optimization bounds [sigma_sc1, sigma_sc2, theta1]:
            bounds = [(r/10,10*r),(r/10,10*r),(-np.pi/2,+np.pi/2)]
            
            # If we fix r_ss from sigma_sc, there are no constraints:
            # linear_constraint = ()
            
            # If we constrain sigma_sc2 >= sigma_sc1, we have the following:
            linear_constraint = LinearConstraint(
                [-1, 1, 0],
                [0],
                [np.inf],
                )
            
        opt_sol = differential_evolution(
            compute_helix_openmm_energy_vary_LJ_2sc_rotation_nonequal,
            bounds,
            constraints=linear_constraint,
            args=params,
            polish=True,
            popsize=DE_popsize,
        )

        sigma_sc1_opt = opt_sol.x[0]
        sigma_sc2_opt = opt_sol.x[1]
        
        theta_opt = []
        
        for a in range(n_rotation_angles):
            theta_opt.append(opt_sol.x[a+2])
           
    # Compute particle spacing based on bond constraints
    if sigma_bb is None:
        t_delta_opt = t_delta
    else:
        t_delta_opt = get_t_from_bond_distance(r,c,bond_dist_bb)
        sigma_bb_opt = sigma_bb
    
    if t_delta_opt < 0:
        print(t_delta_opt)
        t_delta_opt *= -1
    
    t_par = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t_par[i] = i*t_delta_opt 

    # Equilibrium LJ distance (for visual representation)
    r_eq_bb = sigma_bb_opt.value_in_unit(unit.angstrom)*np.power(2,(1/6))
    r_eq_sc1 = sigma_sc1_opt*np.power(2,(1/6))
    r_eq_sc2 = sigma_sc2_opt*np.power(2,(1/6))
    
    r_eq_bb_sc1 = (sigma_bb_opt.value_in_unit(unit.angstrom)+sigma_sc1_opt)/2*np.power(2,(1/6))
    r_eq_bb_sc2 = (sigma_bb_opt.value_in_unit(unit.angstrom)+sigma_sc2_opt)/2*np.power(2,(1/6))
    r_eq_sc1_sc2 = (sigma_sc1_opt+sigma_sc2_opt)/2*np.power(2,(1/6))
    
    # Can set this to r_eq_bb_sc1 for both equal and nonequal sigma_sc cases:
    r_bs = r_eq_bb_sc1
    r_ss = r_eq_sc1_sc2
    
    # Get particle coorindates:
    positions = get_helix_coordinates_2sc_rotation(
        r, c, t_par, r_bs, r_ss, r_eq_bb_sc2, theta_opt, alignment
        )
    
    # Write pdb file
    cgmodel.positions = positions
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
    
    geometry['sigma_bb'] = sigma_bb_opt.in_units_of(r_unit)
    geometry['sigma_sc1'] = (sigma_sc1_opt*unit.angstrom).in_units_of(r_unit)
    geometry['sigma_sc2'] = (sigma_sc2_opt*unit.angstrom).in_units_of(r_unit) 
    
    if alignment == 'center' and n_rotation_angles == 2:
        # Use the i-->i+n / i-->i+m sequences
        geometry[f'rotation_angle_type1'] = (theta_opt[0]*unit.radian).in_units_of(unit.degrees)
        geometry[f'rotation_angle_type2'] = (theta_opt[n_seq]*unit.radian).in_units_of(unit.degrees)
        
    else:    
        for a in range(n_rotation_angles):
            geometry[f'rotation_angle_type{a+1}'] = (theta_opt[a]*unit.radian).in_units_of(unit.degrees)
    
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
    
    # Get bb-bb-sc angle
    for a in range(n_rotation_angles):
        angle_indices = np.array([[bbs_angle_list[a][0], bbs_angle_list[a][1], bbs_angle_list[a][2]]])
        geometry[f'bb_bb_sc_angle_type{a+1}'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-sc-sc angle
    for a in range(n_rotation_angles):
        angle_indices = np.array([[bss_angle_list[a][0], bss_angle_list[a][1], bss_angle_list[a][2]]])
        geometry[f'bb_sc_sc_angle_type{a+1}'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)[0][0]                 
    
    # Get bb-bb-bb-bb torsion
    dihedral_indices = np.array([[bbbb_torsion_list[0][0], bbbb_torsion_list[0][1], bbbb_torsion_list[0][2], bbbb_torsion_list[0][3]]])
    geometry['bb_bb_bb_bb_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]
    
    # Get bb-bb-bb-sc torsion
    for a in range(n_rotation_angles):
        dihedral_indices = np.array([[bbbs_torsion_list[a][0], bbbs_torsion_list[a][1], bbbs_torsion_list[a][2], bbbs_torsion_list[a][3]]])
        geometry[f'bb_bb_bb_sc_angle_type{a+1}'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]           
    
    # Get bb-bb-sc-sc torsion
    for a in range(n_rotation_angles):
        dihedral_indices = np.array([[bbss_torsion_list[a][0], bbss_torsion_list[a][1], bbss_torsion_list[a][2], bbss_torsion_list[a][3]]])
        geometry[f'bb_bb_sc_sc_angle_type{a+1}'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0]    

    # Get sc-bb-bb-sc torsion (there are n_residues-1 of these)
    if n_rotation_angles == n_particle_bb: 
        for a in range(n_rotation_angles-1):
            dihedral_indices = np.array([[sbbs_torsion_list[a][0], sbbs_torsion_list[a][1], sbbs_torsion_list[a][2], sbbs_torsion_list[a][3]]])
            geometry[f'sc_bb_bb_sc_angle_type{a+1}'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0] 
    else:
        for a in range(n_rotation_angles):
            dihedral_indices = np.array([[sbbs_torsion_list[a][0], sbbs_torsion_list[a][1], sbbs_torsion_list[a][2], sbbs_torsion_list[a][3]]])
            geometry[f'sc_bb_bb_sc_angle_type{a+1}'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)[0][0] 
    return opt_sol, geometry


def optimize_helix_LJ_parameters_triangle_sidechain(radius, pitch, n_particle_bb, sigma_bb,
    bond_dist_bb, DE_popsize=200, pdbfile='LJ_helix_3sc_triangle_opt.pdb', alignment='center',
    alternating=True, exclusions={}):
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
    
    :param sigma_bb: LJ sigma parameter for backbone beads. If None, sigma_bb will also be optimized.
    :type sigma_bb: Quantity
    
    :param bond_dist_bb: bond distance for bb-bb bonds. If None, sigma_bb will also be optimized.
    :type bond_dist_bb: Quantity

    :param DE_popsize: population size to use in SciPy differential_evolution solver (default=50)
    :type DE_popsize: int

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix_3sc_triangle_opt.pdb')
    :type pdbfile: str
    
    :param alignment: sidechain alignment scheme - can be 'center' (center of triangle is fixed normal to backbone) or 'first' (first bead is normal to backbone) (default='center')
    :type alignment: str
    
    :param alternating: Option to treat odd and even triangle in-plane rotation independently (default=True)
    :type alternating: str
    
    :param exclusions: pass cg_openmm exclusion rules to the cgmodel (by default [0,0,1] is applied to bb-bb, [0,1,1] to bb-sc, sc-sc)
    :type exclusions: dict    
    
    :returns:
      - opt_sol - Results from scipy.optimize (dict)
      - geometry - Dictionary containing key geometric parameters of the optimized helix
    """
    
    # Check input:
    if alignment not in ['center','first']:
        print(f'Error: invalid alignment input {alignment}')
        exit()    
    
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
    
    if sigma_bb is not None:
        cgmodel = get_helix_cgmodel_triangle(sigma_bb,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb,exclusions)
    else:
        cgmodel = get_helix_cgmodel_triangle(sigma_sc,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb,exclusions)
    
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
    
    if bond_dist_bb is not None:
        bond_dist_bb = bond_dist_bb.value_in_unit(unit.angstrom)

    params = (simulation, particle_type_list, r, c, n_particle_bb, bond_dist_bb, sigma_bb, alignment, alternating)

    # Set optimization bounds [sigma_sc, theta1, theta2]:

    if sigma_bb is not None and bond_dist_bb is not None:
        if alternating:
            bounds = [(r/50,5*r),(0,2*np.pi/3),(0,2*np.pi/3)]
            
            # We have a linear constraint that r_ss <= 2*r_bs
            if alignment == 'center':
                # No constraint needed - only that sigma_sc be positive:
                linear_constraint = LinearConstraint(
                    [1, 0, 0],
                    [0],
                    [np.inf]
                    )
            elif alignment == 'first':
                linear_constraint = LinearConstraint(
                    [(np.power(2,(-1/3))-np.power(2,(1/6))), 0, 0],
                    [-np.power(2,(-1/3))*sigma_bb.value_in_unit(unit.angstrom)],
                    [np.inf]
                    ) 
                    
        else:
            bounds = [(r/50,5*r),(0,2*np.pi/3)]
        
            if alignment == 'center':
                # No constraint needed - only that sigma_sc be positive:
                linear_constraint = LinearConstraint(
                    [1, 0],
                    [0],
                    [np.inf]
                    )
            elif alignment == 'first':
                linear_constraint = LinearConstraint(
                    [(np.power(2,(-1/3))-np.power(2,(1/6))), 0],
                    [-np.power(2,(-1/3))*sigma_bb.value_in_unit(unit.angstrom)],
                    [np.inf]
                    )

    elif sigma_bb is None and bond_dist_bb is None:
        if alternating:
            # [sigma_sc, theta1, theta2, sigma_bb, t]
            bounds = [(r/50,5*r),(0,2*np.pi/3),(0,2*np.pi/3),(r/50,15*r),(0.01,2*np.pi/3)]
            # TODO: update the constraints for alignment == 'center'
            linear_constraint = ()
            
        else:
            # [sigma_sc, theta1, sigma_bb, t]
            bounds = [(r/50,5*r),(0,2*np.pi/3),(r/50,15*r),(0.01,2*np.pi/3)]
            # TODO: update the constraints for alignment == 'center'
            linear_constraint = ()
            
    
    else:
        print('Error: sigma_bb and bond_dist_bb must either both be specified, or both be None')
        exit()
                
    opt_sol = differential_evolution(
        compute_helix_openmm_energy_vary_LJ_triangle,
        bounds,
        args=params,
        polish=True,
        popsize=DE_popsize,
    )
    
    #r_bs = opt_sol.x[0]
    
    sigma_sc_opt = opt_sol.x[0]
    r_ss = sigma_sc_opt*np.power(2,(1/6))
    
    # Angles of rotation in-plane (x axis) for triangle templates 1 and 2
    theta1_opt = opt_sol.x[1]
    
    if alternating:
        theta2_opt = opt_sol.x[2]
    else:
        theta2_opt = theta1_opt
       
    if bond_dist_bb is None: 
        if alternating:
            sigma_bb = opt_sol.x[3] * unit.angstrom
            t_delta_opt = opt_sol.x[4]
        else:
            sigma_bb = opt_sol.x[2] * unit.angstrom
            t_delta_opt = opt_sol.x[3]
    
    else:
        # Compute particle spacing based on bond constraints
        t_delta_opt = get_t_from_bond_distance(r,c,bond_dist_bb)
        
    if t_delta_opt < 0:
        print(t_delta_opt)
        t_delta_opt *= -1

    t = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t[i] = i*t_delta_opt  

    # Equilibrium LJ distance (for visual representation)
    r_eq_bb = sigma_bb.value_in_unit(unit.angstrom)*np.power(2,(1/6))
    r_eq_sc = sigma_sc_opt*np.power(2,(1/6))
    r_eq_bb_sc = (sigma_bb.value_in_unit(unit.angstrom)+sigma_sc_opt)/2*np.power(2,(1/6))
    
    r_bs = r_eq_bb_sc
    
    # Get particle positions:
    positions = get_helix_coordinates_3sc_triangle(
        r, c, t, r_bs, r_ss, r_eq_bb_sc, theta1_opt, theta2_opt, alignment
        )

    # Write pdb file
    cgmodel.positions = positions
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
        
    geometry['rotation_angle_type1'] = (theta1_opt*unit.radian).in_units_of(unit.degrees)
    geometry['rotation_angle_type2'] = (theta2_opt*unit.radian).in_units_of(unit.degrees)        
        
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

    return opt_sol, geometry


def compute_helix_openmm_energy_vary_LJ(geo, simulation, bb_array, sc_array, 
    particle_type_list, r, c, n_particle_bb, equal_bonds):
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
        
    xyz = get_helix_backbone_coordinates(r,c,t1) 
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated. In the cgmodel, all force constants
    # are zero.
        
    # Place sidechain particles normal to helix:
    r_bb = dist_unitless(xyz[0,:],xyz[1,:])
    
    if not equal_bonds:
        r_bs = geo[3] # Angstrom
    else:
        r_bs = r_bb
    
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


def compute_helix_openmm_energy_vary_LJ_constrained(
    geo, simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, bond_dist_bb, bond_dist_sc, sigma_bb_in):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    """
    
    # Backbone sigma parameter
    if sigma_bb_in is None:
        sigma_bb = geo[0] * unit.angstrom
    else:
        sigma_bb = sigma_bb_in
    
    # Sidechain sigma parameter
    if sigma_bb_in is None:
        sigma_sc = geo[1] * unit.angstrom
    else:
        sigma_sc = geo[0] * unit.angstrom
    
    if bond_dist_bb is None and bond_dist_sc is not None:
        # variable bond_dist_bb, fixed bond_dist_sc
        # Particle spacing (radians):
        t_delta = geo[2]
        r_bb = get_bond_distance_from_t(r,c,t_delta)
        r_bs = bond_dist_sc
        
    elif bond_dist_sc is None and bond_dist_bb is not None:
        # variable bond_dist_sc, fixed_bond_dist_bb
        t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
        r_bb = bond_dist_bb
        if sigma_bb_in is None:
            r_bs = geo[2]
        else:
            r_bs = geo[1]
            
    elif bond_dist_sc == 'LJ' and bond_dist_bb is not None:
        # set bb-sc bond from sigmas, fixed bond_dist_bb
        t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
        r_bb = bond_dist_bb
        r_bs = (sigma_bb.value_in_unit(unit.angstrom)+sigma_sc.value_in_unit(unit.angstrom))/2*np.power(2,(1/6))
        
    # TODO: add case of 'LJ' bond_dist_sc and variable bond_dist_bb    
        
    else:
        # both bond distances fixed
        t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
        r_bb = bond_dist_bb
        r_bs = bond_dist_sc
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_backbone_coordinates(r,c,t1)
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated. In the cgmodel, all force constants
    # are zero.
        
    # Place sidechain particles normal to helix with same bond length as bb_bb
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


def compute_energy_diff_1sc(distance_vars, simulation, bb_array, sc_array, particle_type_list, r, c, n_particle_bb, DE_popsize,
    bb_interaction_list, bs_interaction_list, ss_interaction_list, return_inner=False):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    """
    
    # Particle spacing (radians)
    if np.size(distance_vars) == 1:
        t_delta = distance_vars
    else:
        t_delta = distance_vars[0]
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
    
    # Get the backbone coordinates:
    xyz = get_helix_backbone_coordinates(r,c,t1) 
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated. In the cgmodel, all force constants
    # are zero.
        
    # Place sidechain particles normal to helix with same bond length as bb_bb
    r_bb = dist_unitless(xyz[0,:],xyz[1,:])
    
    if np.size(distance_vars) == 1:
        # Equal bonds - just varying particle spacing (r_bb)
        r_bs = r_bb
    else:
        # In brute finishing minimization, enforce this to be positive:
        r_bs = np.abs(distance_vars[1])
        
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
    
    # Compute the helix backbone angle:
    # 0 -- 1 -- 2
    vec01 = np.array(xyz[0]-xyz[1])
    vec21 = np.array(xyz[2]-xyz[1])
    
    norm = np.linalg.norm(vec01)*np.linalg.norm(vec21)
    
    theta_bbb_rad = np.arccos(np.dot(vec01,vec21)/norm)
    theta_bbb = theta_bbb_rad*180/np.pi    
    
    # Load positions into an mdtraj trajectory:
    traj_helix = md.Trajectory(
        xyz=positions,
        topology=md.Topology.from_openmm(simulation.topology),
    )
    
    # Get the bounds for sigma_bb and sigma_sc from the shortest interacting particle distances:
    bb_distances = md.compute_distances(traj_helix,bb_interaction_list,periodic=False,opt=True)
    min_bb_distance_helix = np.min(bb_distances)
    
    bs_distances = md.compute_distances(traj_helix,bs_interaction_list,periodic=False,opt=True)
    min_bs_distance_helix = np.min(bs_distances)
    
    ss_distances = md.compute_distances(traj_helix,ss_interaction_list,periodic=False,opt=True)
    min_ss_distance_helix = np.min(ss_distances)
    
    # We also need to the check the distances in the extended model:
    positions_extended = get_extended_positions_1sc(
        r_bb, r_bs, n_particle_bb, theta_bbb, 
        )
        
    positions_extended *= unit.angstrom    
        
    traj_ext = md.Trajectory(
        xyz=positions_extended,
        topology=md.Topology.from_openmm(simulation.topology),
    )   
    
    bb_distances = md.compute_distances(traj_ext,bb_interaction_list,periodic=False,opt=True)
    min_bb_distance_ext = np.min(bb_distances)
    
    bs_distances = md.compute_distances(traj_ext,bs_interaction_list,periodic=False,opt=True)
    min_bs_distance_ext = np.min(bs_distances)
    
    ss_distances = md.compute_distances(traj_ext,ss_interaction_list,periodic=False,opt=True)
    min_ss_distance_ext = np.min(ss_distances)
    
    # Now take the minimum distance for each:
    if min_bb_distance_ext < min_bb_distance_helix:
        min_bb_distance = min_bb_distance_ext
    else:
        min_bb_distance = min_bb_distance_helix
        
    if min_bs_distance_ext < min_bs_distance_helix:
        min_bs_distance = min_bs_distance_ext
    else:
        min_bs_distance = min_bs_distance_helix

    if min_ss_distance_ext < min_ss_distance_helix:
        min_ss_distance = min_ss_distance_ext
    else:
        min_ss_distance = min_ss_distance_helix        
    
    bounds_sigma = [(min_bb_distance/10,min_bb_distance),(min_ss_distance/10,min_ss_distance)]
        
    # Additionally, we have a constraint that (sigma_bb + sigma_sc)/2 <= min_bs_distance
    # For now, ignore this since the sigma_bb and sigma_sc bounds should provide a reasonable
    # solution.
    
    params_inner = (simulation, positions, theta_bbb, r_bb, r_bs, particle_type_list, r, c, n_particle_bb)
        
    opt_sol_inner = differential_evolution(
        compute_energy_diff_1sc_inner,
        bounds_sigma,
        args=params_inner,
        polish=True,
        popsize=DE_popsize,
        )     

    if return_inner:
        # Final optimization to retreive inner results:
        return opt_sol_inner
    else:
        # This is the full nested optimization - return only the energy
        # print(f'Opt energy for t_delta = {t_delta}, r_bs = {r_bs}')
        # print(opt_sol_inner)
        # print(f'b-b distance: {r_bb}')
        # print(f'max_sigma_bb: {min_bb_distance}')
        # print(f'max_sigma_bs: {min_bs_distance}')
        # print(f'max_sigma_ss: {min_ss_distance}', flush=True)
        # print('')
        
        return opt_sol_inner.fun
        
 
def compute_energy_diff_1sc_inner(x, simulation, positions, theta_bbb, r_bb, r_bs, particle_type_list, r, c, n_particle_bb):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    (inner optimization of sigma_bb, sigma_sc given fixed bond lengths and helix geometry)
    """

    sigma_bb = x[0] * unit.angstrom
    sigma_sc = x[1] * unit.angstrom

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

    # Also compute the energy of an extended conformation, and compute the difference between helix/extended
    positions_extended = get_extended_positions_1sc(
        r_bb, r_bs, n_particle_bb, theta_bbb
        )
        
    positions_extended *= unit.angstrom    
        
    # Update the positions:        
    simulation.context.setPositions(positions_extended)
    potential_energy_extended = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        
    U_extended = potential_energy_extended.value_in_unit(unit.kilojoules_per_mole)
    
    
    # print(f'Helix energy: {U_helix}')
    # print(f'Extended energy: {U_extended}')
    # print('')
    # Makes sure U_helix > U_extended is not favored over the reverse
    return U_helix-U_extended  

 
def compute_helix_openmm_energy_vary_LJ_2sc_equal(geo, simulation,
    particle_type_list, r, c, n_particle_bb, bond_dist_bb):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    (2sc model, equal sigma_sc for sidechain beads)
    """

    # Backbone-sidechain bond length:
    r_bs = geo[0]
    
    # Sidechain-sidechain bond length:
    r_ss = geo[1]
    
    # Sidechain sigma parameter
    sigma_sc = geo[2] * unit.angstrom
    
    # Particle spacing (radians)
    t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_backbone_coordinates(r,c,t1)
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated. In the cgmodel, all force constants
    # are zero.
        
    # Place sidechain particles normal to helix:
    side_xyz1 = np.zeros((n_particle_bb,3))
    side_xyz2 = np.zeros_like(side_xyz1)
    
    side_xyz1[:,0] = (1+r_bs/r)*xyz[:,0]
    side_xyz1[:,1] = (1+r_bs/r)*xyz[:,1]
    side_xyz1[:,2] = xyz[:,2]
    
    side_xyz2[:,0] = (1+(r_bs+r_ss)/r)*xyz[:,0]
    side_xyz2[:,1] = (1+(r_bs+r_ss)/r)*xyz[:,1]
    side_xyz2[:,2] = xyz[:,2]
    
    # Now, set the backbone and sidechain positions to the correct bead indices:
    positions = np.zeros((3*n_particle_bb,3))
    
    j = -1
    for i in range(n_particle_bb):
        j += 1
        positions[j] = xyz[i]
        
        j += 1
        positions[j] = side_xyz1[i]
        
        j += 1
        positions[j] = side_xyz2[i]
            
    positions *= unit.angstrom
    
    # Update the nonbonded parameters:
    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            for particle_index in range(len(particle_type_list)):
                (q,sigma_old,eps) = force.getParticleParameters(particle_index)
                
                # Only need to change the sigma values here:
                if particle_type_list[particle_index] == 'bb':
                    # This is constant and set when creating the cgmodel
                    pass
                else:
                    force.setParticleParameters(particle_index,q,sigma_sc,eps)
                force.updateParametersInContext(simulation.context)
            
    # Update the positions:        
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
        
    return U_helix     
    
    
def compute_helix_openmm_energy_vary_LJ_2sc_nonequal(geo, simulation,
    particle_type_list, r, c, n_particle_bb, bond_dist_bb):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    (2sc model, different sigma_sc for sidechain beads)
    """

    # Backbone-sidechain bond length:
    r_bs = geo[0]
    
    # Sidechain-sidechain bond length:
    r_ss = geo[1]  
    
    # Sidechain sigma parameters
    sigma_sc1 = geo[2] * unit.angstrom
    sigma_sc2 = geo[3] * unit.angstrom
    
    # Particle spacing (radians)
    t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
    
    t1 = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t1[i] = i*t_delta
        
    xyz = get_helix_backbone_coordinates(r,c,t1)
        
    # If the bonds, angles, and backbone torsions are at their equilibrium positions,
    # then we don't need to update any parameters in the simulation object. Just
    # the nonbonded energies need to be evaluated. In the cgmodel, all force constants
    # are zero.
        
    # Place sidechain particles normal to helix:
    side_xyz1 = np.zeros((n_particle_bb,3))
    side_xyz2 = np.zeros_like(side_xyz1)
    
    side_xyz1[:,0] = (1+r_bs/r)*xyz[:,0]
    side_xyz1[:,1] = (1+r_bs/r)*xyz[:,1]
    side_xyz1[:,2] = xyz[:,2]
    
    side_xyz2[:,0] = (1+(r_bs+r_ss)/r)*xyz[:,0]
    side_xyz2[:,1] = (1+(r_bs+r_ss)/r)*xyz[:,1]
    side_xyz2[:,2] = xyz[:,2]    
    
    # Now, set the backbone and sidechain positions to the correct bead indices:
    positions = np.zeros((3*n_particle_bb,3))
    
    j = -1
    for i in range(n_particle_bb):
        j += 1
        positions[j] = xyz[i]
        
        j += 1
        positions[j] = side_xyz1[i]
        
        j += 1
        positions[j] = side_xyz2[i]
            
    positions *= unit.angstrom
    
    # Update the nonbonded parameters:
    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            for particle_index in range(len(particle_type_list)):
                (q,sigma_old,eps) = force.getParticleParameters(particle_index)
                
                # Only need to change the sigma values here:
                if particle_type_list[particle_index] == 'bb':
                    # This is constant and set when creating the cgmodel
                    pass
                elif particle_type_list[particle_index] == 'sc1':
                    force.setParticleParameters(particle_index,q,sigma_sc1,eps)
                elif particle_type_list[particle_index] == 'sc2':
                    force.setParticleParameters(particle_index,q,sigma_sc2,eps)    
                force.updateParametersInContext(simulation.context)
            
    # Update the positions:        
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
        
    return U_helix  
    
    
def compute_helix_openmm_energy_vary_LJ_2sc_rotation_equal(geo, simulation,
    particle_type_list, r, c, n_particle_bb, bond_dist_bb, fixed_sigma_bb, alignment):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    (2sc model, equal sigma_sc for sidechain beads)
    """
    
    if fixed_sigma_bb is None:
        t_delta = geo[0]
        sigma_bb = geo[1] * unit.angstrom
        sigma_sc = geo[2] * unit.angstrom
    else:
        t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
        sigma_bb = fixed_sigma_bb
        sigma_sc = geo[0] * unit.angstrom
    
    # Sidechain-sidechain equilibrium distance
    r_ss = np.power(2,(1/6))*sigma_sc.value_in_unit(unit.angstrom)
    
    # Equilibrium contact distance for backbone and sidechain bead 2:
    r_eq_bb_sc2 = (sigma_bb.value_in_unit(unit.angstrom)+sigma_sc.value_in_unit(unit.angstrom))/2*np.power(2,(1/6))      
    r_bs = r_eq_bb_sc2
    
    t = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t[i] = i*t_delta
    
    res_per_turn = 2*np.pi/t_delta
    n_seq = np.floor(res_per_turn).astype(int)
    m_seq = np.ceil(res_per_turn).astype(int)
    
    # Angle of rotation for sidechain 2 bead about backbone-sc1 axis:
    theta_list = []
        
    # For center alignment hexagonal packing with n < residues/turn < m 
    # we should apply angles in the following pattern:
    # (type1 * n), (type2 * m), (type1 * n), (type2 * m), ...    
    
    if fixed_sigma_bb is None:
        if alignment == 'center' and len(geo) == 5:
            theta1 = geo[3]
            theta2 = geo[4]
            i = 0
            while i < n_particle_bb:
                for a in range(n_seq):
                    theta_list.append(theta1)
                    i += 1
                for b in range(m_seq):
                    theta_list.append(theta2)
                    i += 1
                    
            theta_list = theta_list[0:n_particle_bb]
            
        else:
            # Regularly repeating sequences:
            for a in range(1,len(geo)):
                theta_list.append(geo[a])   
    else:
        if alignment == 'center' and len(geo) == 3:
            theta1 = geo[1]
            theta2 = geo[2]
            i = 0
            while i < n_particle_bb:
                for a in range(n_seq):
                    theta_list.append(theta1)
                    i += 1
                for b in range(m_seq):
                    theta_list.append(theta2)
                    i += 1
                    
            theta_list = theta_list[0:n_particle_bb]
            
        else:
            # Regularly repeating sequences:
            for a in range(1,len(geo)):
                theta_list.append(geo[a])                    
            
    # Get particle coorindates:
    positions = get_helix_coordinates_2sc_rotation(
        r, c, t, r_bs, r_ss, r_eq_bb_sc2, theta_list, alignment
        )
    
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
    
    
def compute_helix_openmm_energy_vary_LJ_2sc_rotation_nonequal(geo, simulation,
    particle_type_list, r, c, n_particle_bb, bond_dist_bb, sigma_bb, alignment):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    (2sc model, equal sigma_sc for sidechain beads)
    """
    
    # Sidechain sigma parameters
    sigma_sc1 = geo[0] * unit.angstrom
    sigma_sc2 = geo[1] * unit.angstrom
    
    # Equilibrium contact distance for backbone and sidechain beads:
    r_eq_bb_sc1 = (sigma_bb.value_in_unit(unit.angstrom)+sigma_sc1.value_in_unit(unit.angstrom))/2*np.power(2,(1/6))
    r_eq_bb_sc2 = (sigma_bb.value_in_unit(unit.angstrom)+sigma_sc2.value_in_unit(unit.angstrom))/2*np.power(2,(1/6))
    r_eq_sc1_sc2 = (sigma_sc1.value_in_unit(unit.angstrom)+sigma_sc2.value_in_unit(unit.angstrom))/2*np.power(2,(1/6))
    
    r_bs = r_eq_bb_sc1
    r_ss = r_eq_sc1_sc2
    
    # Angle of rotation for sidechain 2 bead about backbone-sc1 axis:
    theta_list = []
    for a in range(2,len(geo)):
        theta_list.append(geo[a])
    
    # Particle spacing (radians)
    t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
    
    t = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t[i] = i*t_delta
        
    # Get particle coorindates:
    positions = get_helix_coordinates_2sc_rotation(
        r, c, t, r_bs, r_ss, r_eq_bb_sc2, theta_list, alignment
        )
    
    # Update the nonbonded parameters:
    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            for particle_index in range(len(particle_type_list)):
                (q,sigma_old,eps) = force.getParticleParameters(particle_index)
                
                # Only need to change the sigma values here:
                if particle_type_list[particle_index] == 'bb':
                    # This is constant and set when creating the cgmodel
                    pass
                elif particle_type_list[particle_index] == 'sc1':
                    force.setParticleParameters(particle_index,q,sigma_sc1,eps)
                elif particle_type_list[particle_index] == 'sc2':
                    force.setParticleParameters(particle_index,q,sigma_sc2,eps)    
                    
                force.updateParametersInContext(simulation.context)
            
    # Update the positions:        
    simulation.context.setPositions(positions)
    potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
        
    return U_helix    
    
        
def compute_helix_openmm_energy_vary_LJ_triangle(geo, simulation,
    particle_type_list, r, c, n_particle_bb, bond_dist_bb, sigma_bb, alignment, alternating):
    """
    Internal function for computing openmm energy of Lennard-Jones 12-6 helix
    """
    
    # Backbone-sidechain bond length:
    # r_bs = geo[0]
    
    # Sidechain sigma parameter
    sigma_sc = geo[0] * unit.angstrom
    
    # Angle of rotation in x for each triangle reference:
    theta1 = geo[1]
    
    if alternating:
        # Independent odd/even residue in-plane triangle rotation (4D optimization)
        theta2 = geo[2]
    else:
        # All residues get the same in-plane triangle rotation (3D optimization)
        theta2 = theta1
    
    # sidechain-sidechain bond length:
    r_ss = sigma_sc.value_in_unit(unit.angstrom)*np.power(2,(1/6))
    
    if sigma_bb is None:
        if alternating:
            sigma_bb_var = geo[3]
            t_delta = geo[4]
        else:
            sigma_bb_var = geo[2]
            t_delta = geo[3]
    else:
        sigma_bb_var = sigma_bb.value_in_unit(unit.angstrom)
        # Particle spacing (radians)
        t_delta = get_t_from_bond_distance(r,c,bond_dist_bb)
    
    # sidechain-backbone equilibrium distance:
    r_eq_bb_sc = (sigma_bb_var+sigma_sc.value_in_unit(unit.angstrom))/2*np.power(2,(1/6))
    r_bs = r_eq_bb_sc    
        
    t = np.zeros(n_particle_bb)
    for i in range(n_particle_bb):
        t[i] = i*t_delta
        
    # Get particle positions:
    positions = get_helix_coordinates_3sc_triangle(
        r, c, t, r_bs, r_ss, r_eq_bb_sc, theta1, theta2, alignment
        )
    
    # Update the nonbonded parameters:
    try:
        for force_index, force in enumerate(simulation.system.getForces()):
            force_name = force.__class__.__name__
            if force_name == 'NonbondedForce':
                for particle_index in range(len(particle_type_list)):
                
                    # Only need to change the sigma_sc values here:
                    if particle_type_list[particle_index] == 'sc':
                        (q,sigma_old,eps) = force.getParticleParameters(particle_index)
                        
                        force.setParticleParameters(particle_index,q,sigma_sc,eps)
                        force.updateParametersInContext(simulation.context)
                
        # Update the positions:        
        simulation.context.setPositions(positions)
        potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        U_helix = potential_energy.value_in_unit(unit.kilojoules_per_mole)
   
    except:
        U_helix = 1E6
   
    return U_helix            