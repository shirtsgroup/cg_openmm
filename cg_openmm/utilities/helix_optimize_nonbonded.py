import os
import numpy as np
import mdtraj as md
from simtk import unit
from simtk.openmm import LangevinIntegrator
from simtk.openmm.app import Simulation
from simtk.openmm.app.pdbfile import PDBFile
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from cg_openmm.utilities.helix_utils import *
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

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix.pdb')
    :type pdbfile: str
    
    :param plotfile: Path to pdf file for plotting the helical equations and particle positions (default='LJ_helix.pdf')
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
        bounds = [(0.01,np.pi),(r/50,15*r),(r/50,15*r)]

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
        if sigma_bb > sigma_sc:
            bounds = [(r/50,15*r),(r/50,15*r)]
        else:
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
    xyz_par = get_helix_coordinates(r,c,t_par)
    
    # Place sidechain particles normal to helix
    if bond_dist_sc == None:
        # Use optimized bond length from first two backbone beads:
        r_bs = dist_unitless(xyz_par[0,:],xyz_par[1,:])
    else:
        # Use specified bb-sc bond distance:
        r_bs = bond_dist_sc
    
    side_xyz = np.zeros((n_particle_bb,3))

    side_xyz[:,0] = (1+r_bs/r)*xyz_par[:,0]
    side_xyz[:,1] = (1+r_bs/r)*xyz_par[:,1]
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
    
    geometry['sigma_bb'] = (sigma_bb_opt*unit.angstrom).in_units_of(r_unit)
    geometry['sigma_sc'] = (sigma_sc_opt*unit.angstrom).in_units_of(r_unit)
    
    geometry['helical_radius'] = (r*unit.angstrom).in_units_of(r_unit)
    geometry['particle_spacing'] = t_delta_opt * unit.radian
    geometry['pitch'] = (2*np.pi*c*unit.angstrom).in_units_of(r_unit)
        
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
    
        