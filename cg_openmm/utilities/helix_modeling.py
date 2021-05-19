import os
import numpy as np
import mdtraj as md
from simtk import unit
from simtk.openmm import LangevinIntegrator
from simtk.openmm.app import Simulation
from simtk.openmm.app.pdbfile import PDBFile
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from scipy.optimize import differential_evolution, root_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource


def optimize_helix(n_particle_bb, sigma, epsilon, sidechain=True, DE_popsize=50,
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
    bounds = [(0.1,np.pi/2),(sigma/4,2*sigma),(0.01,sigma)]
    
    params = (sigma, epsilon, n_particle_bb, sidechain)
    
    opt_sol = differential_evolution(
        compute_LJ_helix_energy, bounds, args=params, polish=True, popsize=DE_popsize)
    
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
    
    geometry['bb_bb_bb_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)
    
    # Get bb-bb-bb-bb torsion
    geometry['bb_bb_bb_bb_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)
    
    if sidechain:
        # Get bb-bb-sc angle
        angle_indices = np.array([[0,1,1+int(xyz_par.shape[0]/2)]])
        geometry['bb_bb_sc_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)
        
        # Get sc-bb-bb-sc torsion
        dihedral_indices = np.array([[int(xyz_par.shape[0]/2),0,1,1+int(xyz_par.shape[0]/2)]])
        geometry['sc_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)
        
        # Get bb-bb-bb-sc torsion
        dihedral_indices = np.array([[0,1,2,2+int(xyz_par.shape[0]/2)]])
        geometry['bb_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)
    
    # Plot results:
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

    :param pdbfile: Path to pdb file for saving the helical structure (default='LJ_helix.pdb')
    :type pdbfile: str
    
    :param plotfile: Path to pdf file for plotting the helical equations and particle positions (default='LJ_helix.pdf')
    :type plotfile: str
    
    :returns:
      - opt_sol - Results from scipy.optimize (dict)
      - geometry - Dictionary containing key geometric parameters of the optimized helix
    """
    
    sigma_unit = sigma_bb.unit
    # Use angstrom for writing pdb file:
    sigma_bb = sigma_bb.value_in_unit(unit.angstrom)
    sigma_sc = sigma_sc.value_in_unit(unit.angstrom)
    
    eps_unit = epsilon_bb.unit
    epsilon_bb = epsilon_bb.value_in_unit(unit.kilojoule_per_mole)
    epsilon_sc = epsilon_sc.value_in_unit(unit.kilojoule_per_mole)
    
    # t_delta is related to the specified bond distance - this must be computed at each iteration
    
    # Here we need to create a cgmodel
    cgmodel = get_helix_cgmodel(sigma_bb,sigma_sc,epsilon_bb,epsilon_sc,n_particle_bb)
    
    # Now, we need to determine which of the positions are backbone beads, and which
    # are sidechain beads
    particle_list = cgmodel.create_particle_list()
    
    bb_list = []
    sc_list = []
    
    # Use particle indices rather than the full particle dictionary:
    for par in range(len(particle_list)):
        if cgmodel.get_particle_type_name(par) == 'bb':
            bb_list.append(par)
        elif cgmodel.get_particle_type_name(par) == 'sc':
            sc_list.append(par)
    
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
        if sigma_bb > sigma_sc:
            bounds = [(0.1,np.pi/2),(sigma_bb/4,2*sigma_bb),(0.01,sigma_bb)]
        else:
            bounds = [(0.1,np.pi/2),(sigma_sc/4,2*sigma_sc),(0.01,sigma_sc)]    
        
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
        if sigma_bb > sigma_sc:
            bounds = [(sigma_bb/4,2*sigma_bb),(0.01,sigma_bb)]
        else:
            bounds = [(sigma_sc/4,2*sigma_sc),(0.01,sigma_sc)]    
        
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
    r_eq_bb = sigma_bb*np.power(2,(1/6))
    r_eq_sc = sigma_sc*np.power(2,(1/6))
    
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
    
    # Store key geometric parameters
    geometry = {}
    
    geometry['helical_radius'] = (r_opt*unit.angstrom).in_units_of(sigma_unit)
    geometry['particle_spacing'] = t_delta_opt * unit.radian
    geometry['pitch'] = (2*np.pi*c_opt*unit.angstrom).in_units_of(sigma_unit)
        
    # Load pdb file into mdtraj
    traj = md.load(pdbfile)
    
    # Get bb-bb bond distance
    geometry['bb_bb_distance'] = (dist_unitless(positions[bb_bond_list[0][0],:],positions[bb_bond_list[0][1],:]) * unit.angstrom).in_units_of(sigma_unit)
    geometry['bb_sc_distance'] = (dist_unitless(positions[sc_bond_list[0][0],:],positions[sc_bond_list[0][1],:]) * unit.angstrom).in_units_of(sigma_unit)
    
    # Get bb-bb-bb angle
    angle_indices = np.array([[b_angle_list[0][0], b_angle_list[0][1], b_angle_list[0][2]]])
    geometry['bb_bb_bb_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)
    
    # Get bb-bb-sc angle
    angle_indices = np.array([[s_angle_list[0][0], s_angle_list[0][1], s_angle_list[0][2]]])
    geometry['bb_bb_sc_angle'] = (md.compute_angles(traj,angle_indices)*unit.radians).in_units_of(unit.degrees)    
    
    # Get bb-bb-bb-bb torsion
    dihedral_indices = np.array([[bbbb_torsion_list[0][0], bbbb_torsion_list[0][1], bbbb_torsion_list[0][2], bbbb_torsion_list[0][3]]])
    geometry['bb_bb_bb_bb_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)
    
    # Get sc-bb-bb-sc torsion
    dihedral_indices = np.array([[sbbs_torsion_list[0][0], sbbs_torsion_list[0][1], sbbs_torsion_list[0][2], sbbs_torsion_list[0][3]]])
    geometry['sc_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)
    
    # Get bb-bb-bb-sc torsion
    dihedral_indices = np.array([[bbbs_torsion_list[0][0], bbbs_torsion_list[0][1], bbbs_torsion_list[0][2], bbbs_torsion_list[0][3]]])
    geometry['bb_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)

    plot_LJ_helix(r_opt,c_opt,t_par,r_eq_bb,r_eq_sc=r_eq_sc,plotfile=plotfile)
    
    return opt_sol, geometry       
   
   
def dist_unitless(positions_1, positions_2):
    # Distance function:
    return np.sqrt(np.sum(np.power((positions_1 - positions_2),2)))   
    
   
def compute_LJ_helix_energy(geo, sigma, epsilon, n_particle_bb, sidechain):
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
        "sigma": sigma_bb * unit.angstrom,
        "epsilon": epsilon_bb * unit.kilojoules_per_mole,
        "mass": mass
    }
        
    # Sidechain particle:
    sc = {
        "particle_type_name": "sc",
        "sigma": sigma_sc * unit.angstrom,
        "epsilon": epsilon_sc * unit.kilojoules_per_mole,
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
