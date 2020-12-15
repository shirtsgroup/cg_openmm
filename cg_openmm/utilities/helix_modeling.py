import os
import numpy as np
import mdtraj as md
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from scipy.optimize import basinhopping, shgo, dual_annealing, differential_evolution, brute
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource


def optimize_helix(n_particle_bb, sigma, epsilon, sidechain=True, pdbfile='LJ_helix.pdb', plotfile='LJ_helix.pdf'):
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
    x0 = (0.5, sigma, sigma/3)
    bounds = [(0.1,np.pi/2),(sigma/4,2*sigma),(0.01,sigma)]
    
    params = (sigma, epsilon, n_particle_bb, sidechain)
    
    opt_sol = differential_evolution(compute_LJ_helix_energy, bounds, args=params, polish=True, popsize=25)
    
    t_delta_opt = opt_sol.x[0]
    r_opt = opt_sol.x[1]
    c_opt = opt_sol.x[2]
    
    # Plot results:
    
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
        dihedral_indices = np.array([[1+int(xyz_par.shape[0]/2),1,2,2+int(xyz_par.shape[0]/2)]])
        geometry['sc_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)
        
        # Get bb-bb-bb-sc torsion
        dihedral_indices = np.array([[0,1,2,2+int(xyz_par.shape[0]/2)]])
        geometry['bb_bb_bb_sc_angle'] = (md.compute_dihedrals(traj,dihedral_indices)*unit.radians).in_units_of(unit.degrees)
    
    plot_LJ_helix(r_opt,c_opt,t_par,r_eq,plotfile=plotfile)
    
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
    
        
def plot_LJ_helix(r,c,t_par,r_eq,plotfile='LJ_helix.pdf'):
    """
    Internal function for plotting LJ particles on a helix
    
    :param r: radius of helix
    :type r: float
    
    :param c: vertical rise parameter
    :type c: float
    
    :param t_par: particle coordinates, in radians 
    :type t_par: np.array([float])
    
    :param r_eq: equilibrium LJ12-6 distance for particles
    :type r_eq: float
    
    :param plotfile: path to file for saving plot (default = 'LJ_helix.pdf')
    :type plotfile: str
    """
    
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
        x_sphere = np.cos(u)*np.sin(v)*r_eq/2+xyz_par[i,0]
        y_sphere = np.sin(u)*np.sin(v)*r_eq/2+xyz_par[i,1]
        z_sphere = np.cos(v)*r_eq/2+xyz_par[i,2]
        ax.plot_surface(x_sphere, y_sphere, z_sphere,
            color="r", shade=True, lightsource=light, alpha=0.3)  
        
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


    