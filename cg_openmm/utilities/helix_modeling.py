import os
import numpy as np
import mdtraj as md
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from scipy.optimize import basinhopping, shgo, dual_annealing, differential_evolution, brute

def optimize_helix(n_particle_bb, sigma, epsilon, sidechain=False):
    """
    Optimize backbone particle positions along a helix and helical radius, vertical rise,
    with equal spacing of particles.
    """
    
    # Set optimization bounds [t_delta, r, c]:
    x0 = (0.5, sigma, sigma/3)
    bounds = [(0.1,np.pi/2),(sigma/4,2*sigma),(0.01,sigma)]
    
    params = (sigma, epsilon, n_particle_bb, sidechain)
    
    opt_sol = differential_evolution(compute_LJ_helix_energy, bounds, args=params, polish=True,popsize=10)
    
    return opt_sol
    
    
def compute_LJ_helix_energy(geo, sigma, epsilon, n_particle_bb, sidechain):
    """
    Internal function for computing energy of Lennard-Jones helix
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
        
    # Distance function
    def dist_unitless(positions_1, positions_2):
        return np.sqrt(np.sum(np.power((positions_1 - positions_2),2)))   
        
    # Add any sidechain beads
    if sidechain:
        # Place sidechain particles normal to helix with same bond length as bb_bb
        r_bs = dist_unitless(xyz[0,:],xyz[1,:])
        side_xyz = np.zeros((n_particle_bb,3))
        
        side_xyz[:,0] = (1+r_bs/r)*xyz[:,0]
        side_xyz[:,1] = (1+r_bs/r)*xyz[:,1]
        side_xyz[:,2] = xyz[:,2]
        
        xyz_all = np.zeros(2*n_particle_bb,3)
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
    
        
    
    
    