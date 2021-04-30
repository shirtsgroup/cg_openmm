import os
import numpy as np
import simtk.unit as unit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cg_openmm.simulation.tools import get_mm_energy
import pymbar
import mdtraj as md

kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant

def eval_energy(cgmodel, file_list, temperature_list,
    frame_begin=0, frame_end=-1, frame_stride=1):
    """
    Given a cgmodel with a topology and system, evaluate the energy at all structures in each
    trajectory files specified.
    
    :param cgmodel: CGModel() class object to evaluate energy with
    :type cgmodel: class
    
    :param file_list: List of trajectory files to evaluate the energies of
    :type file_list: list or str
    
    :param temperature_list: List of temperatures associated with traj_file_list
    :type temperature_list: List( float * simtk.unit.temperature )    
    
    :param frame_begin: analyze starting from this frame, discarding all prior as equilibration period (default=0)
    :type frame_begin: int
    
    :param frame_end: analyze up to this frame only, discarding the rest (default=-1).
    :type frame_end: int    
    
    :param frame_stride: advance by this many frames between each evaluation (default=1)
    :type frame_stride: int
    
    :returns:
        - U - A numpy array of energies evaluated with the given cgmodel [frames x states]
    """
    
    if len(file_list) != len(temperature_list):
        print('Mismatch between number of files and number of temperatures given.')
        exit()
    
    for i in range(len(file_list)):
        # Load in the coordinates as mdtraj object:
        T_curr = temperature_list[i]
        
        if file_list[i][-3:] == 'dcd':
            traj = md.load(file_list[i],top=md.Topology.from_openmm(cgmodel.topology))
        else:
            traj = md.load(file_list[i])
        
        # Select frames to analyze:
        if frame_end < 0:
            traj = traj[frame_begin::frame_stride]
        else:
            traj = traj[frame_begin:frame_end:frame_stride]
        
        if i == 0:
            nframes = traj.n_frames
            U = np.zeros((nframes, len(file_list)))
        
        for j in range(nframes):
            positions = traj[j].xyz[0]*unit.nanometer
            # Compute potential energy for current frame
            U[j,i] = get_mm_energy(cgmodel.topology, cgmodel.system, positions).value_in_unit(unit.kilojoules_per_mole)

            # We can compute the reduced energies here as well.
            
    return U
    
    