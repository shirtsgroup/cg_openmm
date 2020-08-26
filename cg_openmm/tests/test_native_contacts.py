"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed
import os
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.parameters.free_energy import *
from cg_openmm.parameters.reweight import get_temperature_list
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))
structures_path = os.path.join(current_path, 'test_structures')
data_path = os.path.join(current_path, 'test_data')

def test_native_contacts(tmpdir):
    """See if we can determine native contacts and classify folded vs. unfolded states"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 36
    min_temp = 50.0 * unit.kelvin
    max_temp = 400.0 * unit.kelvin
    temperature_list_full = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Test on selected temperatures:
    temperature_list = []
    for i in [5,10,15,20,25,30]:
        temperature_list.append(temperature_list_full[i-1])
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of pdb trajectories to analyze
    pdb_file_list = []
    for i in [5,10,15,20,25,30]:
        pdb_file_list.append(f"{data_path}/replica_{i}.pdb")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.pdb"
    
    native_positions = PDBFile(native_structure_file).getPositions()
    
    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 3.5* unit.angstrom

    # Cutoff for current trajectory distances, as a multiple of native_contact_cutoff
    native_contact_cutoff_ratio = 1.25

    # Cutoff for native contact fraction folded vs. unfolded states:
    Q_folded = 0.9
    
    # Determine native contacts:
    native_contact_list, native_contact_distances = get_native_contacts(
        cgmodel,
        native_positions,
        native_contact_cutoff
    )
    
    # Determine native contact fraction of current trajectories:
    rep_traj = md.load(pdb_file_list[0])
    nframes = rep_traj.n_frames
    
    array_folded_states = np.zeros((len(pdb_file_list)*nframes))
    
    for rep in range(len(pdb_file_list)):
        if rep > 0:
            rep_traj = md.load(pdb_file_list[rep])
            
        Q = fraction_native_contacts(
            cgmodel,
            rep_traj,
            native_contact_list,
            native_contact_distances,
            native_contact_cutoff_ratio=native_contact_cutoff_ratio
        )
        
        # Classify into folded/unfolded states:
        for frame in range(len(Q)):
            if Q[frame] >= Q_folded:
                # Folded
                array_folded_states[frame+rep*nframes] = 1
            else:
                # Not folded
                array_folded_states[frame+rep*nframes] = 0
