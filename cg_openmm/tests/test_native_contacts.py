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
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 300.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of pdb trajectories to analyze
    pdb_file_list = []
    for i in range(len(temperature_list)):
        pdb_file_list.append(f"{data_path}/replica_{i+1}.pdb")
        
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
    
    array_folded_states = np.zeros((nframes,len(pdb_file_list)))
    
    # Store statistics for plotting
    Q_avg = np.zeros(len(temperature_list))
    Q_uncertainty = np.zeros(len(temperature_list))
    
    for rep in range(len(pdb_file_list)):
        if rep > 0:
            rep_traj = md.load(pdb_file_list[rep])
            
        Q = fraction_native_contacts(
            rep_traj,
            native_contact_list,
            native_contact_distances,
            native_contact_cutoff_ratio=native_contact_cutoff_ratio
        )
        
        Q_avg[rep] = np.mean(Q)
        # Compute standard error:
        Q_uncertainty[rep] = np.std(Q)/np.sqrt(len(Q))        
        
        # Classify into folded/unfolded states:
        for frame in range(len(Q)):
            if Q[frame] >= Q_folded:
                # Folded
                array_folded_states[frame,rep] = 1
            else:
                # Unfolded
                array_folded_states[frame,rep] = 0
                
    plot_native_contact_fraction(
        temperature_list,
        Q_avg,
        Q_uncertainty,
        plotfile=f"{output_directory}/Q_vs_T.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_vs_T.pdf")
    
    # Test free energy of folding:
    output_data = os.path.join(data_path, "output.nc")
    
    full_T_list, deltaF_values, deltaF_uncertainty = expectations_free_energy(
        array_folded_states,
        temperature_list,
        frame_begin=100,
        output_directory=data_path,
        output_data=output_data,
        num_intermediate_states=1,
    )
    
    plot_free_energy_results(
        full_T_list,
        deltaF_values,
        deltaF_uncertainty,
        plotfile=f"{output_directory}/free_energy"
    )

    assert os.path.isfile(f"{output_directory}/free_energy.pdf")
    
    
def test_expectations_fraction_contacts(tmpdir):
    """See if we can determine native contacts expectations as a function of T"""

    output_directory = tmpdir.mkdir("output")

    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 300.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    pdb_file_list = []
    for i in range(len(temperature_list)):
        pdb_file_list.append(f"{data_path}/replica_{i+1}.pdb")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.pdb"

    native_positions = PDBFile(native_structure_file).getPositions()

    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 3.5* unit.angstrom

    # Get native contacts:
    native_contact_list, native_contact_distances = get_native_contacts(
        cgmodel,
        native_positions,
        native_contact_cutoff
    )

    output_data = os.path.join(data_path, "output.nc")

    results = expectations_fraction_contacts(
        native_contact_list,
        native_contact_distances,
        temperature_list,
        pdb_file_list,
        frame_begin=100,
        output_directory=data_path,
        output_data=output_data,
        num_intermediate_states=1,
    )
