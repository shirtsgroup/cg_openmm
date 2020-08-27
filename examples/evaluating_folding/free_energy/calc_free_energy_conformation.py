# This example script demonstrates the calculation of free energy differences
# between 2 conformational states from a replica exchange simulation, which are
# classified by their RMSD using a Kmeans clustering algorithm.

import os
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from simtk import unit
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.parameters.free_energy import *
from cg_openmm.parameters.reweight import *
from analyze_foldamers.ensembles.cluster import *
import pickle

output_data = "output.nc"
output_directory = "output"
number_replicas = 36
min_temp = 50.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

# Load in cgmodel
cgmodel = pickle.load(open( "stored_cgmodel.pkl", "rb" ))

# Create list of trajectory files for clustering analysis
number_replicas = 36
pdb_file_list = []
for i in range(number_replicas):
    pdb_file_list.append("output/replica_%s.pdb" %(i+1))

# Set clustering parameters
n_clusters=2
frame_start=0
frame_stride=100
frame_end=-1
    
# Run KMeans clustering
medoid_positions, cluster_size, cluster_rmsd = get_cluster_medoid_positions(
    pdb_file_list=pdb_file_list,
    cgmodel=cgmodel,
    n_clusters=n_clusters,
    frame_start=frame_start,
    frame_stride=frame_stride,
    frame_end=-1,
    output_dir=output_directory)
    
print(f"Intra-cluster rmsd: {cluster_rmsd}")

# The smaller intra-cluster rmsd will be the folded state if it is very stable
if cluster_rmsd[0] < cluster_rmsd[1]:
    native_structure_file=f"{output_directory}/medoid_0.pdb"
else:
    native_structure_file=f"{output_directory}/medoid_1.pdb"
    
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

array_folded_states = np.zeros((number_replicas*nframes))

# Store statistics for plotting
Q_avg = np.zeros(len(temperature_list))
Q_uncertainty = np.zeros(len(temperature_list))

for rep in range(number_replicas):
    if rep > 0:
        rep_traj = md.load(pdb_file_list[rep])
        
    Q = fraction_native_contacts(
        cgmodel,
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
            array_folded_states[frame+rep*nframes] = 1
        else:
            # Not folded
            array_folded_states[frame+rep*nframes] = 0
            
plot_native_contact_fraction(temperature_list, Q_avg, Q_uncertainty)

# Save folded state array for further analysis / comparing various cutoffs            
with open('array_folded_states.pkl','wb') as array_file:
    pickle.dump(array_folded_states,array_file)

num_intermediate_states = 1

full_T_list, deltaF_values, deltaF_uncertainty = expectations_free_energy(
    array_folded_states,
    temperature_list,
    output_directory,
    output_data,
    num_intermediate_states=num_intermediate_states,
)
    
print(f"T (K), deltaF (J/mol), deltaF_uncertainty (J/mol)")
for i in range(len(full_T_list)):
    print(f"{full_T_list[i]:>6.4f}, {deltaF_values['state0_state1'][i]:>6.8f}, {deltaF_uncertainty['state0_state1'][i]:>6.8f}")
      
plot_free_energy_results(full_T_list, deltaF_values, deltaF_uncertainty,plotfile="free_energy")
