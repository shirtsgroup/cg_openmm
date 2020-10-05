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

output_directory = "output"
output_data = os.path.join(output_directory, "output.nc")
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
    file_list=pdb_file_list,
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
native_contact_cutoff_ratio = 1.0

# Cutoff for native contact fraction folded vs. unfolded states:
Q_folded = 0.7

# Determine native contacts:
native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
    cgmodel,
    native_positions,
    native_contact_cutoff
)

# Determine native contact fraction of current trajectories:

Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
    pdb_file_list,
    native_contact_list,
    native_contact_distances,
    frame_begin=4500,
    native_contact_cutoff_ratio=native_contact_cutoff_ratio
)

plot_native_contact_fraction(
    temperature_list,
    Q_avg,
    Q_stderr,
    plotfile="Q_vs_T.pdf",
)

array_folded_states = np.zeros((len(Q[:,0]),len(pdb_file_list)))

for rep in range(len(pdb_file_list)):
    # Classify into folded/unfolded states:
    for frame in range(len(Q[:,rep])):
        if Q[frame,rep] >= Q_folded:
            # Folded
            array_folded_states[frame,rep] = 1
        else:
            # Unfolded
            array_folded_states[frame,rep] = 0

num_intermediate_states = 1

full_T_list, deltaF_values, deltaF_uncertainty = expectations_free_energy(
    array_folded_states,
    temperature_list,
    output_data=output_data,
    num_intermediate_states=num_intermediate_states,
)
    
print(f"T (K), deltaF (J/mol), deltaF_uncertainty (J/mol)")
for i in range(len(full_T_list)):
    print(f"{full_T_list[i]:>6.4f}, {deltaF_values['state0_state1'][i]:>6.8f}, {deltaF_uncertainty['state0_state1'][i]:>6.8f}")
      
plot_free_energy_results(
    full_T_list,
    deltaF_values,
    deltaF_uncertainty,
    plotfile="free_energy.pdf"
)

# Fit free energy curve to spline / evaluate 1st, 2nd derivatives:
ddeltaF_out, d2deltaF_out, spline_tck = get_free_energy_derivative(
    deltaF_values['state0_state1'],
    full_T_list,
    plotfile="ddeltaF_dT.pdf",
)
