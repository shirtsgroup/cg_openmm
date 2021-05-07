#!~/anaconda3/bin/python

import os
import pickle
import time

from analyze_foldamers.ensembles.cluster import *
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.simulation.tools import minimize_structure
from cg_openmm.utilities.util import fit_sigmoid
from simtk import unit

#---------------------------------------#
# Find native structure from clustering #
#---------------------------------------#

# Predict native structure from rmsd clustering:
data_directory = '../run_replica_exchange/output'

# Load in cgmodel:
cgmodel = pickle.load(open("../run_replica_exchange/stored_cgmodel.pkl","rb"))

# Load in trajectory stats:
analysis_stats = pickle.load(open("../run_replica_exchange/analysis_stats_discard_20ns.pkl","rb"))

# Create list of replica trajectory files:
dcd_file_list_rep = []
number_replicas = 12
# We could have also have used state trajectory files here

for rep in range(number_replicas):
    dcd_file_list_rep.append(f"{data_directory}/replica_{rep+1}.dcd")

cluster_output_dir = 'cluster_output'    
    
(medoid_positions, cluster_sizes, cluster_rmsd, n_noise,
silhouette_avg, labels, original_indices) = get_cluster_medoid_positions_DBSCAN(
    file_list=dcd_file_list_rep,
    cgmodel=cgmodel,
    min_samples=100,
    eps=0.10,
    frame_start=analysis_stats["production_start"],
    frame_stride=200, # Using every frame can cause memory issues for large systems
    filter=True,
    filter_ratio=0.25,
    output_dir=cluster_output_dir,
    core_points_only=False,
)

print(f'cluster_sizes: {cluster_sizes}')
print(f'noise_points: {n_noise}')
print(f'cluster_rmsd: {cluster_rmsd}')
print(f'avg_silhouette: {silhouette_avg}')

# Choose the medoid cluster with the smallest rmsd as the native structure.
k_min = np.argmin(cluster_rmsd)

# Minimize energy of native structure
positions, PE_start, PE_end, simulation = minimize_structure(
    cgmodel,
    medoid_positions[k_min],
    output_file="native_medoid_min.dcd",
)

    