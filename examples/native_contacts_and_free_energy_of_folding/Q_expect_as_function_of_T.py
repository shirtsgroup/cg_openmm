###
#
# This script can be used to determine suitable
# settings for calculating 'native contacts'.
#
###

import os
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from simtk import unit
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.parameters.free_energy import *
from cg_openmm.parameters.reweight import *
from cg_openmm.utilities.util import fit_sigmoid
from analyze_foldamers.ensembles.cluster import *
import pickle
import time

# Replica exchange settings
output_directory = "output"
output_data = os.path.join(output_directory, "output.nc")
number_replicas = 36
min_temp = 100.0 * unit.kelvin
max_temp = 500.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

# Load in cgmodel
cgmodel = pickle.load(open( "stored_cgmodel.pkl", "rb" ))

# Create list of trajectory files for clustering analysis
number_replicas = 36
dcd_file_list_rep = []

# For expectation native contact fraction calculation, we use replica trajectories,
# which can be generated from a .nc file using rep_exch.make_replica_dcd_files()

for i in range(number_replicas):
    dcd_file_list_rep.append("output/replica_%s.dcd" %(i+1))

# Here we will estimate the native structure as a medoid from DBSCAN clustering
# with the lowest intra-cluster RMSD

# DBSCAN parameters:
min_samples=10
eps = 0.1

medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = get_cluster_medoid_positions_DBSCAN(
    file_list=dcd_file_list_rep,
    cgmodel=cgmodel,
    min_samples=min_samples,
    eps=eps,
    frame_start=10000,
    frame_stride=100,
    frame_end=frame_end,
    filter_ratio=0.05
    )
    
print(f"min_samples: {min_samples}")
print(f"cluster sizes: {cluster_sizes}")
print(f"noise points: {n_noise}")
print(f"intra-cluster rmsd: {cluster_rmsd}")
print(f"fraction noise: {n_noise/(np.sum(cluster_sizes)+n_noise)}")
print(f"average silhouette score: {silhouette_avg}")

# Choose the medoid cluster with the smallest rmsd as the native structure.

k_min = np.argmin(cluster_rmsd)
native_structure_file=f"cluster_output/medoid_{k_min}.dcd"

# We can also choose to run an energy minimization of the native structure
# positions, PE_start, PE_end, simulation = minimize_structure(
    # cgmodel,
    # medoid_positions[k_min],
    # output_file=f"medoid_min_kmin_{k_min}.dcd",
# )

# print(f"PE start: {PE_start} kJ/mol")
# print(f"PE_end: {PE_end} kJ/mol" )

# Optimize native contact parameters:

(native_contact_cutoff, native_contact_cutoff_ratio, opt_results, Q_expect_results, \
sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut(
    cgmodel, native_structure_file, dcd_file_list_rep, num_intermediate_states=1,
    frame_begin=frame_start, frame_stride=frame_stride, opt_method='Nelder-mead')  
    