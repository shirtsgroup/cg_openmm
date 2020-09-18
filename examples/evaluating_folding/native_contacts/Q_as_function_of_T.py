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
from analyze_foldamers.ensembles.cluster import *
import pickle
import time

# Replica exchange settings
output_directory = "output"
output_data = os.path.join(output_directory, "output.nc")
number_replicas = 36
min_temp = 50.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

# Load in cgmodel
cgmodel = pickle.load(open( "stored_cgmodel.pkl", "rb" ))

# Here we will estimate the native structure as a medoid from KMeans clustering
# with the lowest intra-cluster RMSD.

# Create list of trajectory files for clustering analysis
pdb_file_list_state = []

# For standard native contact fraction calculation, we use state trajectories,
# which can be generated from a .nc file using rep_exch.make_state_pdb_files()
for i in range(number_replicas):
    pdb_file_list_state.append(f"{output_directory}/state_{i+1}.pdb")

# Set clustering parameters
n_clusters=2
frame_start=0
frame_stride=100
frame_end=-1

# Run KMeans clustering
medoid_positions, cluster_size, cluster_rmsd = get_cluster_medoid_positions(
    file_list=pdb_file_list_state,
    cgmodel=cgmodel,
    n_clusters=n_clusters,
    frame_start=frame_start,
    frame_stride=frame_stride,
    frame_end=-1)
    
print(cluster_rmsd)

# The smaller intra-cluster rmsd will be the folded state if it is very stable
if cluster_rmsd[0] < cluster_rmsd[1]:
    native_structure_file="cluster_output/medoid_0.pdb"
else:
    native_structure_file="cluster_output/medoid_1.pdb"

native_positions = PDBFile(native_structure_file).getPositions()


# Scan cutoff for native structure pairwise distances:
for c in [2.5, 3.0, 3.5, 4.0, 4.5]:

    native_contact_cutoff = c* unit.angstrom
    nccut_str = f"{str(native_contact_cutoff)[0]}_{str(native_contact_cutoff)[2]}"

    # Cutoff for current trajectory distances, as a multiple of native_contact_cutoff
    native_contact_cutoff_ratio = 1.00

    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict, decorrelation_spacing = get_native_contacts(
        cgmodel,
        native_positions,
        native_contact_cutoff
    )

    # Determine native contact fraction of current trajectories:

    Q, Q_avg, Q_stderr = fraction_native_contacts(
        pdb_file_list_state,
        native_contact_list,
        native_contact_distances,
        frame_begin=0,
        native_contact_cutoff_ratio=native_contact_cutoff_ratio
    )
    
    plot_native_contact_fraction(
        temperature_list,
        Q_avg,
        Q_stderr,
        plotfile=f"{output_directory}/nccut_{nccut_str}_Q_vs_T.pdf",
    )