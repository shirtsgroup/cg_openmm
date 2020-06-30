import os
from simtk import unit
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from foldamers.cg_model.cgmodel import CGModel
from foldamers.ensembles.cluster import *
import pickle

# Load in cgmodel
cgmodel = pickle.load(open( "stored_cgmodel.pkl", "rb" ))

# Create list of trajectory files for clustering analysis
number_replicas = 48
pdb_file_list = []
for i in range(number_replicas):
    pdb_file_list.append("output/replica_%s.pdb" %(i+1))

# Set clustering parameters
n_clusters=4
frame_start=0
frame_stride=10
frame_end=-1

# Run KMeans clustering
medoid_positions, cluster_size, cluster_rmsd = get_cluster_centroid_positions(
    pdb_file_list=pdb_file_list,
    cgmodel=cgmodel,
    n_clusters=n_clusters,
    frame_start=frame_start,
    frame_stride=frame_stride,
    frame_end=-1)

print('\nCluster sizes:')
print(cluster_size)
print('\nCluster rmsd:')
print(cluster_rmsd)