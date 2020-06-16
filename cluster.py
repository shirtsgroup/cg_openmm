import simtk.unit as unit
import mdtraj as md
import numpy as np
from sklearn.cluster import KMeans
from foldamers.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology


def concatenate_trajectories(pdb_file_list, combined_pdb_file="combined.pdb"):
    """
        Given a list of PDB files, this function reads their coordinates, concatenates them, and saves the combined coordinates to a new file (useful for clustering with MSMBuilder).

        :param pdb_file_list: A list of PDB files to read and concatenate
        :type pdb_file_list: List( str )

        :param combined_pdb_file: The name of file/path in which to save a combined version of the PDB files, default = "combined.pdb"
        :type combined_pdb_file: str

        :returns:
          - combined_pdb_file ( str ) - The name/path for a file within which the concatenated coordinates will be written.

        """
    traj_list = []
    for pdb_file in pdb_file_list:
        traj = md.load(pdb_file)
        traj_list.append(traj)
    return combined_pdb_file


def get_cluster_centroid_positions(pdb_file_list, cgmodel, n_clusters=2, frame_start=0, frame_stride=1, frame_end=-1):
    """
    Given a PDB file and coarse grained model as input, this function performs K-means clustering on the poses in the PDB file, and returns a list of the coordinates for the "centroid" pose of each cluster.

    :param pdb_file_list: A list of PDB files to read and concatenate
    :type pdb_file_list: List( str )

    :param cgmodel: A CGModel() class object
    :type cgmodel: class

    :param n_clusters: The number of "bins" within which to cluster the poses from the input trajectory.
    :type n_clusters: int

    :param frame_start: First frame in pdb trajectory file to use for clustering.
    :type frame_start: int

    :param frame_stride: Advance by this many frames when reading pdb trajectories.
    :type frame_stride: int

    :param frame_end: Last frame in pdb trajectory file to use for clustering.
    :type frame_end: int

    :returns:
    - centroid_positions ( List ( np.array( float * unit.angstrom ( num_particles x 3 ) ) ) ) - A list of the poses corresponding to the centroids of all trajectory clusters.
    - medoid_positions ( List ( np.array( float * unit.angstrom ( num_particles x 3 ) ) ) ) - A list of the poses corresponding to the medoids of all trajectory clusters.

    """

    # Load files as {replica number: replica trajectory}
    rep_traj = {}
    for i in range(len(pdb_file_list)):
        rep_traj[i] = md.load(pdb_file_list[i])

    # Combine all trajectories, selecting specified frames

    if frame_end == -1:
        frame_end = rep_traj[0].n_frames

    if frame_start == -1:
        frame_start == frame_end
        # ***Some more error handling needed here.

    traj_all = rep_traj[0][frame_start:frame_end:frame_stride]

    for i in range(len(pdb_file_list)-1):
        traj_all = traj_all.join(rep_traj[i+1][frame_start:frame_end:frame_stride])

    # Align structures with first frame as reference:
    for i in range(1,traj_all.n_frames):
        md.Trajectory.superpose(traj_all[i],traj_all[0])
        # This rewrites to traj_all

    # Compute pairwise rmsd:
    distances = np.empty((traj_all.n_frames, traj_all.n_frames))
    for i in range(traj_all.n_frames):
        distances[i] = md.rmsd(traj_all, traj_all, i)

    # Cluster with sklearn KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(distances)
    # ***In the future we could generalize to any clustering algorithm, guess number of
    # optimal clusters.

    # Get indices of frames in each cluster:
    cluster_indices = {}
    for k in range(n_clusters):
        cluster_indices[k] = np.argwhere(kmeans.labels_==k)[:,0]

    # Find the structure closest to each center (medoids):
    dist_to_centroids = KMeans.transform(kmeans,distances)
    closest_indices = np.argmin(dist_to_centroids,axis=0)
    closest_xyz = np.zeros([n_clusters,traj_all.n_atoms,3])
    for k in range(n_clusters):
        closest_xyz[k,:,:] = traj_all[closest_indices[k]].xyz[0]

    # Compute the average coordinates (centroid) in each cluster
    avg_xyz = np.zeros([n_clusters,traj_all.n_atoms,3])

    for k in range(n_clusters):
        for i in range(len(cluster_indices[k])):
            avg_xyz[k,:,:] += traj_all[cluster_indices[k][i]].xyz[0]
        avg_xyz[k,:,:] /= len(cluster_indices[k])

    # Write medoids/centroids to file
        
    for k in range(n_clusters):
        positions = avg_xyz[k,:,:] * unit.nanometer
        cgmodel.positions = positions
        file_name = str("centroid_" + str(k) + ".pdb")
        write_pdbfile_without_topology(cgmodel, file_name)

    for k in range(n_clusters):
        positions = closest_xyz[k] * unit.nanometer
        cgmodel.positions = positions
        file_name = str("medoid_" + str(k) + ".pdb")
        write_pdbfile_without_topology(cgmodel, file_name)

    centroid_positions = avg_xyz
    medoid_positions = closest_xyz
        
    return (centroid_positions, medoid_positions)


def align_structures(reference_traj, target_traj):
    """
        Given a reference trajectory, this function performs a structural alignment for a second input trajectory, with respect to the reference.

        :param reference_traj: The trajectory to use as a reference for alignment.
        :type reference_traj: `MDTraj() trajectory <http://mdtraj.org/1.6.2/api/generated/mdtraj.Trajectory.html>`_

        :param target_traj: The trajectory to align with the reference.
        :type target_traj: `MDTraj() trajectory <http://mdtraj.org/1.6.2/api/generated/mdtraj.Trajectory.html>`_

        :returns:
          - aligned_target_traj ( `MDTraj() trajectory <http://mdtraj.org/1.6.2/api/generated/mdtraj.Trajectory.html>`_ ) - The coordinates for the aligned trajectory.

        """

    aligned_target_traj = target_traj.superpose(reference_traj)

    return aligned_target_traj
