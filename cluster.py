import simtk.unit as unit
from msmbuilder.cluster.kcenters import KCenters
import mdtraj as md
from foldamers.utilities.iotools import write_pdbfile_without_topology

def concatenate_trajectories(pdb_file_list,combined_pdb_file="combined.pdb"):
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
        return(combined_pdb_file)

def get_cluster_centroid_positions(pdb_file,cgmodel,n_clusters=None):
        """
        Given a PDB file and coarse grained model as input, this function performs K-means clustering on the poses in the PDB file, and returns a list of the coordinates for the "centroid" pose of each cluster.

        :param pdb_file: The path/name of a file from which to read trajectory data
        :type pdb_file: str

        :param cgmodel: A CGModel() class object
        :type cgmodel: class

        :param n_clusters: The number of "bins" within which to cluster the poses from the input trajectory.
        :type n_clusters: int

        :returns:
          - centroid_positions ( List ( np.array( float * unit.angstrom ( num_particles x 3 ) ) ) ) - A list of the poses corresponding to the centroids of all trajectory clusters.

        """
        centroid_positions = []
        traj = md.load(pdb_file)
        print(traj.n_frames)
        exit()
        if n_clusters == None:
          n_clusters = 50
        cluster_list = KCenters(n_clusters=n_clusters,metric='rmsd')
        cluster_list.fit(traj)
        centroid_list = cluster_list.cluster_centers_
        centroid_index = 1
        for centroid in centroid_list:
          positions = centroid.xyz[0] * unit.nanometer
          cgmodel.positions = positions
          file_name = str("centroid_"+str(centroid_index)+".xyz")
          write_pdbfile_without_topology(cgmodel,file_name)
          centroid_positions.append(positions)
        return(centroid_positions)

def align_structures(reference_traj,target_traj):
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

        return(aligned_target_traj)

