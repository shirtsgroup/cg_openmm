from simtk import unit
from msmbuilder.cluster.kcenters import KCenters
import mdtraj as md
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.src.utilities.iotools import write_pdbfile_without_topology

def concatenate_trajectories(pdb_file_list,combined_pdb_file=None):
        """
        """
        if combined_pdb_file == None:
          combined_pdb_file = "combined.pdb"
        file = open(combined_pdb_file,mode="w")
        for pdb_file in pdb_file_list:
          pdb_obj = PDBFile(pdb_file)
          frame=0
          success = True
          while success:
            try:
              positions = pdb_obj.getPositions(frame=frame)
              topology = pdb_obj.getTopology()
              PDBFile.writeFile(topology,positions,file=file)
              frame = frame + 1
            except:
              success = False
        file.close()
        return(combined_pdb_file)

def get_cluster_centroid_positions(pdb_file,cgmodel,n_clusters=None):
        """
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
        """

        aligned_target_traj = target_traj.superpose(reference_traj)

        return(aligned_target_traj)

