import os
import numpy as np
import mdtraj as md
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.utilities.plot import plot_distribution

# These functions calculate and plot bond angle and torsion distributions from a CGModel object and pdb trajectory

def calc_bond_angle_distribution(
    cgmodel,pdbfile,nbins=90,plotfile="angle_hist"
):
    """
    Calculate and plot all bond angle distributions from a CGModel object and pdb trajectory

    :param cgmodel: CGModel() object
    :type cgmodel: class
    
    :param pdbfile: path to pdb trajectory file
    :type pdbfile: str
    
    :param nbins: number of bins spanning the range of 0 to 180 degrees, default = 90
    :type nbins: int
    
    :param plotfile: Base filename for saving bond angle distribution pdf plots
    :type plotfile: str
    
    """
    
    # Load in a trajectory pdb file:
    traj = md.load(pdbfile)
    nframes = traj.n_frames
    
    # Get angle list
    angle_list = CGModel.get_bond_angle_list(cgmodel)
    
    ang_types = [] # List of angle types for each angle in angle_list
    
    ang_array = np.zeros((len(angle_list),3))
    
    # Relevant angle types are added to a dictionary as they are discovered 
    ang_dict = {}
    
    # Create an inverse dictionary for getting angle string name from integer type
    inv_ang_dict = {}
    
    # Counter for number of angle types found:
    i_angle_type = 0
    
    # Assign angle types:
    
    for i in range(len(angle_list)):
        ang_array[i,0] = angle_list[i][0]
        ang_array[i,1] = angle_list[i][1]
        ang_array[i,2] = angle_list[i][2]
        
        particle_types = [
            CGModel.get_particle_type_name(cgmodel,angle_list[i][0]),
            CGModel.get_particle_type_name(cgmodel,angle_list[i][1]),
            CGModel.get_particle_type_name(cgmodel,angle_list[i][2])
        ]
        
        string_name = ""
        reverse_string_name = ""
        for particle in particle_types:
            string_name += f"{particle}_"
        string_name = string_name[:-1]
        for particle in reversed(particle_types):
            reverse_string_name += f"{particle}_"
        reverse_string_name = reverse_string_name[:-1]
            
        if (string_name in ang_dict.keys()) == False:
            # New angle type found, add to angle dictionary
            i_angle_type += 1
            ang_dict[string_name] = i_angle_type
            ang_dict[reverse_string_name] = i_angle_type
            # For inverse dict we will use only the forward name based on first encounter
            inv_ang_dict[str(i_angle_type)] = string_name
            print(f"adding new angle type {i_angle_type}: {string_name} to dictionary")
            print(f"adding reverse version {i_angle_type}: {reverse_string_name} to dictionary")
            
        ang_types.append(ang_dict[string_name])
                    
    # Sort angles by type into separate sub arrays for mdtraj compute_angles
    ang_sub_arrays = {}
    for i in range(i_angle_type):
        ang_sub_arrays[str(i+1)] = np.zeros((ang_types.count(i+1),3))
    
    # Counter vector for all angle types
    n_i = np.zeros((i_angle_type,1), dtype=int)
    
    for i in range(len(angle_list)):
        ang_sub_arrays[str(ang_types[i])][n_i[ang_types[i]-1],:] = ang_array[i,:]
        n_i[ang_types[i]-1] += 1
         
    # Create dictionary for saving angle histogram data:
    angle_hist_data = {}
    
    # Set bin edges:
    angle_bin_edges = np.linspace(0,180,nbins+1)
    angle_bin_centers = np.zeros((len(angle_bin_edges)-1,1))
    for i in range(len(angle_bin_edges)-1):
        angle_bin_centers[i] = (angle_bin_edges[i]+angle_bin_edges[i+1])/2
            
    for i in range(i_angle_type):
        # Compute all angle values in trajectory
        # This returns an [nframes x n_angles] array
        ang_val_array = md.compute_angles(traj,ang_sub_arrays[str(i+1)])
        
        # Reshape arrays and convert to degrees:  
        ang_val_array = (180/np.pi)*np.reshape(ang_val_array, (nframes*n_i[i][0],1))
        
        # Histogram and plot results:
        
        n_out, bin_edges_out = np.histogram(
            ang_val_array, bins=angle_bin_edges,density=True)
            
        
        angle_hist_data[f"{inv_ang_dict[str(i+1)]}_density"]=n_out
        angle_hist_data[f"{inv_ang_dict[str(i+1)]}_bin_centers"]=angle_bin_centers
        
    plot_distribution(
        inv_ang_dict,
        angle_hist_data,
        xlabel="Bond angle (degrees)",
        ylabel="Probability density",
        xlim=[0,180],
        figure_title="Angle distributions",
        file_name=f"{plotfile}",
        marker_string='o-r',
    )
        
    return angle_hist_data

    
def calc_torsion_distribution(
    cgmodel,pdbfile,nbins=180,plotfile="torsion_hist"
):
    """
    Calculate and plot all torsion distributions from a CGModel object and pdb trajectory

    :param cgmodel: CGModel() object
    :type cgmodel: class
    
    :param pdbfile: path to pdb trajectory file
    :type pdbfile: str
    
    :param nbins: number of bins spanning the range of -180 to 180 degrees, default = 180
    :type nbins: int
    
    :param plotfile: Base filename for saving torsion distribution pdf plots
    :type plotfile: str
    
    """
    
    # Load in a trajectory pdb file:
    traj = md.load(pdbfile)
    nframes = traj.n_frames
    
    # Get torsion list
    torsion_list = CGModel.get_torsion_list(cgmodel)
    torsion_types = [] # List of torsion types for each torsion in torsion_list
    torsion_array = np.zeros((len(torsion_list),4))
    
    # Relevant torsion types are added to a dictionary as they are discovered 
    torsion_dict = {}
    
    # Create an inverse dictionary for getting torsion string name from integer type
    inv_torsion_dict = {}
    
    # Counter for number of torsion types found:
    i_torsion_type = 0    
    
    # Assign torsion types:
    
    for i in range(len(torsion_list)):
        torsion_array[i,0] = torsion_list[i][0]
        torsion_array[i,1] = torsion_list[i][1]
        torsion_array[i,2] = torsion_list[i][2]
        torsion_array[i,3] = torsion_list[i][3]
        
        particle_types = [
            CGModel.get_particle_type_name(cgmodel,torsion_list[i][0]),
            CGModel.get_particle_type_name(cgmodel,torsion_list[i][1]),
            CGModel.get_particle_type_name(cgmodel,torsion_list[i][2]),
            CGModel.get_particle_type_name(cgmodel,torsion_list[i][3])
        ]
        
        string_name = ""
        reverse_string_name = ""
        for particle in particle_types:
            string_name += f"{particle}_"
        string_name = string_name[:-1]
        for particle in reversed(particle_types):
            reverse_string_name += f"{particle}_"
        reverse_string_name = reverse_string_name[:-1]
            
        if (string_name in torsion_dict.keys()) == False:
            # New torsion type found, add to torsion dictionary
            i_torsion_type += 1
            torsion_dict[string_name] = i_torsion_type
            torsion_dict[reverse_string_name] = i_torsion_type
            # For inverse dict we will use only the forward name based on first encounter
            inv_torsion_dict[str(i_torsion_type)] = string_name
            
            print(f"adding new torsion type {i_torsion_type}: {string_name} to dictionary")
            print(f"adding reverse version {i_torsion_type}: {reverse_string_name} to dictionary")
            
            
        torsion_types.append(torsion_dict[string_name])
                        
    # Sort torsions by type into separate sub arrays for mdtraj compute_dihedrals
    torsion_sub_arrays = {}
    for i in range(i_torsion_type):
        torsion_sub_arrays[str(i+1)] = np.zeros((torsion_types.count(i+1),4))
    
    # Counter vector for all angle types
    n_i = np.zeros((i_torsion_type,1), dtype=int) 
    
    for i in range(len(torsion_list)):
        torsion_sub_arrays[str(torsion_types[i])][n_i[torsion_types[i]-1],:] = torsion_array[i,:]
        n_i[torsion_types[i]-1] += 1
    
    # Create dictionary for saving torsion histogram data:
    torsion_hist_data = {}
    
    # Set bin edges:
    torsion_bin_edges = np.linspace(-180,180,nbins+1)
    torsion_bin_centers = np.zeros((len(torsion_bin_edges)-1,1))
    for i in range(len(torsion_bin_edges)-1):
        torsion_bin_centers[i] = (torsion_bin_edges[i]+torsion_bin_edges[i+1])/2
        
    for i in range(i_torsion_type):
        # Compute all torsion values in trajectory
        # This returns an [nframes x n_torsions] array
        torsion_val_array = md.compute_dihedrals(
            traj,torsion_sub_arrays[str(i+1)])
        
        # Reshape arrays and convert to degrees:  
        torsion_val_array = (180/np.pi)*np.reshape(torsion_val_array, (nframes*n_i[i][0],1))
        
        # Histogram and plot results:
        n_out, bin_edges_out = np.histogram(
            torsion_val_array, bins=torsion_bin_edges,density=True)
        
        torsion_hist_data[f"{inv_torsion_dict[str(i+1)]}_density"]=n_out
        torsion_hist_data[f"{inv_torsion_dict[str(i+1)]}_bin_centers"]=torsion_bin_centers  
      
    plot_distribution(
        inv_torsion_dict,
        torsion_hist_data,
        xlabel="Torsion angle (degrees)",
        ylabel="Probability density",
        xlim=[-180,180],
        figure_title="Torsion_distributions",
        file_name=f"{plotfile}",
        marker_string='o-b',
    )
      
    return torsion_hist_data    