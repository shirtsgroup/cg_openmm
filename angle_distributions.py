import os
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel

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
    
    # Angle types:
    # 1 bb-bb-bb
    # 2 bb-bb-sc (sc-bb-bb)
    # 3 bb-sc-sc (sc-sc-bb)
    # 4 sc-sc-sc
    
    # Assign angle types:
    
    for i in range(len(angle_list)):
        ang_array[i,0] = angle_list[i][0]
        ang_array[i,1] = angle_list[i][1]
        ang_array[i,2] = angle_list[i][2]
        at1_type = CGModel.get_particle_type(cgmodel,angle_list[i][0])
        at2_type = CGModel.get_particle_type(cgmodel,angle_list[i][1])
        at3_type = CGModel.get_particle_type(cgmodel,angle_list[i][2])
        
        if at1_type == 'backbone': # bb-X-X
            if at2_type == 'backbone': # bb-bb-X
                if at3_type == 'backbone': # bb-bb-bb
                    ang_types.append(1)
                    
                elif at3_type == 'sidechain': # bb-bb-sc
                    ang_types.append(2)
                    
            elif at2_type == 'sidechain': # bb-sc-X
                ang_types.append(3) # bb-sc-sc
                
        elif at1_type == 'sidechain': # sc-X-X
            if at2_type == 'backbone': # sc-bb-X
                ang_types.append(2) # sc-bb-bb
                
            elif at2_type == 'sidechain': # sc-sc-X
                if at3_type == 'backbone': # sc-sc-bb
                    ang_types.append(3)
                    
                elif at3_type == 'sidechain': # sc-sc-sc
                    ang_types.append(4)
                    
    # Sort angles by type into separate arrays for mdtraj compute_angles
    # There's probably a more efficient way to do this
    ang1_array = np.zeros((ang_types.count(1),3))
    ang2_array = np.zeros((ang_types.count(2),3))
    ang3_array = np.zeros((ang_types.count(3),3))
    ang4_array = np.zeros((ang_types.count(4),3))

    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    
    for i in range(len(angle_list)):
        if ang_types[i] == 1:
            ang1_array[n1,:] = ang_array[i,:]
            n1 += 1
        if ang_types[i] == 2:
            ang2_array[n2,:] = ang_array[i,:]
            n2 += 1
        if ang_types[i] == 3:
            ang3_array[n3,:] = ang_array[i,:]
            n3 += 1
        if ang_types[i] == 4:
            ang4_array[n4,:] = ang_array[i,:]    
            n4 += 1
    
    # Save sub_arrays for each angle type to dictionary
    ang_sub_arrays = {}
    if n1 > 0:
        ang_sub_arrays["bb_bb_bb"]=ang1_array
    if n2 > 0:
        ang_sub_arrays["bb_bb_sc"]=ang2_array
    if n3 > 0:
        ang_sub_arrays["bb_sc_sc"]=ang3_array    
    if n4 > 0:
        ang_sub_arrays["sc_sc_sc"]=ang4_array        
    
    counts = [n1, n2, n3, n4]
    type_str = ["bb_bb_bb","bb_bb_sc","bb_sc_sc","sc_sc_sc"]
    
    # Create dictionary for saving angle histogram data:
    angle_hist_data = {}
    
    # Set bin edges:
    angle_bin_edges = np.linspace(0,180,nbins+1)
    angle_bin_centers = np.zeros((len(angle_bin_edges)-1,1))
    for i in range(len(angle_bin_edges)-1):
        angle_bin_centers[i] = (angle_bin_edges[i]+angle_bin_edges[i+1])/2
            
    for type in range(4):
        if counts[type] > 0:
            # Compute all angle values in trajectory
            # This returns an [nframes x n_angles] array
            ang_val_array = md.compute_angles(traj,ang_sub_arrays[type_str[type]])
            
            # Reshape arrays and convert to degrees:  
            ang_val_array = (180/np.pi)*np.reshape(ang_val_array, (nframes*counts[type],1))
            
            # Histogram and plot results:
            n_out, bin_edges_out = np.histogram(
                ang_val_array, bins=angle_bin_edges,density=True)
            pyplot.plot(angle_bin_centers,n_out,'o-r',linewidth=1,markersize=6)    
            pyplot.xlim(0,180)
            pyplot.title(f"Angle distribution for type ({type_str[type]})")
            pyplot.xlabel("Bond angle (degrees)")
            pyplot.ylabel("Probability density")
            pyplot.savefig(f"{plotfile}_{type_str[type]}.pdf")
            pyplot.close()
            angle_hist_data[f"{type_str[type]}_density"]=n_out
            angle_hist_data[f"{type_str[type]}_bin_edges"]=bin_edges_out
        
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
    
    # Torsion types:
    # 1 bb-bb-bb-bb
    # 2 bb-bb-bb-sc (sc-bb-bb-bb)
    # 3 bb-bb-sc-sc (sc-sc-bb-bb)
    # 4 bb-sc-sc-sc (sc-sc-sc-bb)
    # 5 sc-sc-sc-sc
    
    # Assign torsion types:
    
    for i in range(len(torsion_list)):
        torsion_array[i,0] = torsion_list[i][0]
        torsion_array[i,1] = torsion_list[i][1]
        torsion_array[i,2] = torsion_list[i][2]
        torsion_array[i,3] = torsion_list[i][3]
        at1_type = CGModel.get_particle_type(cgmodel,torsion_list[i][0])
        at2_type = CGModel.get_particle_type(cgmodel,torsion_list[i][1])
        at3_type = CGModel.get_particle_type(cgmodel,torsion_list[i][2])
        at4_type = CGModel.get_particle_type(cgmodel,torsion_list[i][3])
        
        if at1_type == 'backbone': # bb-X-X-X
            if at2_type == 'backbone': # bb-bb-X-X
                if at3_type == 'backbone': # bb-bb-bb-X
                    if at4_type == 'backbone': # bb-bb-bb-bb
                        torsion_types.append(1)
                        
                    elif at4_type == 'sidechain': # bb-bb-bb-sc
                        torsion_types.append(2)
                        
                elif at3_type == 'sidechain': # bb-bb-sc-X
                    torsion_types.append(3) # bb-bb-sc-sc
                    
            elif at2_type == 'sidechain': # bb-sc-X-X
                torsion_types.append(4) # bb-sc-sc-sc
                
        elif at1_type == 'sidechain': # sc-X-X-X
            if at2_type == 'backbone': # sc-bb-X-X
                torsion_types.append(2) # sc-bb-bb-bb
                
            elif at2_type == 'sidechain': # sc-sc-X-X
                if at3_type == 'backbone': # sc-sc-bb-X
                    torsion_types.append(3) # sc-sc-bb-bb
                    
                elif at3_type == 'sidechain': # sc-sc-sc-X
                    if at4_type == 'backbone': # sc-sc-sc-bb
                        torsion_types.append(4)
                        
                    elif at4_type == 'sidechain': # sc-sc-sc-sc
                        torsion_types.append(5)
                    
    # Sort torsions by type into separate arrays for mdtraj compute_torsions
    # There's probably a more efficient way to do this
    torsion1_array = np.zeros((torsion_types.count(1),4))
    torsion2_array = np.zeros((torsion_types.count(2),4))
    torsion3_array = np.zeros((torsion_types.count(3),4))
    torsion4_array = np.zeros((torsion_types.count(4),4))
    torsion5_array = np.zeros((torsion_types.count(5),4))

    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0

    for i in range(len(torsion_list)):
        if torsion_types[i] == 1:
            torsion1_array[n1,:] = torsion_array[i,:]
            n1 += 1
        elif torsion_types[i] == 2:
            torsion2_array[n2,:] = torsion_array[i,:]
            n2 += 1
        elif torsion_types[i] == 3:
            torsion3_array[n3,:] = torsion_array[i,:]
            n3 += 1
        elif torsion_types[i] == 4:
            torsion4_array[n4,:] = torsion_array[i,:]
            n4 += 1
        elif torsion_types[i] == 5:
            torsion5_array[n5,:] = torsion_array[i,:]
            n5 += 1
    
    # Save sub_arrays for each torsion type to dictionary
    torsion_sub_arrays = {}
    if n1 > 0:
        torsion_sub_arrays["bb_bb_bb_bb"]=torsion1_array
    if n2 > 0:
        torsion_sub_arrays["bb_bb_bb_sc"]=torsion2_array
    if n3 > 0:
        torsion_sub_arrays["bb_bb_sc_sc"]=torsion3_array    
    if n4 > 0:
        torsion_sub_arrays["bb_sc_sc_sc"]=torsion4_array
    if n5 > 0:
        torsion_sub_arrays["sc_sc_sc_sc"]=torsion5_array  
    
    counts = [n1, n2, n3, n4, n5]
    type_str = ["bb_bb_bb_bb","bb_bb_bb_sc","bb_bb_sc_sc","bb_sc_sc_sc","sc_sc_sc_sc"]
    
    # Create dictionary for saving torsion histogram data:
    torsion_hist_data = {}
    
    # Set bin edges:
    torsion_bin_edges = np.linspace(-180,180,nbins+1)
    torsion_bin_centers = np.zeros((len(torsion_bin_edges)-1,1))
    for i in range(len(torsion_bin_edges)-1):
        torsion_bin_centers[i] = (torsion_bin_edges[i]+torsion_bin_edges[i+1])/2
            
    for type in range(5):
        if counts[type] > 0:
            # Compute all torsion values in trajectory
            # This returns an [nframes x n_torsions] array
            torsion_val_array = md.compute_dihedrals(
                traj,torsion_sub_arrays[type_str[type]])
            
            # Reshape arrays and convert to degrees:  
            torsion_val_array = (180/np.pi)*np.reshape(torsion_val_array, (nframes*counts[type],1))
            
            # Histogram and plot results:
            n_out, bin_edges_out = np.histogram(
                torsion_val_array, bins=torsion_bin_edges,density=True)
            pyplot.plot(torsion_bin_centers,n_out,'o-b',linewidth=1,markersize=6)    
            pyplot.xlim(-180,180)
            pyplot.title(f"Torsion distribution for type ({type_str[type]})")
            pyplot.xlabel("Torsion angle (degrees)")
            pyplot.ylabel("Probability density")
            pyplot.savefig(f"{plotfile}_{type_str[type]}.pdf")
            pyplot.close()
            torsion_hist_data[f"{type_str[type]}_density"]=n_out
            torsion_hist_data[f"{type_str[type]}_bin_edges"]=bin_edges_out
        
    return torsion_hist_data    