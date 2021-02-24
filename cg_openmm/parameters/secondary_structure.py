import os, subprocess
import numpy as np
import simtk.unit as unit
from statistics import mean
from scipy.stats import linregress
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cg_openmm.utilities.random_builder import *
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeAnalyzer
import pymbar
from pymbar import timeseries
import mdtraj as md
from scipy.optimize import minimize, Bounds, brute, differential_evolution
from scipy.special import erf
from scipy.optimize import minimize_scalar
from cg_openmm.utilities.util import fit_sigmoid
from sklearn.utils import resample

kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant

def get_native_contacts(cgmodel, native_structure_file, native_contact_distance_cutoff):
    """
    Given a coarse grained model, positions for the native structure, and cutoff, this function determines which pairs
    are native contacts.

    :param cgmodel: CGModel() class object
    :type cgmodel: class

    :param native_structure_file: Path to file ('pdb' or 'dcd') containing particle positions for the native structure.
    :type native_structure_file: str

    :param native_contact_distance_cutoff: The maximum distance for two nonbonded particles that are defined as "native",default=None
    :type native_contact_distance_cutoff: `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

    :returns:
       - native_contact_list - A list of the nonbonded interactions whose inter-particle distances are less than the 'native_contact_cutoff_distance'.
       - native_contact_distances - A Quantity numpy array of the native pairwise distances corresponding to native_contact_list
       - contact_type_dict - A dictionary of {native contact particle type pair: counts}
    """

    # Parse native structure file
    if native_structure_file[-3:] == 'dcd':
        native_traj = md.load(native_structure_file,top=md.Topology.from_openmm(cgmodel.topology))
        # ***Note: The clustering dcds are written with unit nanometers,
        # but this may not always be the case.
        native_structure = native_traj[0].xyz[0]*unit.nanometer
    else:
        native_structure = PDBFile(native_structure_file).getPositions()
        
    # Include only pairs contained in the nonbonded_interaction_list    
    nonbonded_interaction_list = cgmodel.nonbonded_interaction_list
    native_structure_distances = distances(nonbonded_interaction_list, native_structure)
    native_contact_list = []
    native_contact_distances_list = []
    
    for interaction in range(len(nonbonded_interaction_list)):
        if native_structure_distances[interaction] < (native_contact_distance_cutoff):
            native_contact_list.append(nonbonded_interaction_list[interaction])
            native_contact_distances_list.append(distances(native_contact_list, native_structure))
    
    # Units get messed up if converted using np.asarray
    native_contact_distances = np.zeros((len(native_contact_distances_list)))
    for i in range(len(native_contact_distances_list)):
        native_contact_distances[i] = native_contact_distances_list[i][0].value_in_unit(unit.nanometer)
    native_contact_distances *= unit.nanometer
    
    # Determine particle types of the native contacts:
    # Store the numbers of contact interactions by type in dict:
    contact_type_dict = {}
    for contact in native_contact_list:
        type1 = cgmodel.get_particle_type_name(contact[0])
        type2 = cgmodel.get_particle_type_name(contact[1])
        string_name = f"{type1}_{type2}"
        reverse_string_name = f"{type2}_{type1}"
        if ((string_name in contact_type_dict.keys()) == False and 
            (reverse_string_name in contact_type_dict.keys()) == False):
            # Found a new type of contact:
            # Only store counts in forward string of first encounter
            contact_type_dict[string_name] = 1
            #print(f"adding contact type {string_name} to dict") 
        else:
            if (string_name in contact_type_dict.keys()) == True:
                # Add to forward_string count:
                contact_type_dict[string_name] += 1
            else:
                # Add to reverse string count:
                contact_type_dict[reverse_string_name] += 1
            
    return native_contact_list, native_contact_distances, contact_type_dict
    
    
def get_helix_contacts(cgmodel, native_structure_file, backbone_type_name='bb', verbose=False):
    """
    Given a coarse grained model and positions for the native structure this function determines which pairs
    are native contacts. This function assumes helical geometry with native contacts being backbone pairs
    iteracting as (i) to (i+n) neighbors, where n defines the pairs which on average are the shortest distance.

    :param cgmodel: CGModel() class object
    :type cgmodel: class

    :param native_structure_file: Path to file ('pdb' or 'dcd') containing particle positions for the native structure.
    :type native_structure_file: str

    :param backbone_type_name: type name in cgmodel which corresponds to the particles forming the helical backbone.
    :type backbone_type_name: str
    
    :returns:
       - native_contact_list - A list of the nonbonded interactions whose inter-particle distances are less than the 'native_contact_cutoff_distance'.
       - native_contact_distances - A Quantity numpy array of the native pairwise distances corresponding to native_contact_list
       - opt_seq_spacing - The (i) to (i+n) number n defining contacting backbone beads
    """

    # Parse native structure file
    if native_structure_file[-3:] == 'dcd':
        native_traj = md.load(native_structure_file,top=md.Topology.from_openmm(cgmodel.topology))
        # ***Note: The clustering dcds are written with unit nanometers,
        # but this may not always be the case.
        native_structure = native_traj[0].xyz[0]*unit.nanometer
    else:
        native_structure = PDBFile(native_structure_file).getPositions()
        
    # Include only pairs contained in the nonbonded_interaction_list
    # This is a list of lists, with inner lists being the interaction pairs
    nonbonded_interaction_list = cgmodel.nonbonded_interaction_list
    
    # Now, filter out non-backbone types and determine the number of bonds separating each.
    bb_interaction_list = []
    for pair in nonbonded_interaction_list:
        if cgmodel.get_particle_type_name(pair[0])==backbone_type_name and cgmodel.get_particle_type_name(pair[1])==backbone_type_name:
            bb_interaction_list.append(pair)
            
    # From the bonds list, determine the sequence of backbone particles
    bond_list = cgmodel.get_bond_list()
    bb_bond_list = []
    bb_bond_particle_list = []
    for bond in bond_list:
        if cgmodel.get_particle_type_name(bond[0])==backbone_type_name and cgmodel.get_particle_type_name(bond[1])==backbone_type_name:
            bb_bond_list.append(bond)
            # Save the bond particles to a single list:
            bb_bond_particle_list.append(bond[0])
            bb_bond_particle_list.append(bond[1])
            
    if verbose:        
        print(f'bb_bond_list: {bb_bond_list}')        
            
    # Now determine the ordering. Find an end bead and build from there.
    bb_sequence = []
    for bead in bb_bond_particle_list:
        if bb_bond_particle_list.count(bead) == 1:
            # End of chain found
            bb_sequence.append(bead)
            break
    
    # Find which bead is bonded to the chain end, and so on and so forth
    # We should test this on arbitrary sequences such as:
    #[6*, 4], [4,2], [2,8*]
    for i in range(len(bb_bond_list)):
        for bond in bb_bond_list:
            if bond[0] == bb_sequence[-1]:
                # Check if this is a new pair:
                if bond[1] not in bb_sequence:
                    bb_sequence.append(bond[1])
                    break
            elif bond[1] == bb_sequence[-1]:
                # Check if this is a new pair:
                if bond[0] not in bb_sequence:
                    bb_sequence.append(bond[0])
                    break
               
    if verbose:           
        print(f'bb sequence: {bb_sequence}')            
                                
    # Now compute the relevant distances for (i) to (i+k):
    ik_pairs = {}
    ik_dist_arr = {}
    seq_spacing = [3,4,5,6,7]
    i_mean_dist_array = np.zeros(5)
    j = 0
    
    for k in seq_spacing:
        ik_pairs[f'i{k}'] = []
        for i in range(len(bb_sequence)-k):
            ik_pairs[f'i{k}'].append([bb_sequence[i],bb_sequence[i+k]])
        ik_dist_list = distances(ik_pairs[f'i{k}'], native_structure)
        ik_dist_arr[f'i{k}'] = np.zeros(len(ik_dist_list))
        for i in range(len(ik_dist_list)):
            ik_dist_arr[f'i{k}'][i] = ik_dist_list[i].value_in_unit(unit.nanometer)
        i_mean_dist_array[j] = np.mean(ik_dist_arr[f'i{k}'])
        j += 1
    
    if verbose:
        print(f"i to i+3: {ik_dist_arr['i3']}")
        print(f"i to i+4: {ik_dist_arr['i4']}")
        print(f"i to i+5: {ik_dist_arr['i5']}")
        print(f"i to i+6: {ik_dist_arr['i6']}")
        print(f"i to i+7: {ik_dist_arr['i7']}")
        print(f'i_mean_dist_array: {i_mean_dist_array}')
    
    opt_seq_spacing = seq_spacing[np.argmin(i_mean_dist_array)]
    native_contact_list = ik_pairs[f'i{opt_seq_spacing}']
    native_contact_distances = ik_dist_arr[f'i{opt_seq_spacing}']*unit.nanometer
       
    return native_contact_list, native_contact_distances, opt_seq_spacing


def expectations_fraction_contacts(fraction_native_contacts, frame_begin=0, sample_spacing=1,
    output_data="output/output.nc", num_intermediate_states=0, bootstrap_energies=None):
    """
    Given a .nc output, a temperature list, and a number of intermediate states to insert for the temperature list, this function calculates the native contacts expectation.   
    
    :param fraction_native_contacts: The fraction of native contacts for all selected frames in the trajectories.
    :type fraction_native_contacts: numpy array (float * nframes x nreplicas)
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int

    :param sample_spacing: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData
    :type sample_spacing: int       
    
    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data, default = None                                                                                                  
    :type output_data: str
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list'
    :type num_intermediate_states: int    
    
    :param bootstrap_energies: a custom replica_energies array to be used for bootstrapping calculations. Used instead of the energies in the .nc file.
    :type bootstrap_energies: 2d numpy array (float)
    
    """

    if bootstrap_energies is not None:
        # Use a subsampled replica_energy matrix instead of reading from file
        replica_energies = bootstrap_energies    
        # Still need to get the thermodynamic states
        reporter = MultiStateReporter(output_data, open_mode="r")
    else:
        # extract reduced energies and the state indices from the .nc  
        reporter = MultiStateReporter(output_data, open_mode="r")
        analyzer = ReplicaExchangeAnalyzer(reporter)
        (
            replica_energies_all,
            unsampled_state_energies,
            neighborhoods,
            replica_state_indices,
        ) = analyzer.read_energies()
        
        # Select production frames to analyze
        replica_energies = replica_energies_all[:,:,frame_begin::sample_spacing]
    
    # Get the temperature list from .nc file:
    states = reporter.read_thermodynamic_states()[0]
    
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)
    
    # Check the size of the fraction_native_contacts array:
    if np.shape(replica_energies)[2] != np.shape(fraction_native_contacts)[0]:
        # Mismatch in number of frames.
        if np.shape(replica_energies_all[:,:,frame_begin::sample_spacing])[2] == np.shape(fraction_native_contacts[::sample_spacing,:])[0]:
            # Correct starting frame, need to apply sampling stride:
            fraction_native_contacts = fraction_native_contacts[::sample_spacing,:]
        elif np.shape(replica_energies_all)[2] == np.shape(fraction_native_contacts)[0]:
            # This is the full fraction_native_contacts, slice production frames:
            fraction_native_contacts = fraction_native_contacts[production_start::sample_spacing,:]

    # determine the numerical values of beta at each state in units consistent with the temperature
    Tunit = temperature_list[0].unit
    temps = np.array([temp.value_in_unit(Tunit)  for temp in temperature_list])  # should this just be array to begin with
    beta_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * temps)

    # convert the energies from replica/evaluated state/sample form to evaluated state/sample form
    replica_energies = pymbar.utils.kln_to_kn(replica_energies)
    n_samples = len(replica_energies[0,:])
    # n_samples in [nreplica x nsamples_per_replica]
    
    # calculate the number of states we need expectations at.  We want it at all of the original
    # temperatures, each intermediate temperature, and then temperatures +/- from the original
    # to take finite derivatives.

    # create  an array for the temperature and energy for each state, including the
    # finite different state.
    n_sampled_T = len(temps)
    n_unsampled_states = (n_sampled_T + (n_sampled_T-1)*num_intermediate_states)
    unsampled_state_energies = np.zeros([n_unsampled_states,n_samples])
    full_T_list = np.zeros(n_unsampled_states)

    # delta is the spacing between temperatures.
    delta = np.zeros(n_sampled_T-1)

    # fill in a list of temperatures at all original temperatures and all intermediate states.
    full_T_list[0] = temps[0]  
    t = 0
    for i in range(n_sampled_T-1):
        delta[i] = (temps[i+1] - temps[i])/(num_intermediate_states+1)
        for j in range(num_intermediate_states+1):
            full_T_list[t] = temps[i] + delta[i]*j
            t += 1
    full_T_list[t] = temps[-1]
    n_T_vals = t+1

    # calculate betas of all of these temperatures
    beta_full_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * full_T_list)
    
    ti = 0
    N_k = np.zeros(n_unsampled_states)
    for k in range(n_unsampled_states):
        # Calculate the reduced energies at all temperatures, sampled and unsample.
        unsampled_state_energies[k, :] = replica_energies[0,:]*(beta_full_k[k]/beta_k[0])
        if ti < len(temps):
            # store in N_k which states do and don't have samples.
            if full_T_list[k] == temps[ti]:
                ti += 1
                N_k[k] = n_samples//len(temps)  # these are the states that have samples

    # call MBAR to find weights at all states, sampled and unsampled
    mbarT = pymbar.MBAR(unsampled_state_energies,N_k,verbose=False, relative_tolerance=1e-12);
    
    # Now we have the weights at all temperatures, so we can
    # calculate the expectations.
    
    # Reshape fraction native contacts [nframes x nreplicas] column by column for pymbar
    Q = np.reshape(fraction_native_contacts,np.size(fraction_native_contacts), order='F')
            
    # calculate the expectation of Q at each unsampled states         
    results = mbarT.computeExpectations(Q)  # compute expectations of Q at all points
    Q_expect = results[0]
    dQ_expect = results[1]

    # return the results in a dictionary (better than in a list)
    return_results = dict()
    return_results["T"] = full_T_list*unit.kelvin
    return_results["Q"] = Q_expect
    return_results["dQ"] = dQ_expect

    return return_results

    
def fraction_native_contacts(
    cgmodel,
    file_list,
    native_contact_list,
    native_contact_distances,
    frame_begin=0,
    native_contact_tol=1.1,
    subsample=True,
):
    """
    Given a cgmodel, mdtraj trajectory object, and positions for the native structure, this function calculates the fraction of native contacts for the model.
    
    :param cgmodel: CGModel() class object
    :type cgmodel: class
        
    :param file_list: A list of replica PDB or DCD trajectory files corresponding to the energies in the .nc file, or a single file name
    :type file_list: List( str ) or str

    :param native_contact_list: A list of the nonbonded interactions whose inter-particle distances are less than the 'native_contact_cutoff_distance'.
    :type native_contact_list: List
    
    :param native_contact_distances: A numpy array of the native pairwise distances corresponding to native_contact_list
    :type native_contact_distances: Quantity

    :param frame_begin: Frame at which to start native contacts analysis (default=0)
    :type frame_begin: int        
    
    :param native_contact_tol: Tolerance factor beyond the native distance for determining whether a pair of particles is 'native' (in multiples of native distance)
    :type native_contact_tol: float
    
    :param subsample: option to use pymbar subsampleCorrelatedData to detect and return the interval between uncorrelated data points (default=True)
    :type subsample: Boolean

    :returns:
      - Q ( numpy array (float * nframes x nreplicas) ) - The fraction of native contacts for all selected frames in the trajectories.
      - Q_avg ( numpy array (float * nreplicas) ) - Mean values of Q for each replica.
      - Q_stderr ( numpy array (float * nreplicas) ) - Standard error of the mean of Q for each replica.
      - decorrelation_spacing ( int ) - Number of frames between uncorrelated native contact fractions
    
    """

    if len(native_contact_list)==0:
        print("ERROR: there are 0 'native' interactions with the current cutoff distance.")
        print("Try increasing the 'native_structure_contact_distance_cutoff'")
        exit()
        
        
    if type(file_list) == list:
        n_replicas = len(file_list)
    elif type(file_list) == str:
        n_replicas = 1
        # Convert to a 1 element list if not one
        file_list = file_list.split()
      
    nc_unit = native_contact_distances.unit
    Q_avg = np.zeros((n_replicas))
    Q_stderr = np.zeros((n_replicas))
      
    for rep in range(n_replicas):            
        # This should work for pdb or dcd
        # However for dcd we need to insert a topology, and convert it from openmm->mdtraj topology 
        if file_list[0][-3:] == 'dcd':
            rep_traj = md.load(file_list[rep],top=md.Topology.from_openmm(cgmodel.topology))
        else:
            rep_traj = md.load(file_list[rep])
        # Select frames for analysis:
        rep_traj = rep_traj[frame_begin:]
        
        if rep == 0:
            nframes = rep_traj.n_frames
            Q = np.zeros((nframes,n_replicas))
        
        traj_distances = md.compute_distances(
            rep_traj,native_contact_list,periodic=False,opt=True)
        # This produces a [nframe x len(native_contacts)] array
  
        # Compute Boolean matrix for whether or not a distance is native
        native_contact_matrix = (traj_distances<native_contact_tol*(native_contact_distances.value_in_unit(nc_unit)))

        number_native_interactions=np.sum(native_contact_matrix,axis=1)

        Q[:,rep] = number_native_interactions/len(native_contact_distances)
        Q_avg[rep] = np.mean(Q[:,rep])
        # Compute standard error:
        Q_stderr[rep] = np.std(Q[:,rep])/np.sqrt(len(Q[:,rep]))
    
    # Determine the decorrelation time of native contact fraction timeseries data:
    # Note: if these are replica trajectories, we will get a folding rate
    # If these are state trajectories, we will get decorrelation time for constant state
    
    # Assume a normal distribution (very rough approximation), and use mean plus
    # the number of standard deviations which leads to (n_replica-1)/n_replica coverage
    # For 12 replicas this should be the mean + 1.7317 standard deviations
    
    # x standard deviations is the solution to (n_replica-1)/n_replica = erf(x/sqrt(2))
    # This is equivalent to a target of 23/24 CDF value
    
    def erf_fun(x):
        return np.power((erf(x/np.sqrt(2))-(n_replicas-1)/n_replicas),2)    
    
    if subsample:
        g = np.zeros(n_replicas)
        subsample_indices = {}
        for rep in range(n_replicas):
            subsample_indices[rep] = timeseries.subsampleCorrelatedData(
                Q[:,rep],
                conservative=True,
            )
            g[rep] = (subsample_indices[rep][1]-subsample_indices[rep][0])
        
        # Determine value of decorrelation time to use  
        opt_g_results = minimize_scalar(
            erf_fun,
            bounds=(0,10)
            )
        decorrelation_spacing = int(np.ceil(np.mean(g)+opt_g_results.x*np.std(g)))
    else:
        decorrelation_spacing = None
        
    return Q, Q_avg, Q_stderr, decorrelation_spacing
    
    
def fraction_native_contacts_preloaded(
    cgmodel,
    traj_dict,
    native_contact_list,
    native_contact_distances,
    frame_begin=0,
    native_contact_tol=1.1,
    subsample=True,
):
    """
    Given a cgmodel, mdtraj trajectory object, and positions for the native structure, this function calculates the fraction of native contacts for the model.
    
    :param cgmodel: CGModel() class object
    :type cgmodel: class
        
    :param traj_data: A dictionary of preloaded MDTraj trajectory objects
    :type traj_data: dict{replica: MDTraj trajectory object}

    :param native_contact_list: A list of the nonbonded interactions whose inter-particle distances are less than the 'native_contact_cutoff_distance'.
    :type native_contact_list: List
    
    :param native_contact_distances: A numpy array of the native pairwise distances corresponding to native_contact_list
    :type native_contact_distances: Quantity

    :param frame_begin: Frame at which to start native contacts analysis (default=0)
    :type frame_begin: int        
    
    :param native_contact_tol: Tolerance factor beyond the native distance for determining whether a pair of particles is 'native' (in multiples of native distance)
    :type native_contact_tol: float
    
    :param subsample: option to use pymbar subsampleCorrelatedData to detect and return the interval between uncorrelated data points (default=True)
    :type subsample: Boolean

    :returns:
      - Q ( numpy array (float * nframes x nreplicas) ) - The fraction of native contacts for all selected frames in the trajectories.
      - Q_avg ( numpy array (float * nreplicas) ) - Mean values of Q for each replica.
      - Q_stderr ( numpy array (float * nreplicas) ) - Standard error of the mean of Q for each replica.
      - decorrelation_spacing ( int ) - Number of frames between uncorrelated native contact fractions
      
    """

    if len(native_contact_list)==0:
        print("ERROR: there are 0 'native' interactions with the current cutoff distance.")
        print("Try increasing the 'native_structure_contact_distance_cutoff'")
        exit()
        
    n_replicas = len(traj_dict)
      
    nc_unit = native_contact_distances.unit
    Q_avg = np.zeros((n_replicas))
    Q_stderr = np.zeros((n_replicas))
      
    for rep in range(n_replicas):              
        if rep == 0:
            nframes = traj_dict[rep][frame_begin:].n_frames
            Q = np.zeros((nframes,n_replicas))
        
        traj_distances = md.compute_distances(
            traj_dict[rep][frame_begin:],native_contact_list,periodic=False,opt=True)
            
        # This produces a [nframe x len(native_contacts)] array
  
        # Compute Boolean matrix for whether or not a distance is native
        native_contact_matrix = (traj_distances<native_contact_tol*(native_contact_distances.value_in_unit(nc_unit)))

        number_native_interactions=np.sum(native_contact_matrix,axis=1)

        Q[:,rep] = number_native_interactions/len(native_contact_distances)
        Q_avg[rep] = np.mean(Q[:,rep])
        # Compute standard error:
        Q_stderr[rep] = np.std(Q[:,rep])/np.sqrt(len(Q[:,rep]))
    
    # Determine the decorrelation time of native contact fraction timeseries data:
    # Note: if these are replica trajectories, we will get a folding rate
    # If these are state trajectories, we will get decorrelation time for constant state
    
    # Assume a normal distribution (very rough approximation), and use mean plus
    # the number of standard deviations which leads to (n_replica-1)/n_replica coverage
    # For 12 replicas this should be the mean + 1.7317 standard deviations
    
    # x standard deviations is the solution to (n_replica-1)/n_replica = erf(x/sqrt(2))
    # This is equivalent to a target of 23/24 CDF value
    
    def erf_fun(x):
        return np.power((erf(x/np.sqrt(2))-(n_replicas-1)/n_replicas),2)    
    
    if subsample:
        g = np.zeros(n_replicas)
        subsample_indices = {}
        for rep in range(n_replicas):
            subsample_indices[rep] = timeseries.subsampleCorrelatedData(
                Q[:,rep],
                conservative=True,
            )
            g[rep] = (subsample_indices[rep][1]-subsample_indices[rep][0])
        
        # Determine value of decorrelation time to use  
        opt_g_results = minimize_scalar(
            erf_fun,
            bounds=(0,10)
            )
        decorrelation_spacing = int(np.ceil(np.mean(g)+opt_g_results.x*np.std(g)))
    else:
        decorrelation_spacing = None
        
    return Q, Q_avg, Q_stderr, decorrelation_spacing
    
    
def optimize_Q_cut(
    cgmodel, native_structure_file, traj_file_list, output_data="output/output.nc",
    num_intermediate_states=0, frame_begin=0, frame_stride=1,
    plotfile='native_contacts_opt.pdf', verbose=False, minimizer_options=None):
    """
    Given a coarse grained model and a native structure as input, optimize the distance cutoff defining
    the native contact pairs, and the distance tolerance for scanning the trajectory for native contacts.

    :param cgmodel: CGModel() class object
    :type cgmodel: class
    
    :param native_structure_file: Path to file ('pdb' or 'dcd') containing particle positions for the native structure.
    :type native_structure_file: str 
    
    :param traj_file_list: A list of replica PDB or DCD trajectory files corresponding to the energies in the .nc file, or a single file name
    :type traj_file_list: List( str ) or str
    
    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data, default = ("output/output.nc")                                                                                                  
    :type output_data: str

    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list'
    :type num_intermediate_states: int 
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int

    :param frame_stride: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData
    :type frame_stride: int

    :param minimizer_options: dictionary of additional options for scipy.minimize.optimize.differential_evolution (default=None)
    :type minimizer: dict

    :returns:
       - native_contact_cutoff ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The ideal distance below which two nonbonded, interacting particles should be defined as a "native contact"
       - native_contact_tol( float ) -  tolerance factor beyond the native distance for determining whether a pair of particles is 'native' (in multiples of native contact distances) 
       - opt_results ( dict ) - results of the native contact cutoff scipy.optimize.minimize optimization
       - Q_expect_results ( dict ) - results of the native contact fraction expectation calculation containing 'Q' and 'T'
       - sigmoid_param_opt ( 1D numpy array ) - optimized sigmoid parameters (x0, y0, y1, d) 
       - sigmoid_param_cov ( 2D numpy array ) - estimated covariance of sigmoid_param_opt
       - contact_type_dict ( dict ) - a dictionary of {native contact particle type pair: counts}
    """

    # Pre-load the replica trajectories into MDTraj objects, to avoid having to load them
    # at each iteration (very costly for pdb in particular)
    
    traj_dict = {}
    
    if type(traj_file_list) == list:
        n_replicas = len(traj_file_list)
    elif type(traj_file_list) == str:
        # Convert to a 1 element list if not one
        traj_file_list = traj_file_list.split()  
        n_replicas = 1
        
    for rep in range(n_replicas):
        if traj_file_list[rep][-3:] == 'dcd':
            traj_dict[rep] = md.load(traj_file_list[rep],top=md.Topology.from_openmm(cgmodel.topology))
        else:
            traj_dict[rep] = md.load(traj_file_list[rep])
    
    def minimize_sigmoid_width(x0):
        # Function to minimize:
   
        native_contact_cutoff = x0[0]
        native_contact_tol = x0[1]
        
        # Determine native contacts:
        native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
            cgmodel,
            native_structure_file,
            native_contact_cutoff*unit.angstrom,
        )
        
        if len(native_contact_list) > 0:
            # Get native contact fraction of all frames
            # To avoid loading in files each iteration, use alternate version of fraction_native_contacts code
            Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts_preloaded(
                cgmodel,
                traj_dict,
                native_contact_list,
                native_contact_distances,
                frame_begin=frame_begin,
                native_contact_tol=native_contact_tol,
                subsample=False,
            )
            
            if Q.sum() == 0:
                # There are no native contacts in the trajectory
                min_val = 1E9
                param_opt = None
                
            else:    
                try:
                    # Get expectations 
                    results = expectations_fraction_contacts(
                        Q,
                        frame_begin=frame_begin,
                        sample_spacing=frame_stride,
                        output_data=output_data,
                        num_intermediate_states=num_intermediate_states,
                    )
                    
                    param_opt, param_cov = fit_sigmoid(results["T"],results["Q"],plotfile=None)
                    
                    # This minimizes the width of the sigmoid:
                    # min_val = param_opt[3]**2
                    
                    # Maximum the difference between the max and min Q:
                    min_val = 1-abs(param_opt[2]-param_opt[1])
                    if verbose:
                        print(f'min_val: {min_val}')
                
                except:
                    # Error with computing expectation, likely due to few non-zero Q
                    min_val = 1E9
                    param_opt = None
            
        else:
            # There are no native contacts for this iteration
            min_val = 1E9
            param_opt = None
            
        if verbose:
            # Print parameters at each iteration:
            print(f"native_contact_cutoff: {native_contact_cutoff}")
            print(f"native_contact_tol: {native_contact_tol}")
            print(f"number native contacts: {len(native_contact_list)}")
            print(f"sigmoid params: {param_opt}\n")
            
        return min_val
    
    # Get bounds from equilibrium LJ distance
    particle_list = cgmodel.create_particle_list()
    sigma_bb = None
    for par in particle_list:
        if cgmodel.get_particle_type_name(par) == 'bb':
            sigma_bb = cgmodel.get_particle_sigma(par)
            break
        
    if sigma_bb is None:
        # bb is not a defined particle type
        # Use sigma of the first particle type found
        sigma_bb = cgmodel.get_particle_sigma(1)
        
    # Compute equilibrium LJ distance    
    r_eq = sigma_bb.value_in_unit(unit.angstrom)*np.power(2,(1/6))
    
    bounds = [(r_eq*0.75,r_eq*1.25),(1,2)]
    if verbose:
        print(f'Using bounds based on eq. distance for sigma = {sigma_bb}')
        print(f'{bounds}')
    
    if minimizer_options is not None:
        options_str=""
        for key,value in minimizer_options.items():
            options_str += f", {key}={value}"
        print(options_str)
        opt_results = eval(f'differential_evolution(minimize_sigmoid_width, bounds, polish=True{options_str})')
    else:
        opt_results = differential_evolution(minimize_sigmoid_width, bounds, polish=True)
    
    if opt_results['success'] == True:
        # Repeat for final plotting:
        native_contact_cutoff = opt_results.x[0] * unit.angstrom
        native_contact_tol = opt_results.x[1]
        
        # Determine native contacts:
        native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
            cgmodel,
            native_structure_file,
            native_contact_cutoff,
        )

        # Get native contact fraction of all frames
        Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts_preloaded(
            cgmodel,
            traj_dict,
            native_contact_list,
            native_contact_distances,
            frame_begin=frame_begin,
            native_contact_tol=native_contact_tol
        )

        # Get expectations 
        Q_expect_results = expectations_fraction_contacts(
            Q,
            frame_begin=frame_begin,
            sample_spacing=frame_stride,
            output_data=output_data,
            num_intermediate_states=num_intermediate_states,
        )

        sigmoid_param_opt, sigmoid_param_cov = fit_sigmoid(Q_expect_results["T"],Q_expect_results["Q"],plotfile=plotfile)

    else: 
        print('Error: native contact optimization failed')
        print(opt_results)
        native_contact_cutoff = None
        native_contact_tol = None
        Q_expect_results = None
        sigmoid_param_opt = None
        sigmoid_param_cov = None
        contact_type_dict = None
    
    return native_contact_cutoff, native_contact_tol, opt_results, Q_expect_results, sigmoid_param_opt, sigmoid_param_cov, contact_type_dict
    
    
def bootstrap_native_contacts_expectation(
    cgmodel,
    traj_file_list,
    native_contact_list,
    native_contact_distances,
    output_data='output/output.nc',
    frame_begin=0,
    sample_spacing=1,
    native_contact_tol=1.1,
    num_intermediate_states=0,
    n_trial_boot=200,
    conf_percent='sigma',
    plotfile='Q_vs_T_bootstrap.pdf',
    ):
    """
    Given a cgmodel, native contact definitions, and trajectory file list, this function calculates the
    fraction of native contacts for all specified frames, and uses a bootstrapping scheme to compute
    the uncertainties in the Q vs T folding curve. Intended to be used after the native contact tolerance
    has been optimized (either the helical or generalized versions).
    
    :param cgmodel: CGModel() class object
    :type cgmodel: class
        
    :param traj_file_list: A list of replica PDB or DCD trajectory files corresponding to the energies in the .nc file, or a single file name
    :type traj_file_list: List( str ) or str

    :param native_contact_list: A list of the nonbonded interactions whose inter-particle distances are less than the 'native_contact_cutoff_distance'.
    :type native_contact_list: List
    
    :param native_contact_distances: A numpy array of the native pairwise distances corresponding to native_contact_list
    :type native_contact_distances: Quantity

    :param frame_begin: Frame at which to start native contacts analysis (default=0)
    :type frame_begin: int
    
    :param sample_spacing: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData
    :type sample_spacing: int
    
    :param native_contact_tol: Tolerance factor beyond the native distance for determining whether a pair of particles is 'native' (in multiples of native distance)
    :type native_contact_tol: float
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list'
    :type num_intermediate_states: int
    
    :param n_trial_boot: number of trials to run for generating bootstrapping uncertainties (default=200)
    :type n_trial_boot: int
    
    :param conf_percent: Confidence level in percent for outputting uncertainties (default = 68.27 = 1 sigma)
    :type conf_percent: float
    
    :returns:
       - T_list ( List( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'C_v'
       - C_v_values ( List( float * kJ/mol/K ) ) - The heat capacity values for all (including inserted intermediates) states
       - C_v_uncertainty ( Tuple ( np.array(float) * kJ/mol/K ) ) - confidence interval for all C_v_values computed from bootstrapping
       - Tm_value ( float * unit.simtk.temperature ) - Melting point mean value computed from bootstrapping
       - Tm_uncertainty ( Tuple ( float * unit.simtk.temperature ) ) - confidence interval for melting point computed from bootstrapping
       - FWHM_value ( float * unit.simtk.temperature ) - C_v full width half maximum mean value computed from bootstrapping
       - FWHM_uncertainty ( Tuple ( float * unit.simtk.temperature ) ) - confidence interval for C_v full width half maximum computed from bootstrapping
        
    """
    
    # Pre-load the replica trajectories into MDTraj objects, to avoid having to load them
    # at each iteration (very costly for pdb in particular)
    
    traj_dict = {}
    
    if type(traj_file_list) == list:
        n_replicas = len(traj_file_list)
    elif type(traj_file_list) == str:
        # Convert to a 1 element list if not one
        traj_file_list = traj_file_list.split()  
        n_replicas = 1
        
    for rep in range(n_replicas):
        if traj_file_list[rep][-3:] == 'dcd':
            traj_dict[rep] = md.load(traj_file_list[rep],top=md.Topology.from_openmm(cgmodel.topology))
        else:
            traj_dict[rep] = md.load(traj_file_list[rep])    
   
    # Extract reduced energies and the state indices from the .nc
    reporter = MultiStateReporter(output_data, open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)
    (
        replica_energies_all,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()   
   
            
    # Get native contact fraction of all frames (bootstrapping draws uncorrelated samples from this full dataset)
    # To avoid loading in files each iteration, use alternate version of fraction_native_contacts code
    Q_all, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts_preloaded(
        cgmodel,
        traj_dict,
        native_contact_list,
        native_contact_distances,
        frame_begin=frame_begin,
        native_contact_tol=native_contact_tol,
        subsample=False,
    )
    
    # For each bootstrap trial, compute the expectation of native contacts and fit to sigmoid.
    Q_expect_boot = {}
    sigmoid_Q_max = np.zeros(n_trial_boot)
    sigmoid_Q_min = np.zeros(n_trial_boot)
    sigmoid_d = np.zeros(n_trial_boot)
    sigmoid_Tm = np.zeros(n_trial_boot)
    Q_folded = np.zeros(n_trial_boot)
    
    for i_boot in range(n_trial_boot):
        # Select production frames to analyze
        # Here we can potentially change the reference frame for each bootstrap trial.
        ref_shift = np.random.randint(sample_spacing)
        # ***We should check if these energies arrays will be the same size for
        # different reference frames
        replica_energies = replica_energies_all[:,:,(frame_begin+ref_shift)::sample_spacing]
        # ***Unlike replica energies, Q does not include the equilibration frames
        Q = Q_all[(ref_shift)::sample_spacing,:]
        
        # Get all possible sample indices
        sample_indices_all = np.arange(0,len(replica_energies[0,0,:]))
        # n_samples should match the size of the sliced replica energy dataset
        sample_indices = resample(sample_indices_all, replace=True, n_samples=len(sample_indices_all))
        
        n_state = replica_energies.shape[0]
        
        replica_energies_resample = np.zeros_like(replica_energies)
        # replica_energies is [n_states x n_states x n_frame]
        # Q is [nframes x n_states]
        Q_resample = np.zeros((len(sample_indices),n_replicas))
        
        # Select the sampled frames from array_folded_states and replica_energies:
        j = 0
        for i in sample_indices:
            replica_energies_resample[:,:,j] = replica_energies[:,:,i]
            Q_resample[j,:] = Q[i,:]
            j += 1
            
        # Run the native contacts expectation calculation:
        Q_expect_boot[i_boot] = expectations_fraction_contacts(
            Q_resample,
            frame_begin=frame_begin,
            num_intermediate_states=num_intermediate_states,
            bootstrap_energies=replica_energies_resample,
            output_data=output_data,
        )
        
        # Fit to sigmoid:
        param_opt, param_cov = fit_sigmoid(Q_expect_boot[i_boot]["T"],Q_expect_boot[i_boot]["Q"],plotfile=None)
        
        # Save the individual parameters:
        if param_opt[1] >= param_opt[2]:
            sigmoid_Q_max[i_boot] = param_opt[1]
            sigmoid_Q_min[i_boot] = param_opt[2]
        else:
            # This shouldn't occur unless d is negative
            print(f'Error with sigmoid fitting')
            sigmoid_Q_max[i_boot] = param_opt[2]
            sigmoid_Q_min[i_boot] = param_opt[1]
            
        sigmoid_d[i_boot] = param_opt[3]
        sigmoid_Tm[i_boot] = param_opt[0]
        Q_folded[i_boot] = (param_opt[1]+param_opt[2])/2
        
    # Compute uncertainty at all temps in Q_expect_boot over the n_trial_boot trials performed:
    
    # Convert dicts to array
    # Total number of temps including intermediate states:
    temp_list = Q_expect_boot[0]["T"]
    n_temps = len(temp_list)
    arr_Q_values_boot = np.zeros((n_trial_boot, n_temps))
    
    for i_boot in range(n_trial_boot):
        arr_Q_values_boot[i_boot,:] = Q_expect_boot[i_boot]["Q"]
            
    # Compute mean values:        
    Q_values = np.mean(arr_Q_values_boot,axis=0)
    sigmoid_Q_max_value = np.mean(sigmoid_Q_max)
    sigmoid_Q_min_value = np.mean(sigmoid_Q_min)
    sigmoid_d_value = np.mean(sigmoid_d)*unit.kelvin
    sigmoid_Tm_value = np.mean(sigmoid_Tm)*unit.kelvin
    Q_folded_value = np.mean(Q_folded)
    
    # Compute confidence intervals:
    if conf_percent == 'sigma':
        # Use analytical standard deviation instead of percentile method:
        
        # Q values:
        Q_std = np.std(arr_Q_values_boot,axis=0)
        Q_uncertainty = (-Q_std, Q_std)
        
        # Sigmoid Q_max:
        sigmoid_Q_max_std = np.std(sigmoid_Q_max)
        sigmoid_Q_max_uncertainty = (-sigmoid_Q_max_std, sigmoid_Q_max_std)   
        
        # Sigmoid Q_min:
        sigmoid_Q_min_std = np.std(sigmoid_Q_min)
        sigmoid_Q_min_uncertainty = (-sigmoid_Q_min_std, sigmoid_Q_min_std) 

        # Sigmoid d:
        sigmoid_d_std = np.std(sigmoid_d)
        sigmoid_d_uncertainty = (-sigmoid_d_std*unit.kelvin, sigmoid_d_std*unit.kelvin)
        
        # Sigmoid Tm:
        sigmoid_Tm_std = np.std(sigmoid_Tm)
        sigmoid_Tm_uncertainty = (-sigmoid_Tm_std*unit.kelvin, sigmoid_Tm_std*unit.kelvin)
        
        # Q_folded:
        Q_folded_std = np.std(Q_folded)
        Q_folded_uncertainty = (-Q_folded_std, Q_folded_std)
        
    else:
        # Compute specified confidence interval:
        p_lo = (100-conf_percent)/2
        p_hi = 100-p_lo
                
        # Q values:
        Q_diff = arr_Q_values_boot-np.mean(arr_Q_values_boot,axis=0)
        Q_conf_lo = np.percentile(Q_diff,p_lo,axis=0,interpolation='linear')
        Q_conf_hi = np.percentile(Q_diff,p_hi,axis=0,interpolation='linear')
      
        Q_uncertainty = (Q_conf_lo, Q_conf_hi) 
                    
        # Sigmoid Q_max:
        sigmoid_Q_max_diff = sigmoid_Q_max-np.mean(sigmoid_Q_max)
        sigmoid_Q_max_conf_lo = np.percentile(sigmoid_Q_max_diff,p_lo,interpolation='linear')
        sigmoid_Q_max_conf_hi = np.percentile(sigmoid_Q_max_diff,p_hi,interpolation='linear')
        
        sigmoid_Q_max_uncertainty = (sigmoid_Q_max_conf_lo, sigmoid_Q_max_conf_hi)
        
        # Sigmoid Q_min:
        sigmoid_Q_min_diff = sigmoid_Q_min-np.mean(sigmoid_Q_min)
        sigmoid_Q_min_conf_lo = np.percentile(sigmoid_Q_min_diff,p_lo,interpolation='linear')
        sigmoid_Q_min_conf_hi = np.percentile(sigmoid_Q_min_diff,p_hi,interpolation='linear')
        
        sigmoid_Q_min_uncertainty = (sigmoid_Q_min_conf_lo, sigmoid_Q_min_conf_hi)
        
        # Sigmoid d:
        sigmoid_d_diff = sigmoid_d-np.mean(sigmoid_d)
        sigmoid_d_conf_lo = np.percentile(sigmoid_d_diff,p_lo,interpolation='linear')
        sigmoid_d_conf_hi = np.percentile(sigmoid_d_diff,p_hi,interpolation='linear')
        
        sigmoid_d_uncertainty = (sigmoid_d_conf_lo*unit.kelvin, sigmoid_d_conf_hi*unit.kelvin)
        
        # Sigmoid Tm:
        sigmoid_Tm_diff = sigmoid_Tm-np.mean(sigmoid_Tm)
        sigmoid_Tm_conf_lo = np.percentile(sigmoid_Tm_diff,p_lo,interpolation='linear')
        sigmoid_Tm_conf_hi = np.percentile(sigmoid_Tm_diff,p_hi,interpolation='linear')
        
        sigmoid_Tm_uncertainty = (sigmoid_Tm_conf_lo*unit.kelvin, sigmoid_Tm_conf_hi*unit.kelvin)
        
        # Q_folded:
        Q_folded_diff = Q_folded-np.mean(Q_folded)
        Q_folded_conf_lo = np.percentile(Q_folded_diff,p_lo,interpolation='linear')
        Q_folded_conf_hi = np.percentile(Q_folded_diff,p_hi,interpolation='linear')
        
        Q_folded_uncertainty = (Q_folded_conf_lo, Q_folded_conf_hi*unit.kelvin)      
    
    # Compile sigmoid results into dict:
    sigmoid_results_boot = {}
    
    sigmoid_results_boot['sigmoid_Q_max_value'] = sigmoid_Q_max_value
    sigmoid_results_boot['sigmoid_Q_max_uncertainty'] = sigmoid_Q_max_uncertainty
    
    sigmoid_results_boot['sigmoid_Q_min_value'] = sigmoid_Q_min_value
    sigmoid_results_boot['sigmoid_Q_min_uncertainty'] = sigmoid_Q_min_uncertainty
    
    sigmoid_results_boot['sigmoid_d_value'] = sigmoid_d_value
    sigmoid_results_boot['sigmoid_d_uncertainty'] = sigmoid_d_uncertainty
    
    sigmoid_results_boot['sigmoid_Tm_value'] = sigmoid_Tm_value
    sigmoid_results_boot['sigmoid_Tm_uncertainty'] = sigmoid_Tm_uncertainty
    
    sigmoid_results_boot['Q_folded_value'] = Q_folded_value
    sigmoid_results_boot['Q_folded_uncertainty'] = Q_folded_uncertainty
    
    # Plot Q vs T results with uncertainty and mean sigmoid parameters
    if conf_percent=='sigma':
        plot_native_contact_fraction(
            temp_list, Q_values, Q_std, plotfile=plotfile, sigmoid_dict=sigmoid_results_boot
            )
    # TODO: implement unequal upper and lower error plotting
    
    return Q_values, Q_uncertainty, sigmoid_results_boot
    
    
def optimize_Q_tol_helix(
    cgmodel, native_structure_file, traj_file_list, output_data="output/output.nc",
    num_intermediate_states=0, frame_begin=0, frame_stride=1, backbone_type_name='bb',
    plotfile='native_contacts_helix_opt.pdf', verbose=False, brute_step=0.1*unit.angstrom):
    """
    Given a coarse grained model and a native structure as input, determine which helical backbone
    sequences are native contacts, and the optimal distance tolerance for scanning the
    trajectory for native contacts. Tolerance is determined by brute force scan.

    :param cgmodel: CGModel() class object
    :type cgmodel: class
    
    :param native_structure_file: Path to file ('pdb' or 'dcd') containing particle positions for the native structure.
    :type native_structure_file: str 
    
    :param traj_file_list: A list of replica PDB or DCD trajectory files corresponding to the energies in the .nc file, or a single file name
    :type traj_file_list: List( str ) or str
    
    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data, default = ("output/output.nc")                                                                                                  
    :type output_data: str

    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list'
    :type num_intermediate_states: int 
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int

    :param frame_stride: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData
    :type frame_stride: int 
    
    :param backbone_type_name: type name in cgmodel which corresponds to the particles forming the helical backbone.
    :type backbone_type_name: str
    
    :param brute_step: step size in distance units for brute force tolerance optimization (final optimization searches between intervals)
    :type brute_step: Quantity ( float )

    :returns:
       - opt_seq_spacing ( int ) - the (i) to (i+n) number n defining contacting backbone beads
       - native_contact_tol( float ) -  tolerance factor beyond the native distance for determining whether a pair of particles is 'native' (in multiples of native contact distances) 
       - opt_results ( dict ) - results of the native contact tolerance scipy.optimize.minimize optimization
       - Q_expect_results ( dict ) - results of the native contact fraction expectation calculation containing 'Q' and 'T'
       - sigmoid_param_opt ( 1D numpy array ) - optimized sigmoid parameters (x0, y0, y1, d) 
       - sigmoid_param_cov ( 2D numpy array ) - estimated covariance of sigmoid_param_opt
    """

    # Pre-load the replica trajectories into MDTraj objects, to avoid having to load them
    # at each iteration (very costly for pdb in particular)
    
    traj_dict = {}
    
    if type(traj_file_list) == list:
        n_replicas = len(traj_file_list)
    elif type(traj_file_list) == str:
        # Convert to a 1 element list if not one
        traj_file_list = traj_file_list.split()  
        n_replicas = 1
        
    for rep in range(n_replicas):
        if traj_file_list[rep][-3:] == 'dcd':
            traj_dict[rep] = md.load(traj_file_list[rep],top=md.Topology.from_openmm(cgmodel.topology))
        else:
            traj_dict[rep] = md.load(traj_file_list[rep])
            
    # Determine native contacts:
    native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
        cgmodel,
        native_structure_file,
        backbone_type_name=backbone_type_name,
        )
    
    def minimize_sigmoid_width_1d(x0):
        # Function to minimize:
        native_contact_tol = x0
        
        if len(native_contact_list) > 0:
            # Get native contact fraction of all frames
            # To avoid loading in files each iteration, use alternate version of fraction_native_contacts code
            Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts_preloaded(
                cgmodel,
                traj_dict,
                native_contact_list,
                native_contact_distances,
                frame_begin=frame_begin,
                native_contact_tol=native_contact_tol,
                subsample=False,
            )
            
            if Q.sum() == 0:
                # There are no native contacts in the trajectory
                min_val = 1E9
                param_opt = None
                
            else:    
                try:
                    # Get expectations 
                    results = expectations_fraction_contacts(
                        Q,
                        frame_begin=frame_begin,
                        sample_spacing=frame_stride,
                        output_data=output_data,
                        num_intermediate_states=num_intermediate_states,
                    )
                    
                    param_opt, param_cov = fit_sigmoid(results["T"],results["Q"],plotfile=None)
                    
                    # This minimizes the width of the sigmoid:
                    # min_val = param_opt[3]**2
                    
                    # Maximum the difference between the max and min Q:
                    min_val = 1-abs(param_opt[2]-param_opt[1])
                    if verbose:
                        print(f'min_val: {min_val}')
                
                except:
                    # Error with computing expectation, likely due to few non-zero Q
                    min_val = 1E9
                    param_opt = None
            
        else:
            # There are no native contacts for this iteration
            min_val = 1E9
            param_opt = None
            
        if verbose:
            # Print parameters at each iteration:
            print(f"native_contact_tol: {native_contact_tol}")
            print(f"number native contacts: {len(native_contact_list)}")
            print(f"sigmoid params: {param_opt}\n")
            
        return min_val
        
    bounds = (1,2)
    if verbose:
        print(f'Using the following native_contact_tol bounds:')
    brute_range = [slice(bounds[0],bounds[1],brute_step.value_in_unit(unit.angstrom))]

    opt_results = brute(minimize_sigmoid_width_1d, brute_range)
    
    # Repeat for final plotting:
    native_contact_tol = opt_results[0]
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
        cgmodel,
        native_structure_file,
        backbone_type_name=backbone_type_name,
        )

    # Get native contact fraction of all frames
    Q, Q_avg, Q_stderr, decorrelation_time = fraction_native_contacts_preloaded(
        cgmodel,
        traj_dict,
        native_contact_list,
        native_contact_distances,
        frame_begin=frame_begin,
        native_contact_tol=native_contact_tol
    )

    # Get expectations 
    Q_expect_results = expectations_fraction_contacts(
        Q,
        frame_begin=frame_begin,
        sample_spacing=frame_stride,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
    )

    sigmoid_param_opt, sigmoid_param_cov = fit_sigmoid(Q_expect_results["T"],Q_expect_results["Q"],plotfile=plotfile)
    
    return opt_seq_spacing, native_contact_tol, opt_results, Q_expect_results, sigmoid_param_opt, sigmoid_param_cov 

    
def plot_native_contact_fraction(temperature_list, Q, Q_uncertainty, plotfile="Q_vs_T.pdf", sigmoid_dict=None):
    """
    Given a list of temperatures and corresponding native contact fractions, plot Q vs T.
    If a sigmoid dict from bootstrapping is given, also plot the sigmoid curve.
    Note that this sigmoid curve is generated by using the mean values of the 4 hyperbolic fitting parameters
    taken over all bootstrap trials, not a direct fit to the Q vs T data. 

    :param temperature_list: List of temperatures that will be used to define different replicas (thermodynamics states), default = None
    :type temperature_list: List( `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_ * number_replicas )

    :param Q: native contact fraction for a given temperature
    :type Q: np.array(float * len(temperature_list))
    
    :param Q_uncertainty: uncertainty associated with Q
    :type Q_uncertainty: np.array(float * len(temperature_list))
    
    :param plotfile: Path to output file for plotting results (default='Q_vs_T.pdf')
    :type plotfile: str
    
    :param sigmoid_dict: dictionary containing sigmoid parameter mean values and uncertainties (default=None)
    :type sigmoid_dict: dict
    
    """
    temperature_array = np.zeros((len(temperature_list)))
    for i in range(len(temperature_list)):
        temperature_array[i] = temperature_list[i].value_in_unit(unit.kelvin)
    
    if sigmoid_dict is not None:
        # Also plot sigmoid curve
        def tanh_switch(x,x0,y0,y1,d):
            return (y0+y1)/2-((y0-y1)/2)*np.tanh(np.radians(x-x0)/d)
        
        xsig = np.linspace(temperature_array[0],temperature_array[-1],1000)
        ysig = tanh_switch(
            xsig,
            sigmoid_dict['sigmoid_Tm_value'].value_in_unit(unit.kelvin),
            sigmoid_dict['sigmoid_Q_max_value'],
            sigmoid_dict['sigmoid_Q_min_value'],
            sigmoid_dict['sigmoid_d_value'].value_in_unit(unit.kelvin),
            )
        
        
        line1 = plt.errorbar(
            temperature_array,
            Q,
            yerr=Q_uncertainty,
            linewidth=0.5,
            markersize=4,
            fmt='ob',
            fillstyle='none',
            capsize=4,
            label='bootstrap mean',
        )
        
        line2 = plt.plot(
            xsig, ysig,'k-',
            label='bootstrap hyperbolic fit',
        )
        
        line3 = plt.errorbar(
            sigmoid_dict['sigmoid_Tm_value'].value_in_unit(unit.kelvin),
            sigmoid_dict['Q_folded_value'],
            xerr=sigmoid_dict['sigmoid_Tm_uncertainty'][1].value_in_unit(unit.kelvin),
            yerr=sigmoid_dict['Q_folded_uncertainty'][1],
            linewidth=0.5,
            markersize=4,
            fmt='D-r',
            fillstyle='none',
            capsize=4,
            label='melting point'
        )
        
        xlim = plt.xlim()
        ylim = plt.ylim()        
        
        # TODO: update to use the asymmetric uncertainties here for confidence intervals
        # We can add the hyperbolic fits with parameters for the upper and lower confidence bounds
        plt.text(
            (xlim[0]+0.90*(xlim[1]-xlim[0])),
            (ylim[0]+0.50*(ylim[1]-ylim[0])),
            f"T_m = {sigmoid_dict['sigmoid_Tm_value'].value_in_unit(unit.kelvin):.2f} \u00B1 {sigmoid_dict['sigmoid_Tm_uncertainty'][1].value_in_unit(unit.kelvin):.2f}    \n"\
            f"Q_m = {sigmoid_dict['Q_folded_value']:.4f} \u00B1 {sigmoid_dict['Q_folded_uncertainty'][1]:.4f}\n"\
            f"d = {sigmoid_dict['sigmoid_d_value'].value_in_unit(unit.kelvin):.4f} \u00B1 {sigmoid_dict['sigmoid_d_uncertainty'][1].value_in_unit(unit.kelvin):.4f}\n"\
            f"Qmax = {sigmoid_dict['sigmoid_Q_max_value']:.4f} \u00B1 {sigmoid_dict['sigmoid_Q_max_uncertainty'][1]:.4f}\n"\
            f"Qmin = {sigmoid_dict['sigmoid_Q_min_value']:.4f} \u00B1 {sigmoid_dict['sigmoid_Q_min_uncertainty'][1]:.4f}",
            {'fontsize': 10},
            horizontalalignment='right',
            )
        
        plt.legend()

    else:
        plt.errorbar(
            temperature_array,
            Q,
            Q_uncertainty,
            linewidth=0.5,
            markersize=4,
            fmt='o-',
            fillstyle='none',
            capsize=4,
        )

    # Fix y limits:
    plt.xlim((temperature_array[0],temperature_array[-1]))
    plt.ylim((0,1))
    
    plt.xlabel("T (K)")
    plt.ylabel("Native contact fraction")
    plt.savefig(plotfile)
    plt.close()
    
    
def plot_native_contact_timeseries(
    Q,
    time_interval=1.0*unit.picosecond,
    frame_begin=0,
    plot_per_page=3,
    plotfile="Q_vs_time.pdf",
    figure_title=None,
):
    """
    Given average native contact fractions timeseries for each replica or state, plot Q vs time.

    :param Q: native contact fraction for a given temperature
    :type Q: np.array(float * nframes x len(temperature_list))
    
    :param time_interval: interval between energy exchanges.
    :type time_interval: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int
    
    :param plot_per_page: number of subplots per pdf page (default=3)
    :type plot_per_page: int

    :param plotfile: Path to output file for plotting results (default='Q_vs_time.pdf')
    :type plotfile: str
    
    :param figure_title: title of overall plot
    :type figure_title: str
    
    """
        
    time_shift=frame_begin*time_interval    
        
    simulation_times = np.array(
        [
            step * time_interval.value_in_unit(unit.picosecond)
            for step in range(len(Q[:,0]))
        ]
    )
    
    simulation_times += time_shift.value_in_unit(unit.picosecond)
    
    # Determine number of data series:
    nseries = len(Q[0,:])
    nrow = plot_per_page
    
    # Number of pdf pages
    npage = int(np.ceil(nseries/nrow))
    
    xlabel="Simulation time (ps)"
    ylabel="Q"
    
    # To improve pdf render speed, sparsify data to display less than 2000 data points
    n_xdata = len(simulation_times)
    
    if n_xdata <= 1000:
        plot_stride = 1
    else:
        plot_stride = int(np.floor(n_xdata/1000))    
    
    with PdfPages(plotfile) as pdf:
        plotted_per_page=0
        page_num=1
        figure = plt.figure(figsize=(8.5,11))
        for i in range(nseries):
            plotted_per_page += 1
            
            plt.subplot(nrow,1,plotted_per_page)
            plt.plot(
                simulation_times[::plot_stride],
                Q[::plot_stride,i],
                '-',
                linewidth=0.5,
                markersize=4,
            )
            
            
            plt.ylabel(ylabel)
                    
            plt.title(f"replica {i+1}",fontweight='bold')
            
            if (plotted_per_page >= nrow) or ((i+1)==nseries):
                # Save and close previous page
                
                # Use xlabels for bottom row only:
                plt.xlabel(xlabel)
                
                # Adjust subplot spacing
                plt.subplots_adjust(hspace=0.3)

                if figure_title != None:
                    plt.suptitle(f"{figure_title} ({page_num})",fontweight='bold')
            
                pdf.savefig()
                plt.close()
                plotted_per_page = 0
                page_num += 1
                if (i+1)!= nseries:
                    figure = plt.figure(figsize=(8.5,11))
        
   