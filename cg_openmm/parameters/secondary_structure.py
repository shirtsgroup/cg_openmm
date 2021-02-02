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
from scipy.optimize import minimize, Bounds, brute, dual_annealing, differential_evolution
from cg_openmm.utilities.util import fit_sigmoid  

kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant

def get_native_contacts(cgmodel, native_structure_file, native_contact_distance_cutoff):
    """
    Given a coarse grained model, positions for that model, and positions for the native structure, this function calculates the fraction of native contacts for the model.

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
    # if native_structure_file[-3:] == 'dcd':
    native_traj = md.load(native_structure_file,top=md.Topology.from_openmm(cgmodel.topology))
    # ***Note: The clustering dcds are written with unit nanometers,
    # but this may not always be the case.
    native_structure = native_traj[0].xyz[0]*unit.nanometer
    # else:
        # native_structure = PDBFile(native_structure_file).getPositions()
        
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


def expectations_fraction_contacts(fraction_native_contacts, frame_begin=0, sample_spacing=1, output_data="output/output.nc", num_intermediate_states=0):
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
    
    """

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
    native_contact_cutoff_ratio=1.00,
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
    
    :param native_contact_cutoff_ratio: The distance below which two nonbonded, interacting particles in a non-native pose are assigned as a "native contact", as a ratio of the distance for that contact in the native structure, default=1.00
    :type native_contact_cutoff_ratio: float
    
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
        native_contact_matrix = (traj_distances<(native_contact_cutoff_ratio*native_contact_distances.value_in_unit(nc_unit)))

        number_native_interactions=np.sum(native_contact_matrix,axis=1)

        Q[:,rep] = number_native_interactions/len(native_contact_distances)
        Q_avg[rep] = np.mean(Q[:,rep])
        # Compute standard error:
        Q_stderr[rep] = np.std(Q[:,rep])/np.sqrt(len(Q[:,rep]))
    
    # Determine the decorrelation time of native contact fraction timeseries data:
    # Note: if these are replica trajectories, we will get a folding rate
    # If these are state trajectories, we will get decorrelation time for constant state
    
    #***Update this to use the same heuristic as the replica exchange energy decorrelation.
    
    if subsample:
        max_sample_spacing = 1
        subsample_indices = {}
        for rep in range(n_replicas):
            subsample_indices[rep] = timeseries.subsampleCorrelatedData(
                Q[:,rep],
                conservative=True,
            )
            if (subsample_indices[rep][1]-subsample_indices[rep][0]) > max_sample_spacing:
                max_sample_spacing = (subsample_indices[rep][1]-subsample_indices[rep][0])
        
        decorrelation_spacing = max_sample_spacing
    else:
        decorrelation_spacing = None
        
    return Q, Q_avg, Q_stderr, decorrelation_spacing
    
    
def fraction_native_contacts_preloaded(
    cgmodel,
    traj_dict,
    native_contact_list,
    native_contact_distances,
    frame_begin=0,
    native_contact_cutoff_ratio=1.00,
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
    
    :param native_contact_cutoff_ratio: The distance below which two nonbonded, interacting particles in a non-native pose are assigned as a "native contact", as a ratio of the distance for that contact in the native structure, default=1.00
    :type native_contact_cutoff_ratio: float
    
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
            nframes = traj_dict[rep].n_frames
            Q = np.zeros((nframes,n_replicas))
        
        traj_distances = md.compute_distances(
            traj_dict[rep][frame_begin:],native_contact_list,periodic=False,opt=True)
            
        if rep == 0:
            nframes = traj_dict[rep][frame_begin:].n_frames
            Q = np.zeros((nframes,n_replicas))
            
        # This produces a [nframe x len(native_contacts)] array
  
        # Compute Boolean matrix for whether or not a distance is native
        native_contact_matrix = (traj_distances<(native_contact_cutoff_ratio*native_contact_distances.value_in_unit(nc_unit)))

        number_native_interactions=np.sum(native_contact_matrix,axis=1)

        Q[:,rep] = number_native_interactions/len(native_contact_distances)
        Q_avg[rep] = np.mean(Q[:,rep])
        # Compute standard error:
        Q_stderr[rep] = np.std(Q[:,rep])/np.sqrt(len(Q[:,rep]))
    
    # Determine the decorrelation time of native contact fraction timeseries data:
    # Note: if these are replica trajectories, we will get a folding rate
    # If these are state trajectories, we will get decorrelation time for constant state
    
    #***Update this to use the same heuristic as the replica exchange energy decorrelation.
    
    if subsample:
        max_sample_spacing = 1
        subsample_indices = {}
        for rep in range(n_replicas):
            subsample_indices[rep] = timeseries.subsampleCorrelatedData(
                Q[:,rep],
                conservative=True,
            )
            if (subsample_indices[rep][1]-subsample_indices[rep][0]) > max_sample_spacing:
                max_sample_spacing = (subsample_indices[rep][1]-subsample_indices[rep][0])
        
        decorrelation_spacing = max_sample_spacing
    else:
        decorrelation_spacing = None
        
    return Q, Q_avg, Q_stderr, decorrelation_spacing
    
    
    
def optimize_Q_cut(
    cgmodel, native_structure_file, traj_file_list, output_data="output/output.nc",
    num_intermediate_states=0, frame_begin=0, frame_stride=1, opt_method='TNC',
    plotfile='native_contacts_opt.pdf', verbose=False):
    """
    Given a coarse grained model and a native structure as input

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
    
    :param opt_method: scipy.optimize.minimize method (default='Nelder-Mead')
    :type opt_method: str

    :returns:
       - native_contact_cutoff ( `Quantity() <https://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - The ideal distance below which two nonbonded, interacting particles should be defined as a "native contact"
       - native_contact_cutoff_ratio ( float ) - cutoff for native contacts when scanning trajectory, in multiples of native_contact_cutoff
       - opt_results ( dict ) - results of the native contact cutoff scipy.optimize.minimize optimization
       - Q_expect_results ( dict ) - results of the native contact fraction expectation calculation containing 'Q' and 'T'
       - sigmoid_param_opt ( 1D numpy array ) - optimized sigmoid parameters (x0, y0, y1, d) 
       - sigmoid_param_cov ( 2D numpy array ) - estimated covariance of sigmoid_param_opt
       - contact_type_dict ( dict ) - a dictionary of {native contact particle type pair: counts}
    """

    # Initial guess for native_contact_cutoff (unit.angstrom), native_contact_cutoff_ratio (unitless):
    # TODO: estimate this from the cgmodel rather than hard coding
    
    x0 = [5.0, 1.0]
    
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
        native_contact_cutoff_ratio = x0[1]
        
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
                native_contact_cutoff_ratio=native_contact_cutoff_ratio,
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
                    # min_val = param_opt[3]**2
                    
                    # Or, if we want to maximum the difference between the max and min Q:
                    min_val = 1-abs(param_opt[2]-param_opt[1])
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
            print(f"native_contact_cutoff_ratio: {native_contact_cutoff_ratio}")
            print(f"sigmoid params: {param_opt}\n")
        
        return min_val
        
    # The native_contact_cutoff_ratio should not be less than 1.
    #bounds = Bounds(np.array([0.5,1]),np.array([10,2]))
    bounds = [(0.5,10),(1,2)]
     
    # ***Note: Default tolerance is 1E-6. We should allow this to be specified in the future.    
    #opt_results = minimize(minimize_sigmoid_width, x0, method=opt_method,
    #   bounds=bounds)
    
    opt_results = brute(minimize_sigmoid_width,(slice(0.5,7,0.5),slice(1,1.5,0.05)))
    
    #if opt_results['success'] == True:
        # Repeat for final plotting:
        #native_contact_cutoff = opt_results.x[0] * unit.angstrom
        #native_contact_cutoff_ratio = opt_results.x[1]
        
    native_contact_cutoff = opt_results[0] * unit.angstrom
    native_contact_cutoff_ratio = opt_results[1]
    
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
        native_contact_cutoff_ratio=native_contact_cutoff_ratio
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

    # else: 
        # print('Optimization failed using a starting guess of {x0}')
        # native_contact_cutoff = None
        # native_contact_cutoff_ratio = None
        # Q_expect_results = None
        # sigmoid_param_opt = None
        # sigmoid_param_cov = None
        # contact_type_dict = None
    
    return native_contact_cutoff, native_contact_cutoff_ratio, opt_results, Q_expect_results, sigmoid_param_opt, sigmoid_param_cov, contact_type_dict
    

def plot_native_contact_fraction(temperature_list, Q, Q_uncertainty,plotfile="Q_vs_T.pdf"):
    """
    Given a list of temperatures and corresponding native contact fractions, plot Q vs T.

    :param temperature_list: List of temperatures that will be used to define different replicas (thermodynamics states), default = None
    :type temperature_list: List( `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_ * number_replicas )

    :param Q: native contact fraction for a given temperature
    :type Q: np.array(float * len(temperature_list))
    
    :param Q_uncertainty: uncertainty associated with Q
    :type Q_uncertainty: np.array(float * len(temperature_list))
    
    """
    temperature_array = np.zeros((len(temperature_list)))
    for i in range(len(temperature_list)):
        temperature_array[i] = temperature_list[i].value_in_unit(unit.kelvin)
    
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

    :param plotfile: The pathname of the output file for plotting results, default = "replica_exchange_state_transitions.png"
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
        
   