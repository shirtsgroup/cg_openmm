import os

import matplotlib.pyplot as plt
import numpy as np
import pymbar
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from cg_openmm.utilities.random_builder import *
from openmm import unit
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeAnalyzer
from scipy import interpolate
from sklearn.utils import resample

kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant
kB = kB.in_units_of(unit.kilojoule/unit.kelvin/unit.mole)


def classify_Q_states(Q, Q_folded):
    """
    This function determines the conformational state of each element of the native contacts array, given a threshold
    for being 'folded/unfolded'. In the outputted array, 1 is 'folded' and 0 is 'unfolded'. If Q_folded is a list,
    of multiple cutoffs, additional classification is performed.
    
    :param Q: native contact fraction array of size [n_frames x n_states]
    :type Q: 2D numpy array ( float )
    
    :param Q_folded: threshold for a native contact fraction corresponding to a folded state (Q[i,j] is folded if Q[i,j] >= Q_folded)
    :type Q_folded: float
    
    :returns:
      - array_folded_states ( 2D numpy array ( float ) ) - conformational state matrix of shape [n_frames x n_replicas]
    """
    
    if type(Q_folded) == list:
        # Multiple states specified
        # Sort to ascending order:
        Q_folded = sorted(Q_folded)
        
        array_folded_states = np.zeros_like(Q)
        
        for q in Q_folded:
            array_folded_states += np.multiply((Q>=q),1)
            # 0 is unfolded, 1 is folded state with smallest Q cutoff, 2 is folded state with
            # next largest Q cutoff, and so on.
            
    else:
        # Assume a binary folded/unfolded system
        array_folded_states = np.multiply((Q>=Q_folded),1)
    
    return array_folded_states


def expectations_free_energy(Q, Q_folded, temperature_list, frame_begin=0, sample_spacing=1, output_data="output/output.nc",
    bootstrap_energies=None, num_intermediate_states=0, array_folded_states=None):
    """
    This function calculates the free energy difference (with uncertainty) between all conformational states as a function of temperature.

    :param Q: native contact fraction array of size [n_frames x n_states]
    :type Q: 2D numpy array ( float )
    
    :param Q_folded: threshold for a native contact fraction corresponding to a folded state (Q[i,j] is folded if Q[i,j] >= Q_folded)
    :type Q_folded: float or list ( float )

    :param temperature_list: List of temperatures for the simulation data (necessary because bootstrap version doesn't read in the file)
    :type temperature_list: List( float * simtk.unit.temperature )
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int    
    
    :param sample_spacing: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData
    :type sample_spacing: int     
    
    :param output_data: Path to the simulation .nc file.
    :type output_data: str
    
    :param num_intermediate_states: Number of unsampled thermodynamic states between sampled states to include in the calculation
    :type num_intermediate_states: int
    
    :param bootstrap_energies: a custom replica_energies array to be used for bootstrapping calculations. Used instead of the energies in the .nc file.
    :type bootstrap_energies: 2d numpy array (float)
    
    :param array_folded_states: a precomputed array classifying the different conformational states
    :type array_folded_states: 2d numpy array (int)
    
    :returns:
      - full_T_list - A 1D numpy array listing of all temperatures, including sampled and intermediate unsampled
      - deltaF_values - A dictionary of the form {"statei_statej": 1D numpy array}, containing free energy change for each T in
                        full_T_list, for each conformational state transition.
      - deltaF uncertainty - A dictionary containing 1D numpy arrays of uncertainties corresponding to deltaF_values
    """
    
    if bootstrap_energies is not None:
        # Use a subsampled replica_energy matrix instead of reading from file
        replica_energies = bootstrap_energies
        
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
        
        # Close the data file:
        reporter.close()   
        
        # Select production frames to analyze
        replica_energies = replica_energies_all[:,:,frame_begin::sample_spacing]
        
        # Check the size of the Q array:
        if np.shape(replica_energies)[2] != np.shape(Q)[0]:
            # Mismatch in number of frames.
            if np.shape(replica_energies_all[:,:,frame_begin::sample_spacing])[2] == np.shape(Q[::sample_spacing,:])[0]:
                # Correct starting frame, need to apply sampling stride:
                Q = Q[::sample_spacing,:]
            elif np.shape(replica_energies_all)[2] == np.shape(Q)[0]:
                # This is the full Q, slice production frames:
                Q = Q[production_start::sample_spacing,:]
            else:
                print(f'Error: Q array of shape {Q.shape} incompatible with energies array of shape {replica_energies.shape}')
                exit()    
            
    # Classify Q into folded/unfolded states
    if array_folded_states is None:
        array_folded_states = classify_Q_states(Q,Q_folded)
    else:
        # Use a precomputed array_folded_states instead of standard classification scheme.
        # Q and Q_folded inputs are ignored.
        #***The array_folded_states should be sliced with sample_spacing if not a bootstrap calc:
        pass
    
    # Number of configurational states:
    n_conf_states = len(np.unique(array_folded_states))
        
    # convert the energies from replica/evaluated state/sample form to evaluated state/sample form
    replica_energies = pymbar.utils.kln_to_kn(replica_energies)  
    n_samples = len(replica_energies[0,:])
        
    # Reshape array_folded_states to row vector for pymbar
    # We need to order the data by replica, rather than by frame
    array_folded_states = np.reshape(array_folded_states,(np.size(array_folded_states)),order='F')
        
    # determine the numerical values of beta at each state in units consisten with the temperature
    Tunit = temperature_list[0].unit
    temps = np.array([temp.value_in_unit(Tunit)  for temp in temperature_list])  # should this just be array to begin with
    beta_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * temps)

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

    # Calculate N expectations that a structure is in configurational state n
    # We need the probabilities of being in each state - first construct vectors of 0
    # (not in current state) and 1 (in current state)

    bool_i = np.zeros((n_conf_states,array_folded_states.shape[0]))

    for i in range(n_conf_states):
        i_vector = np.full_like(array_folded_states,i)
        # Convert True/False to integer 1/0 for each energy data point:
        bool_i[i] = np.multiply((i_vector==array_folded_states),1)

    # Calculate the expectation of F at each unsampled states

    # Loop over each thermodynamic state:
    results = {}
    for i in range(len(full_T_list)):
        U_n = unsampled_state_energies[i,:]

        # compute expectations of being in conformational state n
        # Store results in a dictionary
        results[str(i)] = mbarT.computeMultipleExpectations(
            bool_i,U_n,compute_covariance=True)

    deltaF_values = {}
    deltaF_uncertainty = {}
    n_trans = 0 # store the number of unique transitions
    F_unit = (-kB*full_T_list[0]*Tunit).unit # units of free energy

    # Initialize the results dictionaries
    for s1 in range(n_conf_states):
        for s2 in range(s1+1,n_conf_states):
            n_trans += 1
            deltaF_values[f"state{s1}_state{s2}"] = np.zeros(len(full_T_list))
            deltaF_uncertainty[f"state{s1}_state{s2}"] = np.zeros(len(full_T_list))
                
    # Compute free energies from probability ratios:
    for i in range(len(full_T_list)):
        for s1 in range(n_conf_states):
            for s2 in range(s1+1,n_conf_states):
                # Free energy change for s2 --> s1 at temp i
                deltaF_values[f"state{s1}_state{s2}"][i] = (
                    -kB*full_T_list[i]*Tunit*(
                    np.log(results[str(i)][0][s1])-
                    np.log(results[str(i)][0][s2]))).value_in_unit(F_unit)
                    
                # Get covariance matrix:
                theta_i = results[str(i)][2]
                deltaF_uncertainty[f"state{s1}_state{s2}"][i] = (
                    kB*full_T_list[i]*unit.kelvin*np.sqrt(
                    theta_i[s1,s1] + theta_i[s2,s2] - (theta_i[s2,s1]+theta_i[s1,s2]))).value_in_unit(F_unit)

    # Add the units back on:
    for s1 in range(n_conf_states):
        for s2 in range(s1+1,n_conf_states):    
            deltaF_values[f"state{s1}_state{s2}"] *= F_unit
            deltaF_uncertainty[f"state{s1}_state{s2}"] *= F_unit
    full_T_list *= Tunit
                    
    return full_T_list, deltaF_values, deltaF_uncertainty
    

def bootstrap_free_energy_folding(Q, Q_folded, array_folded_states=None, output_data="output/output.nc", frame_begin=0,
    sample_spacing=1, n_trial_boot=200, num_intermediate_states=0, conf_percent='sigma', plotfile_dir="output"):
    """
    Function for computing uncertainty of free energy, entropy, and enthalpy using bootstrapping with varying starting frames.

    :param Q: native contact fraction array of size [n_frames x n_states] (with equilibration region already trimmed)
    :type Q: 2D numpy array ( float )
    
    :param Q_folded: threshold for a native contact fraction corresponding to a folded state (Q[i,j] is folded if Q[i,j] >= Q_folded)
    :type Q_folded: float or list ( float )
    
    :param array_folded_states: a precomputed array classifying the different conformational states
    :type array_folded_states: 2d numpy array (int)
    
    :param output_data: Path to the simulation .nc file.
    :type output_data: str    
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int    
    
    :param sample_spacing: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData
    :type sample_spacing: int     
    
    :param n_trial_boot: number of trials to run for generating bootstrapping uncertainties (default=200)
    :type n_trial_boot: int
    
    :param num_intermediate_states: Number of unsampled thermodynamic states between sampled states to include in the calculation
    :type num_intermediate_states: int
    
    :returns:
      - full_T_list - A 1D numpy array listing of all temperatures, including sampled and intermediate unsampled
      - deltaF_values - A dictionary of the form {"statei_statej": 1D numpy array}, containing free energy change for each T in
                        full_T_list, for each conformational state transition.
      - deltaF uncertainty - A dictionary containing tuple of 1D numpy arrays of lower/upper of uncertainties corresponding to deltaF_values  
      - deltaS_values - A dictionary of the form {"statei_statej": 1D numpy array}, containing entropy change for each T in
                        full_T_list, for each conformational state transition.
      - deltaS uncertainty - A dictionary containing tuple of 1D numpy arrays of lower/upper uncertainties corresponding to deltaS_values 
      - deltaU_values - A dictionary of the form {"statei_statej": 1D numpy array}, containing enthalpy change for each T in
                        full_T_list, for each conformational state transition.
      - deltaU uncertainty - A dictionary containing tuple of 1D numpy arrays of lower/upper of uncertainties corresponding to deltaU_values
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
    
    # Get temperature_list from .nc file:
    states = reporter.read_thermodynamic_states()[0]
    
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)    
    
    # Close the data file:
    reporter.close()      
    
    # Select production frames to analyze
    replica_energies_prod = replica_energies_all[:,:,frame_begin::]
    
    # For shifting reference frame bootstrap, we need the entire Q and energy arrays starting from frame_start

    if array_folded_states is None:
        if np.shape(replica_energies_prod)[2] != np.shape(Q)[0]:
            print(f'Error: Q array of shape {Q.shape} incompatible with energies array of shape {replica_energies_prod.shape}')
            exit()
    else:
        if np.shape(replica_energies_prod)[2] != np.shape(array_folded_states)[0]:
            print(f'Error: Q array of shape {Q.shape} incompatible with energies array of shape {replica_energies_prod.shape}')
            exit()
        
    if array_folded_states is None:
        # Use the raw contact fractions as input to the free energy function
        Q_all = Q
    else:
        # Use the precomputed conformational state array as input to the free energy function
        array_folded_states_all = array_folded_states

    # Overall results:
    deltaF_values = {}
    deltaF_uncertainty = {}
    deltaS_values = {}
    deltaS_uncertainty = {}
    deltaU_values = {}
    deltaU_uncertainty = {}
    
    # Uncertainty for each sampling trial:
    deltaF_values_boot = {}
    deltaF_uncertainty_boot = {}
    
    deltaS_values_boot = {}
    deltaS_uncertainty_boot = {}
    
    deltaU_values_boot = {}
    deltaU_uncertainty_boot = {}
    
    # Get units:
    F_unit = (kB*unit.kelvin).unit # units of free energy
    T_unit = temperature_list[0].unit
    S_unit = F_unit/T_unit
    U_unit = F_unit
    
    for i_boot in range(n_trial_boot):
        # Here we can potentially change the reference frame for each bootstrap trial.
        # This requires the array slicing to be done here, not above.
        ref_shift = np.random.randint(sample_spacing)
        
        # Replica energies and Q already have equilibration period removed:
        replica_energies = replica_energies_prod[:,:,ref_shift::sample_spacing]
        
        if array_folded_states is None:
            Q = Q_all[ref_shift::sample_spacing,:]
            n_states = len(Q[0,:])
        else:
            array_folded_states = array_folded_states_all[ref_shift::sample_spacing,:]
            n_states = len(array_folded_states[0,:])
        
        # Get all possible sample indices
        sample_indices_all = np.arange(0,len(replica_energies[0,0,:]))
        # n_samples should match the size of the sliced replica energy dataset
        sample_indices = resample(sample_indices_all, replace=True, n_samples=len(sample_indices_all))
        
        replica_energies_resample = np.zeros_like(replica_energies)
        # replica_energies is [n_states x n_states x n_frame]
        # Q is [nframes x n_states]
        
        if array_folded_states is None:
            Q_resample = np.zeros((len(sample_indices),n_states))
        else:
            array_folded_states_resample = np.zeros((len(sample_indices),n_states))
        
        # Select the sampled frames from Q and replica_energies:
        j = 0
        
        if array_folded_states is None:
            for i in sample_indices:
                replica_energies_resample[:,:,j] = replica_energies[:,:,i]
                Q_resample[j,:] = Q[i,:]
                j += 1
                
        else:
            for i in sample_indices:
                replica_energies_resample[:,:,j] = replica_energies[:,:,i]
                array_folded_states_resample[j,:] = array_folded_states[i,:]
                j += 1
            
        # Run free energy expectation calculation:
        if array_folded_states is None:
            full_T_list, deltaF_values_boot[i_boot], deltaF_uncertainty_boot[i_boot] = expectations_free_energy(
                Q_resample,
                Q_folded,
                temperature_list,
                bootstrap_energies=replica_energies_resample,
                num_intermediate_states=num_intermediate_states,
            )
        else:
            full_T_list, deltaF_values_boot[i_boot], deltaF_uncertainty_boot[i_boot] = expectations_free_energy(
                None,
                None,
                temperature_list,
                bootstrap_energies=replica_energies_resample,
                num_intermediate_states=num_intermediate_states,
                array_folded_states=array_folded_states_resample,
            )
        
        # Get entropy/enthalpy for fitting current free energy data:
        # The inner dictionary keys will be transition names
        deltaS_values_boot[i_boot] = {}
        deltaU_values_boot[i_boot] = {}
        
        deltaS_values_boot[i_boot], deltaU_values_boot[i_boot] = get_entropy_enthalpy(
            deltaF_values_boot[i_boot], full_T_list)
    
    arr_deltaF_values_boot = {}
    arr_deltaS_values_boot = {}
    arr_deltaU_values_boot = {}
    
    # Loop over all conformational transitions:
    for key, value in deltaF_values_boot[0].items():
        arr_deltaF_values_boot[key] = np.zeros((n_trial_boot, len(full_T_list)))
        arr_deltaS_values_boot[key] = np.zeros((n_trial_boot, len(full_T_list)))
        arr_deltaU_values_boot[key] = np.zeros((n_trial_boot, len(full_T_list)))
        
    # Compute mean values:
    # Free energy:
    for i_boot in range(n_trial_boot):
        for key, value in deltaF_values_boot[i_boot].items():
            arr_deltaF_values_boot[key][i_boot,:] = value.value_in_unit(F_unit)
            
    deltaF_values = {}
    
    for key, value in arr_deltaF_values_boot.items():
        deltaF_values[key] = np.mean(value,axis=0)*F_unit
            
    # Entropy:        
    for i_boot in range(n_trial_boot):
        for key, value in deltaS_values_boot[i_boot].items():    
            arr_deltaS_values_boot[key][i_boot,:] = deltaS_values_boot[i_boot][key].value_in_unit(S_unit)
            
    deltaS_values = {}
    
    for key, value in arr_deltaS_values_boot.items():
        deltaS_values[key] = np.mean(value,axis=0)*S_unit         
            
    # Enthalpy:        
    for i_boot in range(n_trial_boot):
        for key, value in deltaU_values_boot[i_boot].items():
            arr_deltaU_values_boot[key][i_boot,:] = deltaU_values_boot[i_boot][key].value_in_unit(U_unit)
            
    deltaU_values = {}
    
    for key, value in arr_deltaU_values_boot.items():
        deltaU_values[key] = np.mean(value,axis=0)*U_unit
           
    # Compute confidence intervals:
    deltaF_uncertainty = {} 
    deltaS_uncertainty = {} 
    deltaU_uncertainty = {}
    
    if conf_percent == 'sigma':
        # Use analytical standard deviation instead of percentile method:
        # Free energy:
        for key, value in arr_deltaF_values_boot.items():
            F_std = np.std(value,axis=0)*F_unit    
            deltaF_uncertainty[key] = (-F_std,F_std)
            
        # Entropy:    
        for key, value in arr_deltaS_values_boot.items():
            S_std = np.std(value,axis=0)*S_unit    
            deltaS_uncertainty[key] =(-S_std,S_std)
            
        # Enthalpy:    
        for key, value in arr_deltaU_values_boot.items():
            U_std = np.std(value,axis=0)*U_unit
            deltaU_uncertainty[key] = (-U_std,U_std)
    
    else:
        # Compute specified confidence interval:
        p_lo = (100-conf_percent)/2
        p_hi = 100-p_lo
    
        # Free energy:
        for key, value in arr_deltaF_values_boot.items():
            F_diff = value-np.mean(value,axis=0)
            F_conf_lo = np.percentile(F_diff,p_lo,axis=0,interpolation='linear')*F_unit
            F_conf_hi = np.percentile(F_diff,p_hi,axis=0,interpolation='linear')*F_unit
            deltaF_uncertainty[key] = (F_conf_lo, F_conf_hi)
            
        # Entropy:
        for key, value in arr_deltaS_values_boot.items():
            S_diff = value-np.mean(value,axis=0)
            S_conf_lo = np.percentile(S_diff,p_lo,axis=0,interpolation='linear')*S_unit
            S_conf_hi = np.percentile(S_diff,p_hi,axis=0,interpolation='linear')*S_unit
            deltaS_uncertainty[key] = (S_conf_lo, S_conf_hi)
            
        # Enthalpy:
        for key, value in arr_deltaU_values_boot.items():
            U_diff = value-np.mean(value,axis=0)
            U_conf_lo = np.percentile(U_diff,p_lo,axis=0,interpolation='linear')*U_unit
            U_conf_hi = np.percentile(U_diff,p_hi,axis=0,interpolation='linear')*U_unit
            deltaU_uncertainty[key] = (U_conf_lo, U_conf_hi)
    
    # Plot results:
    
    # Free energy:
    plot_free_energy_results(
        full_T_list, deltaF_values, deltaF_uncertainty, plotfile=f"{plotfile_dir}/free_energy_boot.pdf")
        
    # Entropy and enthalpy:
    plot_entropy_enthalpy(
        full_T_list, deltaS_values, deltaU_values,
        deltaS_uncertainty=deltaS_uncertainty, deltaU_uncertainty=deltaU_uncertainty,
        plotfile_entropy=f"{plotfile_dir}/entropy_boot.pdf", plotfile_enthalpy=f"{plotfile_dir}/enthalpy_boot.pdf")
                    
    return full_T_list, deltaF_values, deltaF_uncertainty, deltaS_values, deltaS_uncertainty, deltaU_values, deltaU_uncertainty
    
    
def get_entropy_enthalpy(deltaF, temperature_list):
    """
    Compute enthalpy change and entropy change upon folding, given free energy of folding for a series of temperatures.
    
    :param deltaF: A dictionary containing free energy change for each T in full_T_list, for each conformational state transition.
    :type deltaF: dict{"statei_statej":1D numpy array}
    
    :param temperature_list: List of temperatures for the simulation data.
    :type temperature_list: List( float * simtk.unit.temperature )
    
    :param plotfile_entropy: path to filename for entropy plot (no plot created if None)
    :type plotfile_entropy: str
    
    :param plotfile_enthalpy: path to filename for enthalpy plot (no plot created if None)
    :type plotfile_enthalpy: str
    
    :returns:
      - deltaS - dict{"statei_statej":1D numpy array} of entropy of folding values for each temperature in temperature_list
      - deltaU - dict{"statei_statej":1D numpy array} of enthalpy of folding values for each temperature in temperature_list
      
    """
    
    ddeltaF = {}
    d2deltaF = {}
    spline_tck = {}
    deltaS = {}
    deltaU = {}

    T_unit = temperature_list[0].unit    
    
    # Loop over all conformational transitions:
    for key,value in deltaF.items():
        ddeltaF[key], d2deltaF[key], spline_tck[key] = \
            get_free_energy_derivative(value, temperature_list)
                
        F_unit = value[0].unit
        S_unit = F_unit/T_unit
        U_unit = F_unit
        
        # Spline fitting function strips off units - add back:
        deltaS[key] = -ddeltaF[key] * F_unit / T_unit
        
        deltaU[key] = value + temperature_list*deltaS[key]    
        
    return deltaS, deltaU
  
  
def plot_entropy_enthalpy(
    full_T_list, deltaS_values, deltaU_values, deltaS_uncertainty=None, deltaU_uncertainty=None,
    plotfile_entropy='entropy.pdf', plotfile_enthalpy='enthalpy.pdf'):
    """
    Plot entropy and enthalpy difference data for each conformational state transition as a function of temperature.

    :param full_T_list: Array listing of all temperatures, including sampled and intermediate unsampled
    :type full_T_list: 1D numpy array
    
    :param deltaS_values: A dictionary containing entropy change for each T in full_T_list, for each conformational state transition.
    :type deltaS_values: dict{"statei_statej":1D numpy array}
    
    :param deltaU_values: A dictionary containing enthalpy change for each T in full_T_list, for each conformational state transition.
    :type deltaU_values: dict{"statei_statej":1D numpy array}
    
    :param deltaS_uncertainty: A dictionary containing uncertainties corresponding to deltaS_values (optional)
    :type deltaS_uncertainty: dict{"statei_statej": (1D numpy array, 1D numpy array)}
    
    :param deltaH_uncertainty: A dictionary containing uncertainties corresponding to deltaU_values (optional)
    :type deltaH_uncertainty: dict{"statei_statej": (1D numpy array, 1D numpy array)}
    
    :param plotfile_entropy: name of entropy plot file, including pdf extension
    :type plotfile_entropy: str
    
    :param plotfile_enthalpy: name of enthalpy plot file, including pdf extension
    :type plotfile_enthalpy: str
    """

    T_unit = full_T_list[0].unit
    S_unit = list(deltaS_values.items())[0][1].unit
    U_unit = list(deltaU_values.items())[0][1].unit
    
    xlabel = f'Temperature {T_unit.get_symbol()}'
    
    # Plot entropy change as a function of T:
    ylabel = f'Entropy change {S_unit.get_symbol()}'
    legend_str = []

    if deltaS_uncertainty is not None:
        for key,value in deltaS_values.items():
            if type(deltaS_uncertainty[f"{key}"]) == tuple:
                # Use separate upper and lower errorbars
                deltaS_uncertainty_value = np.zeros((2,len(full_T_list)))
                deltaS_uncertainty_value[0,:] = -deltaS_uncertainty[f"{key}"][0].value_in_unit(S_unit) # Lower error
                deltaS_uncertainty_value[1,:] = deltaS_uncertainty[f"{key}"][1].value_in_unit(S_unit) # Upper error
            else:
                # Use single symmetric errorbar
                deltaS_uncertainty_value = deltaS_uncertainty[f"{key}"].value_in_unit(S_unit)
            
            plt.errorbar(
                full_T_list.value_in_unit(T_unit),
                deltaS_values[f"{key}"].value_in_unit(S_unit),
                deltaS_uncertainty_value,
                linewidth=1,
                markersize=6,
                fmt='o-',
                fillstyle='none',
                capsize=4,
            )
            legend_str.append(key)
    else:
        for key,value in deltaS_values.items():
            plt.plot(
                full_T_list.value_in_unit(T_unit),
                deltaS_values[f"{key}"].value_in_unit(S_unit),
                'o-',
                linewidth=1,
                markersize=6,
                fillstyle='none',
            )
            legend_str.append(key)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    pyplot.legend(legend_str)
    plt.savefig(f"{plotfile_entropy}") 
    plt.close()
    
    # Plot enthalpy change as a function of T:
    ylabel = f'Enthalpy change {U_unit.get_symbol()}'
    legend_str = []

    if deltaU_uncertainty is not None:
        for key,value in deltaU_values.items():
            if type(deltaU_uncertainty[f"{key}"]) == tuple:
                # Use separate upper and lower errorbars
                deltaU_uncertainty_value = np.zeros((2,len(full_T_list)))
                deltaU_uncertainty_value[0,:] = -deltaU_uncertainty[f"{key}"][0].value_in_unit(U_unit) # Lower error
                deltaU_uncertainty_value[1,:] = deltaU_uncertainty[f"{key}"][1].value_in_unit(U_unit) # Upper error
            else:
                # Use single symmetric errorbar
                deltaU_uncertainty_value = deltaU_uncertainty[f"{key}"].value_in_unit(U_unit)
            
            plt.errorbar(
                full_T_list.value_in_unit(T_unit),
                deltaU_values[f"{key}"].value_in_unit(U_unit),
                deltaU_uncertainty_value,
                linewidth=1,
                markersize=6,
                fmt='o-',
                fillstyle='none',
                capsize=4,
            )
            legend_str.append(key)
    else:
        for key,value in deltaU_values.items():
            plt.plot(
                full_T_list.value_in_unit(T_unit),
                deltaU_values[f"{key}"].value_in_unit(U_unit),
                'o-',
                linewidth=1,
                markersize=6,
                fillstyle='none',
            )
            legend_str.append(key)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    pyplot.legend(legend_str)
    plt.savefig(f"{plotfile_enthalpy}") 
    plt.close()
    
    return
    
    
def get_free_energy_derivative(deltaF, temperature_list, plotfile=None):
    """
    Fit a heat capacity vs T dataset to cubic spline, and compute derivatives
    
    :param deltaF: free energy of folding data series
    :type deltaF: Quantity or numpy 1D array
    
    :param temperature_list: List of temperatures used in replica exchange simulations
    :type temperature: Quantity or numpy 1D array
    
    :param plotfile: path to filename to output plot (default=None)
    :type plotfile: str
    
    :returns:
          - dF_out ( 1D numpy array (float) ) - 1st derivative of free energy, from a cubic spline evaluated at each point in deltaF)
          - d2F_out ( 1D numpy array (float) ) - 2nd derivative of free energy, from a cubic spline evaluated at each point in deltaF)
          - spline_tck ( scipy spline object (tuple) ) - knot points (t), coefficients (c), and order of the spline (k) fit to deltaF data
    
    """
    xdata = temperature_list
    ydata = deltaF
    
    # Strip units off quantities:
    if type(xdata[0]) == unit.quantity.Quantity:
        xdata_val = np.zeros((len(xdata)))
        xunit = xdata[0].unit
        for i in range(len(xdata)):
            xdata_val[i] = xdata[i].value_in_unit(xunit)
        xdata = xdata_val
    
    if type(ydata[0]) == unit.quantity.Quantity:
        ydata_val = np.zeros((len(ydata)))
        yunit = ydata[0].unit
        for i in range(len(ydata)):
            ydata_val[i] = ydata[i].value_in_unit(yunit)
        ydata = ydata_val
            
    # Fit cubic spline to data, no smoothing
    spline_tck = interpolate.splrep(xdata, ydata, s=0)
    
    xfine = np.linspace(xdata[0],xdata[-1],1000)
    yfine = interpolate.splev(xfine, spline_tck, der=0)
    dF = interpolate.splev(xfine, spline_tck, der=1)
    d2F = interpolate.splev(xfine, spline_tck, der=2)
    
    dF_out = interpolate.splev(xdata, spline_tck, der=1)
    d2F_out = interpolate.splev(xdata, spline_tck, der=2)
    
    if plotfile != None:
        figure, axs = plt.subplots(nrows=3,ncols=1,sharex=True)
        
        axs[0].plot(xdata,ydata,'ok',
            markersize=4,
            fillstyle='none',
            label='simulation data',
        )
        
        axs[0].plot(xfine,yfine,'-b',
            label='cubic spline',
        )
        
        axs[0].legend()
        axs[0].set_ylabel(r'$\Delta F (J/mol)$')
        
        axs[1].plot(xfine,dF,'-r',
            label=r'$\frac{d\Delta F}{dT}$',
        )
        
        axs[1].legend()
        axs[1].set_ylabel(r'$\frac{d\Delta F}{dT}$')
        
        axs[2].plot(xfine,d2F,'-g',
            label=r'$\frac{d^{2}\Delta F}{dT^{2}}$',
        )
        
        axs[2].legend()
        axs[2].set_ylabel(r'$\frac{d^{2}\Delta F}{dT^{2}}$')
        axs[2].set_xlabel(r'$T (K)$')
        
        plt.tight_layout()
        
        plt.savefig(plotfile)
        plt.close()
    
    return dF_out, d2F_out, spline_tck    
    

def plot_free_energy_results(full_T_list, deltaF_values, deltaF_uncertainty,plotfile="free_energy_plot.pdf"):   
    """
    Plot free energy difference data for each conformational state transition as a function of temperature.

    :param full_T_list: Array listing of all temperatures, including sampled and intermediate unsampled
    :type full_T_list: 1D numpy array
    
    :param deltaF_values: A dictionary containing free energy change for each T in full_T_list, for each conformational state transition.
    :type deltaF_values: dict{"statei_statej":1D numpy array}
    
    :param deltaF_uncertainty: A dictionary containing uncertainties corresponding to deltaF_values
    :type deltaF_uncertainty: dict{"statei_statej": (1D numpy array, 1D numpy array)} or dict{"statei_statej": (1D numpy array)}
    
    :param plotfile: name of file, including pdf extension
    :type plotfile: str
    
    """

    T_unit = full_T_list[0].unit
    F_unit = list(deltaF_values.items())[0][1].unit
    
    xlabel = f'Temperature {T_unit.get_symbol()}'
    ylabel = f'Free energy change {F_unit.get_symbol()}'
    legend_str = []

    for key,value in deltaF_values.items():
        if type(deltaF_uncertainty[f"{key}"]) == tuple:
            # Use separate upper and lower errorbars
            deltaF_uncertainty_value = np.zeros((2,len(full_T_list)))
            deltaF_uncertainty_value[0,:] = -deltaF_uncertainty[f"{key}"][0].value_in_unit(F_unit) # Lower error
            deltaF_uncertainty_value[1,:] = deltaF_uncertainty[f"{key}"][1].value_in_unit(F_unit) # Upper error
        else:
            # Use single symmetric errorbar
            deltaF_uncertainty_value = deltaF_uncertainty[f"{key}"].value_in_unit(F_unit)
        plt.errorbar(
            full_T_list.value_in_unit(T_unit),
            deltaF_values[f"{key}"].value_in_unit(F_unit),
            deltaF_uncertainty_value,
            linewidth=1,
            markersize=6,
            fmt='o-',
            fillstyle='none',
            capsize=4,
        )
        legend_str.append(key)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    pyplot.legend(legend_str)
    plt.savefig(f"{plotfile}")
    plt.close()

    return