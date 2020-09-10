import os
import numpy as np
import simtk.unit as unit
import matplotlib.pyplot as plt
from cg_openmm.utilities.random_builder import *
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeAnalyzer
import pymbar

kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant

def expectations_free_energy(array_folded_states, temperature_list, frame_begin=0, output_directory="output", output_data="output.nc", num_intermediate_states=0):
    """
    This function calculates the free energy difference (with uncertainty) between all conformational states as a function of temperature.

    :param array_folded_states: An array specifying the configurational state of each structure (ranging from 1 to n)
    :type array_folded_states: np.array( int * n_frames*len(temperature_list) ) 

    :param temperature_list: List of temperatures for the simulation data.
    :type temperature_list: List( float * simtk.unit.temperature )
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int    
    
    :param output_directory: Path to simulation output directory which contains a .nc file.
    :type output_directory: str
    
    :param output_data: Name of the simulation .nc file.
    :type output_data: str
    
    :param num_intermediate_states: Number of unsampled thermodynamic states between sampled states to include in the calculation
    :type num_intermediate_states: int
    
    :returns:
      - full_T_list - A 1D numpy array listing of all temperatures, including sampled and intermediate unsampled
      - deltaF_values - A dictionary of the form {"statei_statej": 1D numpy array}, containing free energy change for each T in
                        full_T_list, for each conformational state transition.
      - deltaF uncertainty - A dictionary containing 1D numpy arrays of uncertainties corresponding to deltaF_values

    """


    # Number of configurational states:
    n_conf_states = len(np.unique(array_folded_states))

    # extract reduced energies and the state indices from the .nc
    reporter = MultiStateReporter(os.path.join(output_directory,output_data), open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)
    (
        replica_energies_all,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()
    
    # Select production frames to analyze
    replica_energies = replica_energies_all[:,:,frame_begin:]

    # Check if array_folded_states needs slicing for production region:
    # array_folded_states is array of [nframes,nreplicas]
    if np.size(array_folded_states) != np.size(replica_energies[0]):
        array_folded_states_production = array_folded_states[frame_begin:,:]
        array_folded_states = array_folded_states_production
        
    # Reshape array_folded_states to row vector for pymbar
    # We need to order the data by replica, rather than by frame
    array_folded_states = np.reshape(array_folded_states,(np.size(array_folded_states)),order='F')
        
    # determine the numerical values of beta at each state in units consisten with the temperature
    Tunit = temperature_list[0].unit
    temps = np.array([temp.value_in_unit(Tunit)  for temp in temperature_list])  # should this just be array to begin with
    beta_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * temps)

    # convert the energies from replica/evaluated state/sample form to evaluated state/sample form
    replica_energies = pymbar.utils.kln_to_kn(replica_energies)
    n_samples = len(replica_energies[0,:])

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

    # calculte the expectation of F at each unsampled states
    # we can either do all temperatures at once for one state probability,
    # or all states at once for one temperature. We will choose the second option
    # to capture the correlation of the different conformational states.

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
    # Should change this to kJ/mol
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
                    -kB*full_T_list[i]*unit.kelvin*(
                    np.log(results[str(i)][0][s1])-
                    np.log(results[str(i)][0][s2]))).value_in_unit(F_unit)
                    
                # Get covariance matrix:
                theta_i = results[str(i)][2]
                deltaF_uncertainty[f"state{s1}_state{s2}"][i] = (
                    kB*full_T_list[i]*unit.kelvin*np.sqrt(
                    theta_i[s1,s1] + theta_i[s2,s2] - (theta_i[s2,s1]+theta_i[s1,s2]))).value_in_unit(F_unit)

    return full_T_list, deltaF_values, deltaF_uncertainty
    

def plot_free_energy_results(full_T_list, deltaF_values, deltaF_uncertainty,plotfile="free_energy_plot"):   
    """
    Plot free energy difference data for each conformational state transition as a function of temperature.

    :param full_T_list: Array listing of all temperatures, including sampled and intermediate unsampled
    :type full_T_list: 1D numpy array
    
    :param deltaF_values: A dictionary containing free energy change for each T in full_T_list, for each conformational state transition.
    :type deltaF_values: dict{"statei_statej":1D numpy array}
    
    :param deltaF_uncertainty: A dictionary containing uncertainties corresponding to deltaF_values
    :type deltaF_uncertainty: dict{"statei_statej":1D numpy array}
    
    :param plotfile: name of file, excluding pdf extension
    :type plotfile: str
    
    """

    xlabel = 'Temperature (K)'
    ylabel = 'Free energy change (joule/mol)'
    legend_str = []

    for key,value in deltaF_values.items():
        plt.errorbar(
            full_T_list,
            deltaF_values[f"{key}"],
            deltaF_uncertainty[f"{key}"],
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
    plt.savefig(f"{plotfile}.pdf")

    return