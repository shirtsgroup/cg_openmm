import os
import numpy as np
from foldamers.parameters.reweight import *
import matplotlib.pyplot as pyplot
from openmmtools.multistate import MultiStateReporter
from openmmtools.multistate import ReplicaExchangeAnalyzer
import pymbar
from pymbar import timeseries

kB = unit.MOLAR_GAS_CONSTANT_R


def plot_heat_capacity(Cv, dCv, temperature_list, file_name="heat_capacity.pdf"):
    """
        Given an array of temperature-dependent heat capacity values and the uncertainties in their estimates, this function plots the heat capacity curve.
    
        :param Cv: The heat capacity data to plot.
        :type Cv: List( float )
    
        :param dC_v: The uncertainties in the heat capacity data
        :type dCv: List( float )
    
        :param file_name: The name/path of the file where plotting output will be written, default = "heat_capacity.png"
        :type file_name: str
    
        """
    figure = pyplot.figure(1)
    Tunit = temperature_list.unit
    Cvunit = Cv.unit
    temperature_list = np.array(temperature_list)
    Cv = np.array(Cv)
    dCv = np.array(dCv)
    pyplot.errorbar(temperature_list, Cv, yerr=dCv, figure=figure)
    pyplot.xlabel(f"Temperature ({Tunit})")
    pyplot.ylabel(f"C$_v$ ({Cvunit})")
    pyplot.title("Heat capacity as a function of T")
    pyplot.savefig(file_name)
    pyplot.close()
    return


def get_heat_capacity(temperature_list, output_data="output.nc", output_directory="output", num_intermediate_states=0,frac_dT=0.05, plot_file=None):
    """

    Given a .nc output, a temperature list, and a number of intermediate states to insert for the temperature list, this function calculates and plots the heat capacity profile.
    
    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data, default = None          :type output_data: str                                                                                                    
                                                                                                                              
    :param output_directory: directory in which the output data is in, default = "output"                                     
    :type output_data: str    

    :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = None
    :type temperature: List( float * simtk.unit.temperature )
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list'
    :type num_intermediate_states: int

    :param frac_dT: The fraction difference between temperatures points used to calculate finite difference derivatives.
    :type num_intermediate_states: float

    :returns:
          - C_v ( List( float ) ) - The heat capacity values for all (including inserted intermediates) states
          - dC_v ( List( float ) ) - The uncertainty in the heat capacity values for intermediate states
          - new_temp_list ( List( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'C_v'

        """

    # extract reduced energies and the state indices from the .nc  
    reporter = MultiStateReporter(os.path.join(output_directory,output_data), open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)
    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()


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
    num_sampled_T = len(temps)
    n_unsampled_states = 3*(num_sampled_T + (num_sampled_T-1)*num_intermediate_states)
    unsampled_state_energies = np.zeros([n_unsampled_states,n_samples])
    full_T_list = np.zeros(n_unsampled_states)

    # delta is the spacing between temperatures.
    delta = np.zeros(num_sampled_T-1)

    # fill in a list of temperatures at all original temperatures and all intermediate states.
    full_T_list[0] = temps[0]  
    t = 0
    for i in range(num_sampled_T-1):
        delta[i] = (temps[i+1] - temps[i])/(num_intermediate_states+1)
        for j in range(num_intermediate_states+1):
            full_T_list[t] = temps[i] + delta[i]*j
            t += 1
    full_T_list[t] = temps[-1]
    n_T_vals = t+1

    # add additional states for finite difference calculation and the requested spacing/    
    full_T_list[n_T_vals] = full_T_list[0] - delta[0]*frac_dT
    full_T_list[2*n_T_vals] = full_T_list[0] + delta[0]*frac_dT
    for i in range(1,n_T_vals-1):
        ii = i//(num_intermediate_states+1)
        full_T_list[i + n_T_vals] = full_T_list[i] - delta[ii]*frac_dT
        full_T_list[i + 2*n_T_vals] = full_T_list[i] + delta[ii]*frac_dT
    full_T_list[2*n_T_vals-1] = full_T_list[n_T_vals-1] - delta[-1]*frac_dT
    full_T_list[3*n_T_vals-1] = full_T_list[n_T_vals-1] + delta[-1]*frac_dT        

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

    for k in range(n_unsampled_states):
        # get the 'unreduced' potential -- we can't take differences of reduced potentials
        # because the beta is different; math is much more confusing with derivatives of the reduced potentials.
        unsampled_state_energies[k, :] /= beta_full_k[k]

    # we don't actually need these expectations, but this code can be used to validate
    #results = mbarT.computeExpectations(unsampled_state_energies, state_dependent=True)
    #E_expect = results[0]
    #dE_expect = results[1]
    
    # expectations for the differences between states, which we need for numerical derivatives                                               
    results = mbarT.computeExpectations(unsampled_state_energies, output="differences", state_dependent=True)
    DeltaE_expect = results[0]
    dDeltaE_expect = results[1]

    # Now calculate heat capacity (with uncertainties) using the finite difference approach. 
    Cv = np.zeros(n_T_vals)
    dCv = np.zeros(n_T_vals)
    for k in range(n_T_vals):
        im = k+n_T_vals  # +/- delta up and down.
        ip = k+2*n_T_vals
        Cv[k] = (DeltaE_expect[im, ip]) / (full_T_list[ip] - full_T_list[im])
        dCv[k] = (dDeltaE_expect[im, ip]) / (full_T_list[ip] - full_T_list[im])

    # add units so the plot has the rght units.  
    Cv *= unit.kilojoule_per_mole / Tunit # always kJ/mol, since the OpenMM output is in kJ/mol.
    dCv *= unit.kilojoule_per_mole / Tunit
    full_T_list *= Tunit

    # plot and return the heat capacity (with units)
    if plot_file is not None:
        plot_heat_capacity(Cv, dCv, full_T_list[0:n_T_vals],file_name=plot_file)
    return (Cv, dCv, full_T_list[0:n_T_vals])


