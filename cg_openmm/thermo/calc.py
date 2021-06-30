import os
import numpy as np
from cg_openmm.parameters.reweight import *
import matplotlib.pyplot as plt
from openmmtools.multistate import MultiStateReporter
from openmmtools.multistate import ReplicaExchangeAnalyzer
import pymbar
from pymbar import timeseries
from scipy import interpolate
from sklearn.utils import resample

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
    figure = plt.figure(1)
    Tunit = temperature_list.unit
    Cvunit = Cv.unit
    temperature_list = np.array(temperature_list)
    Cv = np.array(Cv)
    
    if type(dCv) == tuple:
        # Lower and upper uncertainty values given for each point
        # dCv is a tuple of 2 arrays
        dCv_value = np.zeros((2,len(dCv[0])))
        dCv_value[0,:] = -dCv[0].value_in_unit(Cvunit) # Lower error
        dCv_value[1,:] = dCv[1].value_in_unit(Cvunit) # Upper error
        dCv = dCv_value
    else:
        # Single uncertainty value given for each point
        dCv = np.array(dCv)
        
    plt.errorbar(temperature_list, Cv, yerr=dCv, figure=figure)    
    plt.xlabel(f"Temperature ({Tunit})")
    plt.ylabel(f"C$_v$ ({Cvunit})")
    plt.title("Heat capacity as a function of T")
    plt.savefig(file_name)
    plt.close()
    
    return
    

def get_heat_capacity_derivative(Cv, temperature_list, plotfile='dCv_dT.pdf'):
    """
    Fit a heat capacity vs T dataset to cubic spline, and compute derivatives
    
    :param Cv: heat capacity data series
    :type Cv: Quantity or numpy 1D array
    
    :param temperature_list: List of temperatures used in replica exchange simulations
    :type temperature: Quantity or numpy 1D array
    
    :param plotfile: path to filename to output plot
    :type plotfile: str
    
    :returns:
          - dC_v_out ( 1D numpy array (float) ) - 1st derivative of heat capacity, from a cubic spline evaluated at each point in Cv)
          - d2C_v_out ( 1D numpy array (float) ) - 2nd derivative of heat capacity, from a cubic spline evaluated at each point in Cv)
          - spline_tck ( scipy spline object (tuple) ) - knot points (t), coefficients (c), and order of the spline (k) fit to Cv data

    """
    
    xdata = temperature_list
    ydata = Cv
    
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
    dCv = interpolate.splev(xfine, spline_tck, der=1)
    d2Cv = interpolate.splev(xfine, spline_tck, der=2)
    
    dCv_out = interpolate.splev(xdata, spline_tck, der=1)
    d2Cv_out = interpolate.splev(xdata, spline_tck, der=2)
    
    
    figure, axs = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
    )
    
    axs[0].plot(
        xdata,
        ydata,
        'ok',
        markersize=4,
        fillstyle='none',
        label='simulation data',
    )
    
    axs[0].plot(
        xfine,
        yfine,
        '-b',
        label='cubic spline',
    )
    
    axs[0].set_ylabel(r'$C_{V} (kJ/mol/K)$')
    axs[0].legend()
    
    axs[1].plot(
        xfine,
        dCv,
        '-r',
        label=r'$\frac{dC_{V}}{dT}$',
    )
    
    axs[1].legend()
    axs[1].set_ylabel(r'$\frac{dC_{V}}{dT}$')
    
    axs[2].plot(
        xfine,
        d2Cv,
        '-g',
        label=r'$\frac{d^{2}C_{V}}{dT^{2}}$',
    )
    
    axs[2].legend()
    axs[2].set_ylabel(r'$\frac{d^{2}C_{V}}{dT^{2}}$')
    axs[2].set_xlabel(r'$T (K)$')
    
    plt.tight_layout()
    
    plt.savefig(plotfile)
    plt.close()
    
    return dCv_out, d2Cv_out, spline_tck
        

def get_heat_capacity(frame_begin=0, sample_spacing=1, frame_end=-1, output_data="output/output.nc",
    num_intermediate_states=0,frac_dT=0.05, plot_file=None, bootstrap_energies=None):
    """
    Given a .nc output and a number of intermediate states to insert for the temperature list, this function calculates and plots the heat capacity profile.
                             
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int
    
    :param sample_spacing: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData (default=1)
    :type sample_spacing: int
    
    :param frame_end: index of last frame to include in heat capacity calculation (default=-1)
    :type frame_end: int

    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data (default = "output/output.nc")                                                                                          
    :type output_data: str    
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list' (default=0)
    :type num_intermediate_states: int

    :param frac_dT: The fraction difference between temperatures points used to calculate finite difference derivatives (default=0.05)
    :type frac_dT: float
    
    :param plotfile: path to filename to output plot
    :type plotfile: str
    
    :param bootstrap_energies: a custom replica_energies array to be used for bootstrapping calculations. Used instead of the energies in the .nc file.
    :type bootstrap_energies: 3d numpy array (float)

    :returns:
          - C_v ( List( float ) ) - The heat capacity values for all (including inserted intermediates) states
          - dC_v ( List( float ) ) - The uncertainty in the heat capacity values for intermediate states
          - new_temp_list ( List( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'C_v'
          - N_eff( np.array( float ) ) - The number of effective samples at all (including inserted intermediates) states
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
        if frame_end > 0:
            replica_energies = replica_energies_all[:,:,frame_begin:frame_end:sample_spacing]
        else:
            replica_energies = replica_energies_all[:,:,frame_begin::sample_spacing]
    
    # Get the temperature list from .nc file:
    states = reporter.read_thermodynamic_states()[0]
    
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)
    
    # determine the numerical values of beta at each state in units consistent with the temperature
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
    
    N_eff = mbarT.computeEffectiveSampleNumber()

    # Now calculate heat capacity (with uncertainties) using the finite difference approach. 
    Cv = np.zeros(n_T_vals)
    dCv = np.zeros(n_T_vals)
    for k in range(n_T_vals):
        im = k+n_T_vals  # +/- delta up and down.
        ip = k+2*n_T_vals
        Cv[k] = (DeltaE_expect[im, ip]) / (full_T_list[ip] - full_T_list[im])
        dCv[k] = (dDeltaE_expect[im, ip]) / (full_T_list[ip] - full_T_list[im])

    # add units so the plot has the right units.  
    Cv *= unit.kilojoule_per_mole / Tunit # always kJ/mol, since the OpenMM output is in kJ/mol.
    dCv *= unit.kilojoule_per_mole / Tunit
    full_T_list *= Tunit

    # plot and return the heat capacity (with units)
    if plot_file is not None:
        plot_heat_capacity(Cv, dCv, full_T_list[0:n_T_vals],file_name=plot_file)
    return (Cv, dCv, full_T_list[0:n_T_vals], N_eff)

    
def get_heat_capacity_reeval(
    U_kln, output_data="output/output.nc",
    frame_begin=0, sample_spacing=1, frame_end=-1,
    num_intermediate_states=0,frac_dT=0.05,
    plot_file_sim=None, plot_file_reeval=None,
    bootstrap_energies=None,
    ):
    """
    Given an array of re-evaluated energies at a non-simulated set of force field parameters, 
    and a corresponding temperature list, compute heat capacity as a function of temperature. 
    
    :param U_kln: re-evaluated state energies array to be used for the MBAR calculation (first frame is frame_begin)
    :type U_kln: 3d numpy array (float) with dimensions [replica, evaluated_state, frame]
    
    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data (default = "output/output.nc")                                                                                          
    :type output_data: str 
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int
    
    :param sample_spacing: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData (default=1)
    :type sample_spacing: int
    
    :param frame_end: index of last frame to include in heat capacity calculation (default=-1)
    :type frame_end: int
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list' (default=0)
    :type num_intermediate_states: int

    :param frac_dT: The fraction difference between temperatures points used to calculate finite difference derivatives (default=0.05)
    :type num_intermediate_states: float
    
    :param plot_file_sim: path to filename to output plot for simulated heat capacity (default=None)
    :type plot_file_sim: str
    
    :param plot_file_reeval: path to filename to output plot for reevaluated heat capacity (default=None)
    :type plot_file_reeval: str
    
    :param bootstrap_energies: a custom replica_energies array to be used for bootstrapping calculations. Used instead of the energies in the .nc file. (default=None)
    :type bootstrap_energies: 3d numpy array (float)

    :returns:
          - Cv_sim ( np.array( float ) ) - The heat capacity values for all simulated (including inserted intermediates) states
          - dCv_sim ( np.array( float ) ) - The uncertainty of Cv_sim values
          - Cv_reeval ( np.array( float ) ) - The heat capacity values for all reevaluated (including inserted intermediates) states
          - dCv_reeval ( np.array( float ) ) - The uncertainty of Cv_reeval values
          - full_T_list ( np.array( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'C_v'
          - N_eff( np.array( float ) ) - The number of effective samples at all (including inserted intermediates) states
    """    
    
    # Get the original energies that were actually simulated:

    if bootstrap_energies is not None:
        # Use a subsampled replica_energy matrix instead of reading from file
        replica_energies_sampled = bootstrap_energies    
        # Still need to get the thermodynamic states
        reporter = MultiStateReporter(output_data, open_mode="r")
    else:
        # Extract reduced energies and the state indices from the .nc file
        # This is an expensive step for large files
        reporter = MultiStateReporter(output_data, open_mode="r")
        analyzer = ReplicaExchangeAnalyzer(reporter)
        (
            replica_energies_all,
            unsampled_state_energies,
            neighborhoods,
            replica_state_indices,
        ) = analyzer.read_energies()
    
        # Select frames to analyze:
        if frame_end > 0:
            replica_energies_sampled = replica_energies_all[:,:,frame_begin:frame_end:sample_spacing]
        else:
            replica_energies_sampled = replica_energies_all[:,:,frame_begin::sample_spacing]
        
    # Check number of samples:
    if replica_energies_sampled.shape[2] != U_kln.shape[2]:    
        print(f'Error: mismatch in number of frames in simulated ({replica_energies_sampled.shape[2]}) and re-evaluated ({U_kln.shape[2]}) energy arrays')
        exit()
    
    # Get the temperature list from .nc file:
    states = reporter.read_thermodynamic_states()[0]
    
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)    
    
    # determine the numerical values of beta at each state in units consistent with the temperature
    Tunit = temperature_list[0].unit
    temps = np.array([temp.value_in_unit(Tunit)  for temp in temperature_list])  # should this just be array to begin with
    beta_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * temps)
    
    # convert the sampled energies from replica/evaluated state/sample form to evaluated state/sample form
    replica_energies_sampled = pymbar.utils.kln_to_kn(replica_energies_sampled)
    
    # convert the re-evaluated energies from state/evaluated state/sample form to evaluated state/sampled form
    state_energies_reeval = pymbar.utils.kln_to_kn(U_kln)

    n_samples = len(state_energies_reeval[0,:])
    
    # calculate the number of states we need expectations at.  We want it at all of the original
    # temperatures, each intermediate temperature, and then temperatures +/- from the original
    # to take finite derivatives.

    # create  an array for the temperature and energy for each state, including the
    # finite different state.
    num_sampled_T = len(temps)
    n_unsampled_states = 3*(num_sampled_T + (num_sampled_T-1)*num_intermediate_states)
    
    # The first block of unsampled_state_energies is the simulated part, second block is the reevaluated part
    unsampled_state_energies = np.zeros([2*n_unsampled_states,n_samples])
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
    N_k = np.zeros(2*n_unsampled_states)
    for k in range(n_unsampled_states):
        # Calculate the reduced energies at all temperatures, sampled and unsample.
        unsampled_state_energies[k,:] = replica_energies_sampled[0,:]*(beta_full_k[k]/beta_k[0])
        if ti < len(temps):
            # store in N_k which states do and don't have samples.
            if full_T_list[k] == temps[ti]:
                ti += 1
                N_k[k] = n_samples//len(temps)  # these are the states that have samples
 
    # Now, repeat for the reevaluated states:
    for k in range(n_unsampled_states):
        unsampled_state_energies[k+n_unsampled_states,:] = state_energies_reeval[0,:]*(beta_full_k[k]/beta_k[0])
        N_k[k+n_unsampled_states] = 0 # None of these were actually sampled
       
    # call MBAR to find weights at all states, sampled and unsampled
    mbarT = pymbar.MBAR(
        unsampled_state_energies,
        N_k,
        verbose=False,
        relative_tolerance=1e-12
    )

    for k in range(n_unsampled_states):
        # get the 'unreduced' potential -- we can't take differences of reduced potentials
        # because the beta is different; math is much more confusing with derivatives of the reduced potentials.
        unsampled_state_energies[k,:] /= beta_full_k[k]
        unsampled_state_energies[k+n_unsampled_states,:] /= beta_full_k[k]        

    # we don't actually need these expectations, but this code can be used to validate
    #results = mbarT.computeExpectations(unsampled_state_energies, state_dependent=True)
    #E_expect = results[0]
    #dE_expect = results[1]
    
    # expectations for the differences between states, which we need for numerical derivatives                                               
    results = mbarT.computeExpectations(
        unsampled_state_energies,
        output="differences",
        state_dependent=True
    )
    DeltaE_expect = results[0]
    dDeltaE_expect = results[1]

    N_eff = mbarT.computeEffectiveSampleNumber()
    
    # Now calculate heat capacity (with uncertainties) using the finite difference approach. 
    
    # First, for the simulated energies:
    Cv_sim = np.zeros(n_T_vals)
    dCv_sim = np.zeros(n_T_vals)
    for k in range(n_T_vals):
        im = k+n_T_vals  # +/- delta up and down.
        ip = k+2*n_T_vals
        Cv_sim[k] = (DeltaE_expect[im, ip]) / (full_T_list[ip] - full_T_list[im])
        dCv_sim[k] = (dDeltaE_expect[im, ip]) / (full_T_list[ip] - full_T_list[im])

        
    # expectations for the differences between states, which we need for numerical derivatives         
        
    # Next, for the re-evaluated energies:    
    Cv_reeval = np.zeros(n_T_vals)
    dCv_reeval = np.zeros(n_T_vals)
    for k in range(n_T_vals):
        im = k+n_T_vals  # +/- delta up and down.
        ip = k+2*n_T_vals
        Cv_reeval[k] = (DeltaE_expect[n_unsampled_states+im,n_unsampled_states+ip]) / (full_T_list[ip] - full_T_list[im])
        dCv_reeval[k] = (dDeltaE_expect[n_unsampled_states+im,n_unsampled_states+ip]) / (full_T_list[ip] - full_T_list[im])
        
    # add units so the plot has the right units.
    Cv_sim *= unit.kilojoule_per_mole / Tunit # always kJ/mol, since the OpenMM output is in kJ/mol.
    dCv_sim *= unit.kilojoule_per_mole / Tunit
    full_T_list *= Tunit
    
    Cv_reeval *= unit.kilojoule_per_mole / Tunit # always kJ/mol, since the OpenMM output is in kJ/mol.
    dCv_reeval *= unit.kilojoule_per_mole / Tunit

    # plot and return the heat capacity (with units)
    if plot_file_reeval is not None:
        plot_heat_capacity(Cv_reeval, dCv_reeval, full_T_list[0:n_T_vals],file_name=plot_file_reeval)
    if plot_file_sim is not None:
        plot_heat_capacity(Cv_sim, dCv_sim, full_T_list[0:n_T_vals],file_name=plot_file_sim)

    return (Cv_sim, dCv_sim, Cv_reeval, dCv_reeval, full_T_list[0:n_T_vals], N_eff)
    

def bootstrap_heat_capacity(frame_begin=0, sample_spacing=1, frame_end=-1, plot_file='heat_capacity_boot.pdf',
    output_data="output/output.nc", num_intermediate_states=0,frac_dT=0.05,conf_percent='sigma',
    n_trial_boot=200, U_kln=None):
    """
    Calculate and plot the heat capacity curve, with uncertainty determined using bootstrapping.
    Uncorrelated datasets are selected using a random starting frame, repeated n_trial_boot 
    times. Uncertainty in melting point and full-width half maximum of the C_v curve are also returned.
    If a re-evaluated energy array is specified, the heat capacity of the re-evaluated system is
    calculated using MBAR reweighting, based on the original energies in the output.nc file.
    
    .. note::
       If using a re-evaluated energy matrix, U_kln should include all frames after frame_begin
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int
    
    :param sample_spacing: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData (default=1)
    :type sample_spacing: int
    
    :param frame_end: index of last frame to include in heat capacity calculation (default=-1)
    :type frame_end: int

    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data (default = "output/output.nc")                                                                                          
    :type output_data: str    
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list' (default=0)
    :type num_intermediate_states: int

    :param frac_dT: The fraction difference between temperatures points used to calculate finite difference derivatives (default=0.05)
    :type num_intermediate_states: float    
    
    :param conf_percent: Confidence level in percent for outputting uncertainties (default = 68.27 = 1 sigma)
    :type conf_percent: float
    
    :param n_trial_boot: number of trials to run for generating bootstrapping uncertainties (default=200)
    :type n_trial_boot: int
    
    :param U_kln: re-evaluated state energies array to be used for the MBAR calculation (starts at frame_begin)
    :type U_kln: 3d numpy array (float) with dimensions [replica, evaluated_state, frame]
    
    :returns:
       - T_list ( List( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'C_v'
       - C_v_values ( List( float * kJ/mol/K ) ) - The heat capacity values for all (including inserted intermediates) states
       - C_v_uncertainty ( Tuple ( np.array(float) * kJ/mol/K ) ) - confidence interval for all C_v_values computed from bootstrapping
       - Tm_value ( float * unit.simtk.temperature ) - Melting point mean value computed from bootstrapping
       - Tm_uncertainty ( Tuple ( float * unit.simtk.temperature ) ) - confidence interval for melting point computed from bootstrapping
       - FWHM_value ( float * unit.simtk.temperature ) - C_v full width half maximum mean value computed from bootstrapping
       - FWHM_uncertainty ( Tuple ( float * unit.simtk.temperature ) ) - confidence interval for C_v full width half maximum computed from bootstrapping
       - N_eff_values ( np.array( float ) ) -  The bootstrap mean number of effective samples at all simulated and non-simulated (including inserted intermediates) states
    
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
    
    # Store data for each sampling trial:
    C_v_values_boot = {}
    C_v_uncertainty_boot = {}
    N_eff_boot = {}
    
    Tm_boot = np.zeros(n_trial_boot)
    Cv_height = np.zeros(n_trial_boot)
    FWHM = np.zeros(n_trial_boot)

    # Save the full re-evaluated energy array
    U_kln_all = U_kln
    
    for i_boot in range(n_trial_boot):
    
        # Select production frames to analyze
        # Here we can potentially change the reference frame for each bootstrap trial.
        ref_shift = np.random.randint(sample_spacing)
        # Depending on the reference frame, there may be small differences in numbers of samples per bootstrap trial
        if frame_end > 0:
            replica_energies = replica_energies_all[:,:,(frame_begin+ref_shift):frame_end:sample_spacing]
        else:
            replica_energies = replica_energies_all[:,:,(frame_begin+ref_shift)::sample_spacing]
    
        # Get all possible sample indices
        sample_indices_all = np.arange(0,len(replica_energies[0,0,:]))
        # n_samples should match the size of the sliced replica energy dataset
        sample_indices = resample(sample_indices_all, replace=True, n_samples=len(sample_indices_all))
        
        n_state = replica_energies.shape[0]
        
        # replica_energies is [n_states x n_states x n_frame]        
        replica_energies_resample = np.zeros_like(replica_energies)
        
        if U_kln is not None:
            if frame_end > 0:
                # U_kln should not include the equilibration region
                U_kln = U_kln_all[:,:,ref_shift:frame_end:sample_spacing]
            else:
                U_kln = U_kln_all[:,:,ref_shift::sample_spacing]
            U_kln_resample = np.zeros_like(U_kln)      
        
        # Select the sampled frames from array_folded_states and replica_energies:
        j = 0
        for i in sample_indices:
            replica_energies_resample[:,:,j] = replica_energies[:,:,i]
            if U_kln is not None:
                U_kln_resample[:,:,j] = U_kln[:,:,i]
            j += 1
            
        if U_kln is not None:
            # Run heat capacity calculation for re-evaluated system:
            (Cv_sim, dCv_sim, C_v_values_boot[i_boot], C_v_uncertainty_boot[i_boot],
            T_list, N_eff_boot[i_boot]) = get_heat_capacity_reeval(
                U_kln_resample,
                output_data=output_data,
                frame_begin=frame_begin,
                frame_end=frame_end,
                sample_spacing=sample_spacing,
                num_intermediate_states=num_intermediate_states,
                frac_dT=frac_dT,
                plot_file_sim=None,
                plot_file_reeval=None,
                bootstrap_energies=replica_energies_resample,
                )
        else:    
            # Run standard heat capacity expectation calculation:
            (C_v_values_boot[i_boot], C_v_uncertainty_boot[i_boot], T_list,
            N_eff_boot[i_boot]) = get_heat_capacity(
                output_data=output_data,
                num_intermediate_states=num_intermediate_states,
                frac_dT=frac_dT,
                plot_file=None,
                bootstrap_energies=replica_energies_resample,
                )
            
        if i_boot == 0:
            # Get units:
            C_v_unit = C_v_values_boot[0][0].unit
            T_unit = T_list[0].unit    
            
        # Compute the melting point:
        max_index = np.argmax(C_v_values_boot[i_boot])
        Tm_boot[i_boot] = T_list[max_index].value_in_unit(T_unit)
        
        # Compute the peak height, relative to lowest C_v value in the temp range:
        Cv_height[i_boot] = (np.max(C_v_values_boot[i_boot])-np.min(C_v_values_boot[i_boot])).value_in_unit(C_v_unit)
        
        # Compute the FWHM:
        # C_v value at half-maximum:
        mid_val = np.min(C_v_values_boot[i_boot]).value_in_unit(C_v_unit) + Cv_height[i_boot]/2
        
        #***Note: this assumes that there is only a single heat capacity peak, with
        # monotonic behavior on each side of the peak.
        
        half_lo_found = False
        half_hi_found = False
        
        T_half_lo = None
        T_half_hi = None
        
        # Reverse scan for lower half:
        k = 1
        while half_lo_found == False:
            index = max_index-k
            if index < 0:
                # The lower range does not contain the lower midpoint
                break
            else:    
                curr_val = C_v_values_boot[i_boot][index].value_in_unit(C_v_unit)
                prev_val = C_v_values_boot[i_boot][index+1].value_in_unit(C_v_unit)
                
            if curr_val <= mid_val:
                # The lower midpoint lies within T[index] and T[index+1]
                # Interpolate solution:
                T_half_lo = T_list[index]+(mid_val-curr_val)*(T_list[index+1]-T_list[index])/(prev_val-curr_val)
                half_lo_found = True
            else:
                k += 1
                
        # Forward scan for upper half:
        m = 1

        while half_hi_found == False:
            index = max_index+m
            if index == len(T_list):
                # The upper range does not contain the upper midpoint
                break
            else:
                curr_val = C_v_values_boot[i_boot][index].value_in_unit(C_v_unit)
                prev_val = C_v_values_boot[i_boot][index-1].value_in_unit(C_v_unit)
            if curr_val <= mid_val:
                # The upper midpoint lies within T[index] and T[index-1]
                # Interpolate solution:
                T_half_hi = T_list[index]+(mid_val-curr_val)*(T_list[index-1]-T_list[index])/(prev_val-curr_val)
                half_hi_found = True
            else:
                m += 1
        
        if half_lo_found and half_hi_found:
            FWHM[i_boot] = (T_half_hi-T_half_lo).value_in_unit(T_unit)
        elif half_lo_found == True and half_hi_found == False:
            FWHM[i_boot] = 2*(Tm_boot[i_boot]-T_half_lo.value_in_unit(T_unit))
        elif half_lo_found == False and half_hi_found == True:
            FWHM[i_boot] = 2*(T_half_hi.value_in_unit(T_unit)-Tm_boot[i_boot])
        
    # Compute uncertainty at all temps in T_list over the n_trial_boot trials performed:
    
    # Convert dicts to array
    arr_C_v_values_boot = np.zeros((n_trial_boot, len(T_list)))
    arr_N_eff_boot = np.zeros((n_trial_boot, len(N_eff_boot[0])))
    
    for i_boot in range(n_trial_boot):
        arr_C_v_values_boot[i_boot,:] = C_v_values_boot[i_boot].value_in_unit(C_v_unit)
        arr_N_eff_boot[i_boot,:] = N_eff_boot[i_boot]
            
    # Compute mean values:        
    C_v_values = np.mean(arr_C_v_values_boot,axis=0)*C_v_unit      
    Cv_height_value = np.mean(Cv_height)*C_v_unit      
    Tm_value = np.mean(Tm_boot)*T_unit
    FWHM_value = np.mean(FWHM)*T_unit
    N_eff_values = np.mean(arr_N_eff_boot)
    
    # Compute confidence intervals:
    if conf_percent == 'sigma':
        # Use analytical standard deviation instead of percentile method:
        
        # C_v values:
        C_v_std = np.std(arr_C_v_values_boot,axis=0)
        C_v_uncertainty = (-C_v_std*C_v_unit, C_v_std*C_v_unit)
        
        # C_v peak height:
        Cv_height_std = np.std(Cv_height)
        Cv_height_uncertainty = (-Cv_height_std*C_v_unit, Cv_height_std*C_v_unit)   
        
        # Melting point:
        Tm_std = np.std(Tm_boot)
        Tm_uncertainty = (-Tm_std*T_unit, Tm_std*T_unit)
        
        # Full width half maximum:
        FWHM_std = np.std(FWHM)
        FWHM_uncertainty = (-FWHM_std*T_unit, FWHM_std*T_unit)
        
    else:
        # Compute specified confidence interval:
        p_lo = (100-conf_percent)/2
        p_hi = 100-p_lo
                
        # C_v values:
        C_v_diff = arr_C_v_values_boot-np.mean(arr_C_v_values_boot,axis=0)
        C_v_conf_lo = np.percentile(C_v_diff,p_lo,axis=0,interpolation='linear')
        C_v_conf_hi = np.percentile(C_v_diff,p_hi,axis=0,interpolation='linear')
      
        C_v_uncertainty = (C_v_conf_lo*C_v_unit, C_v_conf_hi*C_v_unit) 
                    
        # C_v peak height:                
        Cv_height_diff = Cv_height-np.mean(Cv_height)
        Cv_height_conf_lo = np.percentile(Cv_height_diff,p_lo,interpolation='linear')
        Cv_height_conf_hi = np.percentile(Cv_height_diff,p_hi,interpolation='linear')
        
        Cv_height_uncertainty = (Cv_height_conf_lo*C_v_unit, Cv_height_conf_hi*C_v_unit)                  
        
        # Melting point: 
        Tm_diff = Tm_boot-np.mean(Tm_boot)
        Tm_conf_lo = np.percentile(Tm_diff,p_lo,interpolation='linear')
        Tm_conf_hi = np.percentile(Tm_diff,p_hi,interpolation='linear')
        
        Tm_uncertainty = (Tm_conf_lo*T_unit, Tm_conf_hi*T_unit)  
        
        # Full width half maximum:
        FWHM_diff = FWHM-np.mean(FWHM)
        FWHM_conf_lo = np.percentile(FWHM_diff,p_lo,interpolation='linear')
        FWHM_conf_hi = np.percentile(FWHM_diff,p_hi,interpolation='linear')
        
        FWHM_uncertainty = (FWHM_conf_lo*T_unit, FWHM_conf_hi*T_unit) 
    
    # Plot and return the heat capacity (with units)
    if plot_file is not None:
        plot_heat_capacity(C_v_values, C_v_uncertainty, T_list, file_name=plot_file)
                    
    return T_list, C_v_values, C_v_uncertainty, Tm_value, Tm_uncertainty, Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty, N_eff_values
        
    