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
    :type Cv: List( float * kJ/mol/K )

    :param dCv: The uncertainties in the heat capacity data
    :type dCv: List( float * kJ/mol/K )
    
    :param temperature_list: List of temperatures used in replica exchange simulations
    :type temperature: Quantity or numpy 1D array        

    :param file_name: The name/path of the file where plotting output will be written, default = "heat_capacity.pdf"
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
    

def plot_partial_heat_capacities(Cv_partial, dCv, temperature_list, file_name="heat_capacity_partial.pdf"):
    """
    Given an array of temperature-dependent heat capacity values and the uncertainties in their estimates, this function plots the heat capacity curve.

    :param Cv_partial: The heat capacity data to plot, grouped by conformational state.
    :type Cv_partial: dict ( list( float * kJ/mol/K ) )

    :param dCv: The uncertainties corresponding to Cv_partial
    :type dCv: dict ( list( float * kJ/mol/K ) )
    
    :param temperature_list: List of temperatures used in replica exchange simulations
    :type temperature: Quantity or numpy 1D array        

    :param file_name: The name/path of the file where plotting output will be written, default = "heat_capacity_partial.pdf"
    :type file_name: str
    """
    
    fig, ax = plt.subplots()
    Tunit = temperature_list.unit
    temperature_list = np.array(temperature_list)
    
    for key, value in Cv_partial.items(): 
        Cvunit = value.unit
        
        Cv = np.array(value)    
        
        if dCv is not None:
            if type(dCv[key]) == tuple:
                # Lower and upper uncertainty values given for each point
                # dCv is dict mapping conformational state to a tuple of 2 arrays
                dCv_value = np.zeros((2,len(dCv[key][0])))
                dCv_value[0,:] = -dCv[key][0].value_in_unit(Cvunit) # Lower error
                dCv_value[1,:] = dCv[key][1].value_in_unit(Cvunit) # Upper error
                dCv[key] = dCv_value
            else:
                # Single uncertainty value given for each point
                dCv[key] = np.array(dCv[key])

            plt.errorbar(
                temperature_list,
                Cv,
                yerr=dCv[key],
                label=f'state {key}',
                )
        
        else:
            # Uncertainty is None
            plt.errorbar(
                temperature_list,
                Cv,
                label=f'state {key}',
                )

    plt.legend(
        loc='upper left',
        fontsize=6
        )    
        
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
          - dCv_out ( 1D numpy array (float) ) - 1st derivative of heat capacity, from a cubic spline evaluated at each point in Cv)
          - d2Cv_out ( 1D numpy array (float) ) - 2nd derivative of heat capacity, from a cubic spline evaluated at each point in Cv)
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
          - Cv ( List( float * kJ/mol/K ) ) - The heat capacity values for all (including inserted intermediates) states
          - dCv ( List( float * kJ/mol/K ) ) - The uncertainty in the heat capacity values for intermediate states
          - new_temp_list ( List( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'Cv'
          - FWHM ( float * unit.simtk.temperature ) - Full width half maximum from heat capacity vs T
          - Tm ( float * unit.simtk.temperature ) - Melting point from heat capacity vs T
          - Cv_height ( float * kJ/mol/K ) - Relative height of heat capacity peak
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
    
    # Close the data file - repeatedly opening the same .nc can cause seg faults:
    reporter.close()       
    
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

    # Now get the full-width half-maximum, melting point, and Cv peak height.
    (FWHM, Tm, Cv_height) = get_cv_FWHM(Cv, full_T_list[0:n_T_vals])

    # add units so the plot has the right units.  
    Cv *= unit.kilojoule_per_mole / Tunit # always kJ/mol, since the OpenMM output is in kJ/mol.
    dCv *= unit.kilojoule_per_mole / Tunit
    full_T_list *= Tunit
    FWHM *= Tunit
    Tm *= Tunit
    Cv_height *= unit.kilojoule_per_mole / Tunit

    # plot and return the heat capacity (with units)
    if plot_file is not None:
        plot_heat_capacity(Cv, dCv, full_T_list[0:n_T_vals],file_name=plot_file)
        
    return (Cv, dCv, full_T_list[0:n_T_vals], FWHM, Tm, Cv_height, N_eff)


def get_partial_heat_capacities(array_folded_states,
    frame_begin=0, sample_spacing=1, frame_end=-1, output_data="output/output.nc",
    num_intermediate_states=0,frac_dT=0.05, plot_file=None,
    bootstrap_energies=None):
    """
    Given an array classifying each frame into discrete conformational states, compute the heat capacity curve,
    and contributions from each conformational class to the heat capacity curve. Uncertainties can be
    calculated using the function bootstrap_partial_heat_capacities.
    
    ***TODO: add the uncertainty calculation
    
    .. note::
       array_folded_states should include all frames after frame_begin

    .. note::
       The partial contributions to heat capacity must be weighted by the probabilities of finding each 
       conformational class to reconstruct the total heat capacity curve.
       
    :param array_folded_states: a precomputed array classifying the different conformational states
    :type array_folded_states: 2d numpy array (int)   
         
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
          - Cv ( dict ( List( float * kJ/mol/K  ) ) ) - For each conformational class, the heat capacity values for all (including inserted intermediates) states
          - ***dCv ( dict ( List( float * kJ/mol/K ) ) ) - The uncertainty in the heat capacity values for intermediate states (not yet implemented)
          - new_temp_list ( List( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'Cv'
          - FWHM_partial ( dict ( float * unit.simtk.temperature ) ) - For each conformational class, full width half maximum from heat capacity vs T
          - Tm_partial ( dict ( float * unit.simtk.temperature ) ) - For each conformational class, melting point from heat capacity vs T
          - Cv_height_partial ( dict ( float * kJ/mol/K ) ) - For each conformational class, relative height of heat capacity peak
          - U_expect_confs ( dict ( np.array( float * kJ/mol ) ) - For each conformational class, the energy expectations (in kJ/mol) at each T, including intermediate states
          - N_eff_partial ( dict ( np.array( float ) ) ) - For each conformational class, the number of effective samples contributing to that state's heat capacity expectation
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
            
        # Also apply frame selection to array_folded_states (starts at frame_begin):
        if frame_end > 0:
            array_folded_states = array_folded_states[:frame_end:sample_spacing,:]
        else:
            array_folded_states = array_folded_states[::sample_spacing,:]
     
    # Get the temperature list from .nc file:
    states = reporter.read_thermodynamic_states()[0]
    
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)
    
    # Close the data file - repeatedly opening the same .nc can cause seg faults:
    reporter.close()    
    
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

    # Now get the partial heat capacities using the mbar weights:
    # <O> = \sum w_i O_i / \sum w_i.
    
    # Get the weights from the mbar object:
    # These weights include all of the intermediate and finite difference temperatures 
    w_nk = mbarT.W_nk

    # These are of shape [n_samples,n_states]
    # where n_samples is n_replica*n_frames
    
    # Reshape array_folded_states to row vector for pymbar
    # (analogous to the free energy calculation with array_folded_states)
    # We need to order the data by replica, rather than by frame
    array_folded_states = np.reshape(array_folded_states,(np.size(array_folded_states)),order='F')       
        
    # Here we need to classify each of the sample mbar weights
    num_confs = {} # w_i*U_i, for each state.
    den_confs = {} # w_i
    N_eff_partial = {} # Number of effective samples for each conformational state and temperature
    w_nk_sq_sum = {}
    U_expect_confs = {} # Energy expectations
    
    n_conf_states = len(np.unique(array_folded_states))    
    
    for c in range(n_conf_states):
        num_confs[c] = 0
        den_confs[c] = 0
        w_nk_sq_sum[c] = 0
        
    for frame in range(len(array_folded_states)):
        # Sum the weighted energies for partial heat capacity eval:
        # These energies should be the same unsampled_state_energies used in the computeExpectations calculation
        
        # Also compute the effective numbers of sample contributing to each partial Cv:
        # N_eff,k = (sum(w_nk))^2/(sum(w_nk^2))
        
        for c in range(n_conf_states):
            if array_folded_states[frame] == c:
                num_confs[c] += w_nk[frame,:]*unsampled_state_energies[:,frame]
                den_confs[c] += w_nk[frame,:]
                w_nk_sq_sum[c] += w_nk[frame,:]**2
                break
                
    # Compute the expectations for each conformational class:
    # These are also needed in the heat capacity decomposition,
    # for the term involving the square of the energy expectation
    # differences.
    
    for c in range(n_conf_states):
        U_expect_confs[c] = num_confs[c]/den_confs[c]
        N_eff_partial[c] = (den_confs[c])**2/w_nk_sq_sum[c]
        
    # Compute the array of differences to use for the heat capacity calculation:
    U_expect_confs_diff = {}
    for c in range(n_conf_states):
        U_expect_confs_diff[c] = np.zeros((len(U_expect_confs[c]),len(U_expect_confs[c])))
        for i in range(len(U_expect_confs[c])):
            for j in range(len(U_expect_confs[c])):
                U_expect_confs_diff[c][i,j] = np.abs(U_expect_confs[c][i]-U_expect_confs[c][j])
        
    # Now compute the heat capacities:
    Cv_partial = {} 
    
    for c in range(n_conf_states):
        Cv_partial[c] = np.zeros(n_T_vals)
        for k in range(n_T_vals):
            im = k+n_T_vals  # +/- delta up and down.
            ip = k+2*n_T_vals
            Cv_partial[c][k] = (U_expect_confs_diff[c][im, ip]) / (full_T_list[ip] - full_T_list[im])

    # Now get the full-width half-maximum, melting point, and Cv peak height.
    # Store these are dicts mapping {state:value}
    FWHM_partial = {}
    Tm_partial = {}
    Cv_height_partial = {}
    
    for c in range(n_conf_states):
        (FWHM_curr, Tm_curr, Cv_height_curr) = get_cv_FWHM(Cv_partial[c], full_T_list[0:n_T_vals])
        
        FWHM_partial[c] = FWHM_curr * Tunit
        Tm_partial[c] = Tm_curr * Tunit
        Cv_height_partial[c] = Cv_height_curr * unit.kilojoule_per_mole / Tunit

    # add units so the plot has the right units.  
    for c in range(n_conf_states):
        Cv_partial[c] *= unit.kilojoule_per_mole / Tunit # always kJ/mol, since the OpenMM output is in kJ/mol.
    full_T_list *= Tunit

    # plot and return the heat capacity (with units)
    if plot_file is not None:
        plot_partial_heat_capacities(Cv_partial, None, full_T_list[0:n_T_vals],file_name=plot_file)
        
    # Uncertainty directly from mbar not implemented here yet    
    dCv_partial = None
    
    # Slice the U_expect_confs arrays to correspond to the output temperature list, and add back units:
    # Also slice the N_eff_partial array:
    for c in range(n_conf_states):
        U_expect_confs[c] = U_expect_confs[c][0:n_T_vals]*unit.kilojoule_per_mole
        N_eff_partial[c] = N_eff_partial[c][0:n_T_vals]
    
    return (Cv_partial, dCv_partial, full_T_list[0:n_T_vals], FWHM_partial, Tm_partial, Cv_height_partial, U_expect_confs, N_eff_partial)

    
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
    :type frac_dT: float
    
    :param plot_file_sim: path to filename to output plot for simulated heat capacity (default=None)
    :type plot_file_sim: str
    
    :param plot_file_reeval: path to filename to output plot for reevaluated heat capacity (default=None)
    :type plot_file_reeval: str
    
    :param bootstrap_energies: a custom replica_energies array to be used for bootstrapping calculations. Used instead of the energies in the .nc file. (default=None)
    :type bootstrap_energies: 3d numpy array (float)

    :returns:
          - Cv_sim ( np.array( float * kJ/mol/K ) ) - The heat capacity values for all simulated (including inserted intermediates) states
          - dCv_sim ( np.array( float * kJ/mol/K ) ) - The uncertainty of Cv_sim values
          - Cv_reeval ( np.array( float * kJ/mol/K ) ) - The heat capacity values for all reevaluated (including inserted intermediates) states
          - dCv_reeval ( np.array( float * kJ/mol/K ) ) - The uncertainty of Cv_reeval values
          - full_T_list ( np.array( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'Cv'
          - FWHM ( float * unit.simtk.temperature  ) - Full width half maximum from heat capacity vs T
          - Tm ( float * unit.simtk.temperature  ) - Melting point from heat capacity vs T
          - Cv_height ( float * kJ/mol/K ) - Relative height of heat capacity peak
          - N_eff( np.array( float ) ) - The number of effective samples at all (including inserted intermediates) states
    """    
    
    # Get the original energies that were actually simulated:

    if bootstrap_energies is not None:
        # Use a subsampled replica_energy matrix instead of reading from file
        # ***Multiple trajectories: bootstrap_energies will be a list of arrays.
        # We also need to get the reporter for each separate simulation.
        replica_energies_sampled = bootstrap_energies    
        # Still need to get the thermodynamic states
        reporter = MultiStateReporter(output_data, open_mode="r")
    else:
        # Extract reduced energies and the state indices from the .nc file
        # This is an expensive step for large files
        
        # ***Multiple trajectories: get the energies from file for each reference simulation
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
    # ***Multiple trajectories: check consistency of each pair
    if replica_energies_sampled.shape[2] != U_kln.shape[2]:    
        print(f'Error: mismatch in number of frames in simulated ({replica_energies_sampled.shape[2]}) and re-evaluated ({U_kln.shape[2]}) energy arrays')
        exit()
    
    # Get the temperature list from .nc file:
    # ***Multiple trajectories: the reference simulations may have different temperature lists - get each
    states = reporter.read_thermodynamic_states()[0]
    
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)    
    
    # Close the data file - repeatedly opening the same .nc can cause seg faults:
    reporter.close()    
    
    # determine the numerical values of beta at each state in units consistent with the temperature
    Tunit = temperature_list[0].unit
    temps = np.array([temp.value_in_unit(Tunit)  for temp in temperature_list])  # should this just be array to begin with
    beta_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * temps)
    
    # convert the sampled energies from replica/evaluated state/sample form to evaluated state/sample form
    # ***Multiple trajectories: convert arrays for each reference simulation
    replica_energies_sampled = pymbar.utils.kln_to_kn(replica_energies_sampled)
    
    # convert the re-evaluated energies from state/evaluated state/sample form to evaluated state/sampled form
    state_energies_reeval = pymbar.utils.kln_to_kn(U_kln)

    n_samples = len(state_energies_reeval[0,:])
    # ***Multiple trajectories: n_samples should be the same for all reference trajectories
    
    # calculate the number of states we need expectations at.  We want it at all of the original
    # temperatures, each intermediate temperature, and then temperatures +/- from the original
    # to take finite derivatives.
    # ***Multiple trajectories: Evaluate at all sampled temperatures across all reference simulations,
    # their intermediates, and the +/- finite difference temperatures. We need to keep track of what
    # was explicitly simulated or not in the N_k array.

    # create an array for the temperature and energy for each state, including the
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

    # Next, for the re-evaluated energies:    
    Cv_reeval = np.zeros(n_T_vals)
    dCv_reeval = np.zeros(n_T_vals)
    for k in range(n_T_vals):
        im = k+n_T_vals  # +/- delta up and down.
        ip = k+2*n_T_vals
        Cv_reeval[k] = (DeltaE_expect[n_unsampled_states+im,n_unsampled_states+ip]) / (full_T_list[ip] - full_T_list[im])
        dCv_reeval[k] = (dDeltaE_expect[n_unsampled_states+im,n_unsampled_states+ip]) / (full_T_list[ip] - full_T_list[im])
          
    # Now get the full-width half-maximum, melting point, and Cv peak height.
    (FWHM, Tm, Cv_height) = get_cv_FWHM(Cv_reeval, full_T_list[0:n_T_vals])
        
    # add units so the plot has the right units.
    Cv_sim *= unit.kilojoule_per_mole / Tunit # always kJ/mol, since the OpenMM output is in kJ/mol.
    dCv_sim *= unit.kilojoule_per_mole / Tunit
    full_T_list *= Tunit
    
    Cv_reeval *= unit.kilojoule_per_mole / Tunit
    dCv_reeval *= unit.kilojoule_per_mole / Tunit
    
    FWHM *= Tunit
    Tm *= Tunit
    Cv_height *= unit.kilojoule_per_mole / Tunit

    # plot and return the heat capacity (with units)
    if plot_file_reeval is not None:
        plot_heat_capacity(Cv_reeval, dCv_reeval, full_T_list[0:n_T_vals],file_name=plot_file_reeval)
    if plot_file_sim is not None:
        plot_heat_capacity(Cv_sim, dCv_sim, full_T_list[0:n_T_vals],file_name=plot_file_sim)

    return (Cv_sim, dCv_sim, Cv_reeval, dCv_reeval, full_T_list[0:n_T_vals], FWHM, Tm, Cv_height, N_eff)
    
    
def bootstrap_partial_heat_capacities(array_folded_states,
    frame_begin=0, sample_spacing=1, frame_end=-1,
    plot_file='partial_heat_capacity_boot.pdf',output_data="output/output.nc",
    num_intermediate_states=0,frac_dT=0.05,conf_percent='sigma',n_trial_boot=200):
    
    """
    Given an array classifying each frame into discrete conformational states, compute the heat capacity curve,
    and contributions from each conformational class to the heat capacity curve, with uncertainties determined
    using bootstrapping. Uncorrelated datasets are selected using a random starting frame, repeated n_trial_boot 
    times.
    
    .. note::
       array_folded_states should include all frames after frame_begin

    .. note::
       The partial contributions to heat capacity must be weighted by the probabilities of finding each 
       conformational class to reconstruct the total heat capacity curve.
    
    :param array_folded_states: a precomputed array classifying the different conformational states
    :type array_folded_states: 2d numpy array (int)    
    
    :param frame_begin: index of first frame defining the range of samples to use as a production period (default=0)
    :type frame_begin: int
    
    :param sample_spacing: spacing of uncorrelated data points, for example determined from pymbar timeseries subsampleCorrelatedData (default=1)
    :type sample_spacing: int
    
    :param frame_end: index of last frame to include in heat capacity calculation (default=-1)
    :type frame_end: int
    
    :param plot_file: path to filename to output plot (default='heat_capacity_boot.pdf')
    :type plot_file: str

    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data (default = "output/output.nc")                                                                                          
    :type output_data: str    
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list' (default=0)
    :type num_intermediate_states: int

    :param frac_dT: The fraction difference between temperatures points used to calculate finite difference derivatives (default=0.05)
    :type frac_dT: float    
    
    :param conf_percent: Confidence level in percent for outputting uncertainties (default = 68.27 = 1 sigma)
    :type conf_percent: float
    
    :param n_trial_boot: number of trials to run for generating bootstrapping uncertainties (default=200)
    :type n_trial_boot: int   

    :returns:
       - T_list ( np.array ( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'Cv'
       - Cv_values ( dict ( np.array ( float * kJ/mol/K ) ) ) - The heat capacity values for all (including inserted intermediates) states
       - Cv_uncertainty ( dict ( Tuple ( np.array(float) * kJ/mol/K ) ) ) - confidence interval for all Cv_values computed from bootstrapping
       - Tm_value ( dict ( float * unit.simtk.temperature ) ) - Melting point mean value computed from bootstrapping
       - Tm_uncertainty ( dict ( Tuple ( float * unit.simtk.temperature ) ) ) - confidence interval for melting point computed from bootstrapping
       - Cv_height_value ( dict ( float * kJ/mol/K ) ) - Height of the Cv peak relative to the lowest value over the temperature range used.
       - Cv_height_uncertainty ( dict ( Tuple ( np.array(float) * kJ/mol/K ) ) ) - confidence interval for all Cv_height_value computed from bootstrapping
       - FWHM_value ( dict ( float * unit.simtk.temperature ) ) - Cv full width half maximum mean value computed from bootstrapping
       - FWHM_uncertainty ( dict ( Tuple ( float * unit.simtk.temperature ) ) ) - confidence interval for Cv full width half maximum computed from bootstrapping
       - U_expect_values ( dict ( np.array ( float * kJ/mol ) ) ) - Energy expectation values for each conformational state, at each T in T_list
       - U_expect_uncertainty ( dict ( Tuple ( np.array(float) * kJ/mol ) ) ) - confidence interval for U_expect_values computed from bootstrapping
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
    
    # Close the data file - repeatedly opening the same .nc can cause seg faults:
    reporter.close()
    
    if frame_end > 0:
        replica_energies_all = replica_energies_all[:,:,frame_begin:frame_end]
    else:
        replica_energies_all = replica_energies_all[:,:,frame_begin::]
        
    # Check the size of array_folded_states:
    if np.shape(replica_energies_all)[2] != np.shape(array_folded_states)[0]:
        print(f'Error: mismatch in number of samples in array_folded_states and specified frames of replica energies')
        exit()    
        
    # Get the number of conformational state classifications:
    n_conf_states = len(np.unique(array_folded_states))    
        
    # Store data for each sampling trial:
    Cv_partial_values_boot = {}
    Cv_partial_uncertainty_boot = {}
    
    U_expect_confs_boot = {}
    
    N_eff_partial_boot = {}
    
    Tm_boot = {}
    Cv_height_boot = {}
    FWHM_boot = {}
    
    for c in range(n_conf_states):
        Tm_boot[c] = np.zeros(n_trial_boot)
        Cv_height_boot[c] = np.zeros(n_trial_boot)
        FWHM_boot[c] = np.zeros(n_trial_boot)
    
    for i_boot in range(n_trial_boot):
    
        # Select production frames to analyze
        ref_shift = np.random.randint(sample_spacing)
        # Depending on the reference frame, there may be small differences in numbers of samples per bootstrap trial
        replica_energies = replica_energies_all[:,:,ref_shift::sample_spacing]
        array_folded_states_boot = array_folded_states[ref_shift::sample_spacing,:]

        # Get all possible sample indices
        sample_indices_all = np.arange(0,len(replica_energies[0,0,:]))
        # n_samples should match the size of the sliced replica energy dataset
        sample_indices = resample(sample_indices_all, replace=True, n_samples=len(sample_indices_all))
        
        n_state = replica_energies.shape[0]
        
        # replica_energies is [n_states x n_states x n_frame]        
        replica_energies_resample = np.zeros_like(replica_energies)
        array_folded_states_resample = np.zeros_like(array_folded_states_boot)
        
        # Select the sampled frames from array_folded_states and replica_energies:
        j = 0
        for i in sample_indices:
            replica_energies_resample[:,:,j] = replica_energies[:,:,i]
            array_folded_states_resample[j,:] = array_folded_states_boot[i,:]
            j += 1
              
        # Run partial heat capacity expectation calculation:
        (Cv_partial_values_boot[i_boot], Cv_partial_uncertainty_boot_out, T_list,
        FWHM_curr, Tm_curr, Cv_height_curr, U_expect_confs_boot[i_boot], N_eff_partial_boot[i_boot]) = get_partial_heat_capacities(
            array_folded_states_resample,
            output_data=output_data,
            num_intermediate_states=num_intermediate_states,
            frac_dT=frac_dT,
            plot_file=None,
            bootstrap_energies=replica_energies_resample,
            )
            
        # Cv_partial_values_boot[i_boot] is a dict mapping {state:Cv_array}
        # U_expect_confs_boot[i_boot] is likewise a dict mapping {state:U_expect_array}
        # T_list is just a single 1d array
            
        if i_boot == 0:
            # Get units:
            Cv_unit = Cv_partial_values_boot[0][0][0].unit
            U_unit = U_expect_confs_boot[0][0][0].unit
            T_unit = T_list[0].unit
            
        # To assign to array elements, need to strip units:
        # FWHM_curr, Tm_curr, and Cv_height_curr are dicts mapping {state:value}
        
        for c in range(n_conf_states):
            FWHM_boot[c][i_boot] = FWHM_curr[c].value_in_unit(T_unit)
            Tm_boot[c][i_boot] = Tm_curr[c].value_in_unit(T_unit)
            Cv_height_boot[c][i_boot] = Cv_height_curr[c].value_in_unit(Cv_unit)
    
    # Convert bootstrap trial dicts to arrays 
    # The keys of Cv_partial_values_boot are [i_boot][state]
    # The keys of U_expect_confs_boot are [i_boot][state]
    
    arr_Cv_values_boot = {}
    arr_U_expect_confs_boot = {}
    arr_N_eff_boot = {}
    
    for c in range(n_conf_states):
        arr_Cv_values_boot[c] = np.zeros((n_trial_boot, len(T_list)))
        arr_U_expect_confs_boot[c] = np.zeros((n_trial_boot, len(T_list)))
        arr_N_eff_boot[c] = np.zeros((n_trial_boot, len(T_list)))
        
        for i_boot in range(n_trial_boot):
            arr_Cv_values_boot[c][i_boot,:] = Cv_partial_values_boot[i_boot][c].value_in_unit(Cv_unit)
            arr_U_expect_confs_boot[c][i_boot,:] = U_expect_confs_boot[i_boot][c].value_in_unit(U_unit)
            arr_N_eff_boot[c][i_boot,:] = N_eff_partial_boot[i_boot][c]
            
    # Compute mean values:
    Cv_values = {}
    U_expect_values = {}
    N_eff_values = {}
    Cv_height_value = {}
    Tm_value = {}
    FWHM_value = {}
    
    for c in range(n_conf_states):
        Cv_values[c] = np.mean(arr_Cv_values_boot[c],axis=0)*Cv_unit
        U_expect_values[c] = np.mean(arr_U_expect_confs_boot[c],axis=0)*U_unit
        N_eff_values[c] = np.mean(arr_N_eff_boot[c],axis=0)
        Cv_height_value[c] = np.mean(Cv_height_boot[c])*Cv_unit
        Tm_value[c] = np.mean(Tm_boot[c])*T_unit
        FWHM_value[c] = np.mean(FWHM_boot[c])*T_unit
    
    # Compute confidence intervals:
    Cv_uncertainty = {}
    U_expect_uncertainty = {}
    Cv_height_uncertainty = {}
    Tm_uncertainty = {}
    FWHM_uncertainty = {}    
    
    if conf_percent == 'sigma':
        # Use analytical standard deviation instead of percentile method:
        
        for c in range(n_conf_states):
            # Cv values:
            Cv_std = np.std(arr_Cv_values_boot[c],axis=0)
            Cv_uncertainty[c] = (-Cv_std*Cv_unit, Cv_std*Cv_unit)
        
            # U expectation values:
            U_std = np.std(arr_U_expect_confs_boot[c],axis=0)
            U_expect_uncertainty[c] = (-U_std*U_unit, U_std*U_unit)
        
            # Cv peak height:
            Cv_height_std = np.std(Cv_height_boot[c])
            Cv_height_uncertainty[c] = (-Cv_height_std*Cv_unit, Cv_height_std*Cv_unit)   
            
            # Melting point:
            Tm_std = np.std(Tm_boot[c])
            Tm_uncertainty[c] = (-Tm_std*T_unit, Tm_std*T_unit)
            
            # Full width half maximum:
            FWHM_std = np.std(FWHM_boot[c])
            FWHM_uncertainty[c] = (-FWHM_std*T_unit, FWHM_std*T_unit)
        
    else:
        # Compute specified confidence interval:
        p_lo = (100-conf_percent)/2
        p_hi = 100-p_lo
                
        for c in range(n_conf_states):        
            # Cv values:
            Cv_diff = arr_Cv_values_boot[c]-np.mean(arr_Cv_values_boot[c],axis=0)
            Cv_conf_lo = np.percentile(Cv_diff,p_lo,axis=0,interpolation='linear')
            Cv_conf_hi = np.percentile(Cv_diff,p_hi,axis=0,interpolation='linear')
          
            Cv_uncertainty[c] = (Cv_conf_lo*Cv_unit, Cv_conf_hi*Cv_unit)
                        
            # U expectation values:            
            U_diff = arr_U_expect_confs_boot[c]-np.mean(arr_U_expect_confs_boot[c],axis=0)
            U_conf_lo = np.percentile(U_diff,p_lo,axis=0,interpolation='linear')
            U_conf_hi = np.percentile(U_diff,p_hi,axis=0,interpolation='linear')
          
            U_expect_uncertainty[c] = (U_conf_lo*U_unit, U_conf_hi*U_unit)
            
            # Cv peak height:                
            Cv_height_diff = Cv_height_boot[c]-np.mean(Cv_height_boot[c])
            Cv_height_conf_lo = np.percentile(Cv_height_diff,p_lo,interpolation='linear')
            Cv_height_conf_hi = np.percentile(Cv_height_diff,p_hi,interpolation='linear')
            
            Cv_height_uncertainty[c] = (Cv_height_conf_lo*Cv_unit, Cv_height_conf_hi*Cv_unit)        
            
            # Melting point: 
            Tm_diff = Tm_boot[c]-np.mean(Tm_boot[c])
            Tm_conf_lo = np.percentile(Tm_diff,p_lo,interpolation='linear')
            Tm_conf_hi = np.percentile(Tm_diff,p_hi,interpolation='linear')
            
            Tm_uncertainty[c] = (Tm_conf_lo*T_unit, Tm_conf_hi*T_unit)
            
            # Full width half maximum:
            FWHM_diff = FWHM_boot[c]-np.mean(FWHM_boot[c])
            FWHM_conf_lo = np.percentile(FWHM_diff,p_lo,interpolation='linear')
            FWHM_conf_hi = np.percentile(FWHM_diff,p_hi,interpolation='linear')
            
            FWHM_uncertainty[c] = (FWHM_conf_lo*T_unit, FWHM_conf_hi*T_unit)
    
    # Plot and return the heat capacity (with units)
    if plot_file is not None:
        plot_partial_heat_capacities(Cv_values, Cv_uncertainty, T_list, file_name=plot_file)

    return (T_list, Cv_values, Cv_uncertainty, Tm_value, Tm_uncertainty, Cv_height_value, Cv_height_uncertainty,
        FWHM_value, FWHM_uncertainty, U_expect_values, U_expect_uncertainty, N_eff_values)
    

def bootstrap_heat_capacity(frame_begin=0, sample_spacing=1, frame_end=-1, plot_file='heat_capacity_boot.pdf',
    output_data="output/output.nc", num_intermediate_states=0,frac_dT=0.05,conf_percent='sigma',
    n_trial_boot=200, U_kln=None, sparsify_stride=1):
    """
    Calculate and plot the heat capacity curve, with uncertainty determined using bootstrapping.
    Uncorrelated datasets are selected using a random starting frame, repeated n_trial_boot 
    times. Uncertainty in melting point and full-width half maximum of the Cv curve are also returned.
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
    
    :param plot_file: path to filename to output plot (default='heat_capacity_boot.pdf')
    :type plot_file: str

    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data (default = "output/output.nc")                                                                                          
    :type output_data: str    
    
    :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list' (default=0)
    :type num_intermediate_states: int

    :param frac_dT: The fraction difference between temperatures points used to calculate finite difference derivatives (default=0.05)
    :type frac_dT: float    
    
    :param conf_percent: Confidence level in percent for outputting uncertainties (default = 68.27 = 1 sigma)
    :type conf_percent: float
    
    :param n_trial_boot: number of trials to run for generating bootstrapping uncertainties (default=200)
    :type n_trial_boot: int
    
    :param U_kln: re-evaluated state energies array to be used for the MBAR calculation (starts at frame_begin)
    :type U_kln: 3d numpy array (float) with dimensions [replica, evaluated_state, frame]
    
    :param sparsify_stride: apply this stride to the replica energies from file, to match sparsified reevaluated energies (default=1)
    :type sparsify_stride: int
    
    :returns:
       - T_list ( List( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'Cv'
       - Cv_values ( List( float * kJ/mol/K ) ) - The heat capacity values for all (including inserted intermediates) states
       - Cv_uncertainty ( Tuple ( np.array(float) * kJ/mol/K ) ) - confidence interval for all Cv_values computed from bootstrapping
       - Tm_value ( float * unit.simtk.temperature ) - Melting point mean value computed from bootstrapping
       - Tm_uncertainty ( Tuple ( float * unit.simtk.temperature ) ) - confidence interval for melting point computed from bootstrapping
       - Cv_height_value ( float * kJ/mol/K ) - Height of the Cv peak relative to the lowest value over the temperature range used.
       - Cv_height_uncertainty ( Tuple ( np.array(float) * kJ/mol/K ) ) - confidence interval for all Cv_height_value computed from bootstrapping
       - FWHM_value ( float * unit.simtk.temperature ) - Cv full width half maximum mean value computed from bootstrapping
       - FWHM_uncertainty ( Tuple ( float * unit.simtk.temperature ) ) - confidence interval for Cv full width half maximum computed from bootstrapping
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
    
    # Close the data file - repeatedly opening the same .nc can cause seg faults:
    reporter.close()    
    
    # If we sparsified replica energies when reevaluating energies, need to apply it here:
    if frame_end > 0:
        replica_energies_all = replica_energies_all[:,:,frame_begin:frame_end:sparsify_stride]
    else:
        replica_energies_all = replica_energies_all[:,:,frame_begin::sparsify_stride]
    
    # Store data for each sampling trial:
    Cv_values_boot = {}
    Cv_uncertainty_boot = {}
    N_eff_boot = {}
    
    Tm_boot = np.zeros(n_trial_boot)
    Cv_height_boot = np.zeros(n_trial_boot)
    FWHM_boot = np.zeros(n_trial_boot)

    # Save the full re-evaluated energy array
    U_kln_all = U_kln
    
    for i_boot in range(n_trial_boot):
    
        # Select production frames to analyze
        # Here we can potentially change the reference frame for each bootstrap trial.
        ref_shift = np.random.randint(int(sample_spacing/sparsify_stride))
        # Depending on the reference frame, there may be small differences in numbers of samples per bootstrap trial
        replica_energies = replica_energies_all[:,:,ref_shift::int(sample_spacing/sparsify_stride)]

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
                # U_kln may be a sparsified energy array 
                U_kln = U_kln_all[:,:,ref_shift:frame_end:int(sample_spacing/sparsify_stride)]
            else:
                U_kln = U_kln_all[:,:,ref_shift::int(sample_spacing/sparsify_stride)]
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
            (Cv_sim, dCv_sim, Cv_values_boot[i_boot], Cv_uncertainty_boot[i_boot],
            T_list, FWHM_curr, Tm_curr, Cv_height_curr,
            N_eff_boot[i_boot]) = get_heat_capacity_reeval(
                U_kln_resample,
                output_data=output_data,
                frame_begin=frame_begin,
                frame_end=frame_end,
                sample_spacing=int(sample_spacing/sparsify_stride),
                num_intermediate_states=num_intermediate_states,
                frac_dT=frac_dT,
                plot_file_sim=None,
                plot_file_reeval=None,
                bootstrap_energies=replica_energies_resample,
                )
            
        else:    
            # Run standard heat capacity expectation calculation:
            (Cv_values_boot[i_boot], Cv_uncertainty_boot[i_boot], T_list,
            FWHM_curr, Tm_curr, Cv_height_curr,
            N_eff_boot[i_boot]) = get_heat_capacity(
                output_data=output_data,
                num_intermediate_states=num_intermediate_states,
                frac_dT=frac_dT,
                plot_file=None,
                bootstrap_energies=replica_energies_resample,
                )
            
        if i_boot == 0:
            # Get units:
            Cv_unit = Cv_values_boot[0][0].unit
            T_unit = T_list[0].unit
            
        # To assign to array elements, need to strip units:    
        FWHM_boot[i_boot] = FWHM_curr.value_in_unit(T_unit)
        Tm_boot[i_boot] = Tm_curr.value_in_unit(T_unit)
        Cv_height_boot[i_boot] = Cv_height_curr.value_in_unit(Cv_unit)
    
    # Convert dicts to array
    arr_Cv_values_boot = np.zeros((n_trial_boot, len(T_list)))
    arr_N_eff_boot = np.zeros((n_trial_boot, len(N_eff_boot[0])))
    
    for i_boot in range(n_trial_boot):
        arr_Cv_values_boot[i_boot,:] = Cv_values_boot[i_boot].value_in_unit(Cv_unit)
        arr_N_eff_boot[i_boot,:] = N_eff_boot[i_boot]
            
    # Compute mean values:        
    Cv_values = np.mean(arr_Cv_values_boot,axis=0)*Cv_unit      
    Cv_height_value = np.mean(Cv_height_boot)*Cv_unit
    Tm_value = np.mean(Tm_boot)*T_unit
    FWHM_value = np.mean(FWHM_boot)*T_unit
    N_eff_values = np.mean(arr_N_eff_boot)
    
    # Compute confidence intervals:
    if conf_percent == 'sigma':
        # Use analytical standard deviation instead of percentile method:
        
        # Cv values:
        Cv_std = np.std(arr_Cv_values_boot,axis=0)
        Cv_uncertainty = (-Cv_std*Cv_unit, Cv_std*Cv_unit)
        
        # Cv peak height:
        Cv_height_std = np.std(Cv_height_boot)
        Cv_height_uncertainty = (-Cv_height_std*Cv_unit, Cv_height_std*Cv_unit)   
        
        # Melting point:
        Tm_std = np.std(Tm_boot)
        Tm_uncertainty = (-Tm_std*T_unit, Tm_std*T_unit)
        
        # Full width half maximum:
        FWHM_std = np.std(FWHM_boot)
        FWHM_uncertainty = (-FWHM_std*T_unit, FWHM_std*T_unit)
        
    else:
        # Compute specified confidence interval:
        p_lo = (100-conf_percent)/2
        p_hi = 100-p_lo
                
        # Cv values:
        Cv_diff = arr_Cv_values_boot-np.mean(arr_Cv_values_boot,axis=0)
        Cv_conf_lo = np.percentile(Cv_diff,p_lo,axis=0,interpolation='linear')
        Cv_conf_hi = np.percentile(Cv_diff,p_hi,axis=0,interpolation='linear')
      
        Cv_uncertainty = (Cv_conf_lo*Cv_unit, Cv_conf_hi*Cv_unit) 
                    
        # Cv peak height:                
        Cv_height_diff = Cv_height_boot-np.mean(Cv_height_boot)
        Cv_height_conf_lo = np.percentile(Cv_height_diff,p_lo,interpolation='linear')
        Cv_height_conf_hi = np.percentile(Cv_height_diff,p_hi,interpolation='linear')
        
        Cv_height_uncertainty = (Cv_height_conf_lo*Cv_unit, Cv_height_conf_hi*Cv_unit)                  
        
        # Melting point: 
        Tm_diff = Tm_boot-np.mean(Tm_boot)
        Tm_conf_lo = np.percentile(Tm_diff,p_lo,interpolation='linear')
        Tm_conf_hi = np.percentile(Tm_diff,p_hi,interpolation='linear')
        
        Tm_uncertainty = (Tm_conf_lo*T_unit, Tm_conf_hi*T_unit)  
        
        # Full width half maximum:
        FWHM_diff = FWHM_boot-np.mean(FWHM_boot)
        FWHM_conf_lo = np.percentile(FWHM_diff,p_lo,interpolation='linear')
        FWHM_conf_hi = np.percentile(FWHM_diff,p_hi,interpolation='linear')
        
        FWHM_uncertainty = (FWHM_conf_lo*T_unit, FWHM_conf_hi*T_unit) 
    
    # Plot and return the heat capacity (with units)
    if plot_file is not None:
        plot_heat_capacity(Cv_values, Cv_uncertainty, T_list, file_name=plot_file)
                    
    return T_list, Cv_values, Cv_uncertainty, Tm_value, Tm_uncertainty, Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty, N_eff_values
        
    
def get_cv_FWHM(Cv_values, T_list):
    """
    Internal function for getting the full-width half-maximum, melting point, and peak height from a heat capacity vs T dataset.
    
    :param Cv_values: heat capacity data series (unitless)
    :type Cv_values: float * 1D np.array
    
    :param T_list: temperature data series (unitless)
    :type T_list: float * 1D np.array
    
    :returns:
       - FWHM ( float ) - Full width half maximum from heat capacity vs T
       - Tm ( float ) - Melting point from heat capacity vs T
       - Cv_height ( float ) - Relative height of heat capacity peak
    """
    
    # Compute the melting point:
    max_index = np.argmax(Cv_values)
    Tm = T_list[max_index]
    
    # Compute the peak height, relative to lowest Cv value in the temp range:
    Cv_height = (np.max(Cv_values)-np.min(Cv_values))
    
    # Compute the FWHM:
    # Cv value at half-maximum:
    mid_val = np.min(Cv_values) + Cv_height/2
    
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
            curr_val = Cv_values[index]
            prev_val = Cv_values[index+1]
            
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
            curr_val = Cv_values[index]
            prev_val = Cv_values[index-1]
        if curr_val <= mid_val:
            # The upper midpoint lies within T[index] and T[index-1]
            # Interpolate solution:
            T_half_hi = T_list[index]+(mid_val-curr_val)*(T_list[index-1]-T_list[index])/(prev_val-curr_val)
            half_hi_found = True
        else:
            m += 1
    
    if half_lo_found and half_hi_found:
        FWHM = (T_half_hi-T_half_lo)
    elif half_lo_found == True and half_hi_found == False:
        FWHM = 2*(Tm-T_half_lo)
    elif half_lo_found == False and half_hi_found == True:
        FWHM = 2*(T_half_hi-Tm)
        
    return FWHM, Tm, Cv_height