from math import exp, log

# OpenMM utilities
import mdtraj as md
import numpy as np
import pymbar
from matplotlib import pyplot as plt
from openmm import unit
from pymbar import timeseries
from scipy import interpolate
from scipy.optimize import Bounds, LinearConstraint, minimize, minimize_scalar

kB = (unit.MOLAR_GAS_CONSTANT_R).in_units_of(unit.kilojoule / (unit.kelvin * unit.mole))

def bin_samples(sample_kn, n_bins):
    """
    """
    max_value, min_value = None, None
    for index_1 in range(len(sample_kn)):
        for index_2 in range(len(sample_kn[index_1])):
            sample = sample_kn[index_1][index_2]
            if max_value == None:
                max_value = sample
            if min_value == None:
                min_value = sample
            if sample > max_value:
                max_value = sample
            if sample < min_value:
                min_value = sample

    bin_size = (max_value - min_value) / (n_bins + 1)
    bins = np.array(
        [[min_value + i * bin_size, min_value + (i + 1) * bin_size] for i in range(n_bins)]
    )
    bin_counts = np.zeros((len(sample_kn), len(bins)))

    for index_1 in range(len(sample_kn)):
        for index_2 in range(len(sample_kn[index])):
            sample = sample_kn[index_1][index_2]

    return (bins, bin_counts)


def get_decorrelated_samples(replica_positions, replica_energies, temperature_list):
    """
    Given a set of replica exchange trajectories, energies, and associated temperatures, this function returns decorrelated samples, as obtained from pymbar with timeseries.subsample_correlated_data.

    :param replica_positions: Positions array for the replica exchange data for which we will write PDB files
    :type replica_positions: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [n_replicas,cgmodel.num_beads,3] ), simtk.unit )

    :param replica_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps 
    :type replica_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

    :param temperature_list: List of temperatures for the simulation data.
    :type temperature_list: List( float * simtk.unit.temperature )

    :returns:
       - configurations ( List( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ (n_decorrelated_samples,cgmodel.num_beads,3), simtk.unit ) ) - A list of decorrelated samples
       - energies ( List( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) - The energies for the decorrelated samples (configurations)

        """
    all_poses = []
    all_energies = []

    for replica_index in range(len(replica_positions)):
        energies = replica_energies[replica_index][replica_index]
        [t0, g, Neff_max] = timeseries.detectEquilibration(energies)
        energies_equil = energies[t0:]
        poses_equil = replica_positions[replica_index][t0:]
        indices = timeseries.subsample_correlated_data(energies_equil)
        for index in indices:
            all_energies.append(energies_equil[index])
            all_poses.append(poses_equil[index])

    all_energies = np.array([float(energy) for energy in all_energies])

    return (all_poses, all_energies)


def get_entropy_differences(mbar):
    """
    Given an `MBAR <https://pymbar.readthedocs.io/en/latest/mbar.html>`_ class object, this function computes the entropy differences for the states defined within.

    :param mbar: An MBAR() class object (from the 'pymbar' package)
    :type mbar: `MBAR <https://pymbar.readthedocs.io/en/latest/mbar.html>`_ class object

    :returns:
      - Delta_s ( np.array( n_mbar_states x n_thermo_states ) - Entropy differences for the thermodynamic states in 'mbar'
      - dDelta_s ( np.array( n_mbar_states x n_thermo_states ) - Uncertainty in the entropy differences for the thermodynamic states in 'mbar'

    """
    results = mbar.compute_entropy_and_enthalpy()
    Delta_s = results["Delta_s"]
    dDelta_s = results["dDelta_s"]
    
    return (Delta_s, dDelta_s)


def get_enthalpy_differences(mbar):
    """
    Given an `MBAR <https://pymbar.readthedocs.io/en/latest/mbar.html>`_ class object, this function computes the enthalpy differences for the states defined within.

    :param mbar: An MBAR() class object (from the 'pymbar' package)
    :type mbar: `MBAR <https://pymbar.readthedocs.io/en/latest/mbar.html>`_ class object

    :returns:
      - Delta_u ( np.array( n_mbar_states x n_thermo_states ) - Enthalpy differences for the thermodynamic states in 'mbar'
      - dDelta_u ( np.array( n_mbar_states x n_thermo_states ) - Uncertainty in the enthalpy differences for the thermodynamic states in 'mbar'

    """
    results = mbar.compute_entropy_and_enthalpy()
    Delta_u = results["Delta_u"]
    dDelta_u = results["dDelta_u"]
    
    return (Delta_u, dDelta_u)


def get_free_energy_differences(mbar):
    """
    Given an `MBAR <https://pymbar.readthedocs.io/en/latest/mbar.html>`_ class object, this function computes the free energy differences for the states defined within.

    :param mbar: An MBAR() class object (from the 'pymbar' package)
    :type mbar: `MBAR <https://pymbar.readthedocs.io/en/latest/mbar.html>`_ class object

    :returns:
      - df_ij ( np.array( n_mbar_states x n_thermo_states ) - Free energy differences for the thermodynamic states in 'mbar'
      - ddf_ij ( np.array( n_mbar_states x n_thermo_states ) - Uncertainty in the free energy differences for the thermodynamic states in 'mbar'

    """
    results = mbar.compute_entropy_and_enthalpy()
    df_ij = results["Delta_f"]
    ddf_ij = results["dDelta_f"]
    
    return (df_ij, ddf_ij)


def get_temperature_list(min_temp, max_temp, num_replicas):
    """
    Given the parameters to define a temperature range as input, this function uses logarithmic spacing to generate a list of intermediate temperatures.

    :param min_temp: The minimum temperature in the temperature list.
    :type min_temp: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

    :param max_temp: The maximum temperature in the temperature list.
    :type max_temp: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

    :param num_replicas: The number of temperatures in the list.
    :type num_replicas: int

    :returns:
       - temperature_list ( 1D numpy array ( float * simtk.unit.temperature ) ) - List of temperatures

    """
    
    T_unit = min_temp.unit
    
    temperature_list = np.logspace(
        np.log10(min_temp.value_in_unit(T_unit)),
        np.log10(max_temp.value_in_unit(T_unit)),
        num=num_replicas
        )
        
    # Reassign units:
    temperature_list *= T_unit
    
    return temperature_list
    
    
def get_opt_temperature_list(temperature_list_init, C_v, number_intermediate_states=0, plotfile=None, verbose=True):
    """
    Given an initial temperature list, and heat capacity curve that resulted from a replica exchange simulation
    using those temperatures, computes a revised temperature list satisfying the constant entropy increase (CEI) method
    
    :param temperature_list_init: List of temperatures for initial replica exchange run
    :type temperature_list_init: 1D numpy array ( float * simtk.unit.temperature )
    
    :param C_v: List of heat capacities evaluated at each temperature in temperature_list_init 
    :type C_v: 1D numpy array [ Quantity ]
    
    :param number_intermediate_states: number of unsampled states between each pair of sampled states (default=0)
    :type number_intermediate_states: int
    
    :param plotfile: path to filename for plotting spline fit to C_v/T vs. T (default=None)
    :type plotfile: str
    
    :param verbose: option to print final output of scipy optimization routines
    :type verbose: bool
    
    :returns:
       - T_opt_list ( 1D numpy array ( float * simtk.unit.temperature ) ) - New optimally spaced temperature list
       - deltaS_list ( 1D numpy array ( float * simtk.unit ) ) - Actual entropy increases for adjacent temperatures in T_opt_list
    """
        
    # Process initial temperature list
    # Check for intermediate states from pymbar
    # (more intermediate states will give better estimate of new temperatures)
    

    T_init_sampled = temperature_list_init[::(number_intermediate_states+1)]
    T_init_sampled_val = np.zeros((len(T_init_sampled)))
    Tunit = T_init_sampled[0].unit
    for i in range(len(T_init_sampled)):
        T_init_sampled_val[i] = T_init_sampled[i].value_in_unit(Tunit)
        
    T_init_sampled = T_init_sampled_val
    
    # First and last temps are fixed bounds:
    T0 = T_init_sampled[0]
    TN = T_init_sampled[-1]
            
    # Fit C_v/T vs. T data to spline
    xdata = temperature_list_init
    ydata = C_v / temperature_list_init
    Cv_unit = C_v[0].unit
    
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
    
    if plotfile is not None:
        # Plot the spline fit:
        figure = plt.figure()
        line1 = plt.plot(xdata,ydata,'ok',fillstyle='none',label='simulation_data')
        
        xspline = np.linspace(xdata[0],xdata[-1],1000)
        yspline = interpolate.splev(xspline, spline_tck, der=0)
        
        line2 = plt.plot(xspline,yspline,'-b',label='spline fit')
        
        plt.xlabel('Temperature (K)')
        plt.ylabel('C_v / T (kJ/mol/K^2)')
        plt.legend()
        
        plt.savefig(f'{plotfile}')
        plt.close()
    
    # Fix the first and last temps, with intermediate temps varied in a minimization,
    # With constraint that the temperature spacings must sum to the original temp range.
    # The objective function is the standard deviation of the entropy differences between all adjacent temps.
    
    def entropy_stdev(T_deltas):
        # Compute standard deviation of entropy differences
        T_opt_list = np.zeros(len(T_deltas)+1)
        deltaS_list = np.zeros(len(T_deltas))
 
        T_opt_list[0] = T0
        for i in range(len(T_deltas)-1):
            T_opt_list[i+1] = T_opt_list[i]+T_deltas[i]  
        T_opt_list[-1] = TN
        
        for i in range(len(T_deltas)):
            deltaS_list[i] = interpolate.splint(T_opt_list[i],T_opt_list[i+1], spline_tck)
        return np.std(deltaS_list)
    
        
    # For initial guess, use the original spacing:
    T_delta0 = T_init_sampled[1::]-T_init_sampled[0:-1]
    
    # Set up linear equality constraint:
    constraint = LinearConstraint(
        np.ones(len(T_init_sampled)-1),
        lb=(T_init_sampled[-1]-T_init_sampled[0]),
        ub=(T_init_sampled[-1]-T_init_sampled[0]),
        )
        
    bounds = Bounds(np.ones(len(T_delta0))*1E-3,np.ones(len(T_delta0))*500)
    
    opt_results = minimize(
        entropy_stdev,
        T_delta0,
        bounds=bounds,
        constraints=constraint,
        method='SLSQP',
        options={
            'ftol':1E-12,
            'maxiter':1E6}
        )         
        
    if not opt_results.success:
        print('Error: CEI optimization did not converge')
        print(f'Constant entropy increase optimization results:\n{opt_results}') 
        exit()
    
    if verbose:
        print(f'Constant entropy increase optimization results:\n{opt_results}')      
    
    # Retreive final temperature list and entropy diff list:
    T_deltas_opt = opt_results.x
    
    T_opt_list = np.zeros(len(T_deltas_opt)+1)
    deltaS_list = np.zeros(len(T_deltas_opt))

    T_opt_list[0] = T0
    for i in range(len(T_deltas_opt)):
        T_opt_list[i+1] = T_opt_list[i]+T_deltas_opt[i]  
    
    for i in range(len(T_deltas_opt)):
        deltaS_list[i] = interpolate.splint(T_opt_list[i],T_opt_list[i+1], spline_tck)
    
    return T_opt_list*unit.kelvin, deltaS_list*Cv_unit
    
        
def get_intermediate_temperatures(T_from_file, NumIntermediates):
    """
        Given a list of temperatures and a number of intermediate states as input, this function calculates the values for temperatures intermediate between those in this list, as the mean between values in the list.

        :param T_from_file: List of temperatures
        :type T_from_file: List( float * simtk.unit.temperature )

        :param NumIntermediates: The number of states to insert between existing states in 'T_from_file'
        :type NumIntermediates: int

        :returns:
           - Temp_k ( List( float * simtk.unit.temperature ) ) - A new list of temperatures that includes the inserted intermediates.

        """

    deltas = []
    for i in range(1, len(T_from_file)):
        deltas.append((T_from_file[i]._value - T_from_file[i - 1]._value) / (NumIntermediates + 1))
        deltas.append((T_from_file[i]._value - T_from_file[i - 1]._value) / (NumIntermediates + 1))
    originalK = len(T_from_file)

    Temp_k = []
    val_k = []
    current_T = min([T_from_file[i]._value for i in range(len(T_from_file))])

    for delta in deltas:
        current_T = current_T + delta
        Temp_k.append(current_T)

    if len(Temp_k) != (
        len(T_from_file) + NumIntermediates * (len(T_from_file) - NumIntermediates - 1)
    ):
        print("Error: new temperatures are not being assigned correctly.")
        print(
            "There were " + str(len(T_from_file)) + " temperatures before inserting intermediates,"
        )
        print(str(NumIntermediates) + " intermediate strucutures were requested,")
        print(
            "and there were " + str(len(Temp_k)) + " temperatures after inserting intermediates."
        )
        exit()

    Temp_k = np.array([temp for temp in Temp_k])
    return Temp_k


def get_mbar_expectation(E_kln, temperature_list, NumIntermediates, output=None, mbar=None):
    """
    Given a properly-formatted matrix of energies with associated temperatures this function reweights with MBAR (if 'mbar'=None), and can also compute the expectation value for any property of interest.

    .. warning:: This function accepts an input matrix thtat has either 'E_kln' or 'E_kn' format, but always provides an 'E_kn'-formatted matrix in return.

    :param E_kln: A matrix of energies or samples for a property that we would like to use to make predictions with MBAR.
    :type E_kln: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas ) OR List( List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas ) for num_replicas )

    :param temperature_list: List of temperatures for the simulation data.
    :type temperature_list: List( float * simtk.unit.temperature )

    :param NumIntermediates: The number of states to insert between existing states in 'T_from_file'
    :type NumIntermediates: int

    :param output: The 'output' option to use when calling MBAR, default = 'differences'
    :type output: str

    :param mbar: An MBAR() class object (from the 'pymbar' package), default = None
    :type mbar: `MBAR <https://pymbar.readthedocs.io/en/latest/mbar.html>`_ class object

    :returns:
      - mbar ( `MBAR <https://pymbar.readthedocs.io/en/latest/mbar.html>`_ ) - An MBAR() class object (from the 'pymbar' package)          
     - E_kn ( List( List( float * simtk.unit.energy for num_samples ) for num_replicas ) ) - A matrix of energies or samples for a property that we would like to use to make predictions with MBAR.
     - result ( List( List( float for num_samples ) for num_replicas ) - The MBAR expectation value for the energies and/or other samples that were provided.
     - dresult ( List( List( float for num_samples ) for num_replicas ) - The MBAR expectation value for the energies and/or other samples that were provided.
     - Temp_k ( List( float * simtk.unit.temperature ) ) - A new list of temperatures that includes the inserted intermediates.

    """

    if mbar == None:
        NumTemps = len(temperature_list)  # Last TEMP # + 1 (start counting at 1)

        T_from_file = np.array([temperature._value for temperature in temperature_list])
        E_from_file = E_kln
        originalK = len(T_from_file)
        N_k = np.zeros(originalK, np.int32)

        g = np.zeros(originalK, np.float64)
        for k in range(originalK):  # subsample the energies
            g[k] = pymbar.timeseries.statistical_inefficiency(E_from_file[k][k])
            indices = np.array(
                pymbar.timeseries.subsample_correlated_data(E_from_file[k][k], g=g[k])
            )  # indices of uncorrelated samples
            N_k[k] = len(indices)
            E_from_file[k, k, 0 : N_k[k]] = E_from_file[k, k, indices]

        if NumIntermediates > 0:
            Temp_k = get_intermediate_temperatures(temperature_list, NumIntermediates)
        else:
            Temp_k = np.array([temperature._value for temperature in temperature_list])

        # Update number of states
        K = len(Temp_k)
        # Loop, inserting E's into blank matrix (leaving blanks only where new Ts are inserted)

        Nall_k = np.zeros(
            [K], np.int32
        )  # Number of samples (n) for each state (k) = number of iterations/energies

        try:
            E_kn = np.zeros([K, len(E_from_file[0][0])], np.float64)
            for k in range(originalK - 1):
                E_kn[k + k * NumIntermediates, 0 : N_k[k]] = E_from_file[k, k, 0 : N_k[k]]
                Nall_k[k + k * NumIntermediates] = N_k[k]

            E_kn[-1][0 : N_k[-1]] = E_from_file[-1][-1][0 : N_k[-1]]
            Nall_k[-1] = N_k[-1]

        except:
            E_kn = np.zeros([K, len(E_from_file[0])], np.float64)
            for k in range(originalK):
                E_kn[k + k * NumIntermediates, 0 : N_k[k]] = E_from_file[k, 0 : N_k[k]]
                Nall_k[k + k * NumIntermediates] = N_k[k]

        beta_k = 1 / (kB._value * Temp_k)

        allE_expect = np.zeros([K], np.float64)
        allE2_expect = np.zeros([K], np.float64)
        dE_expect = np.zeros([K], np.float64)
        u_kn = np.zeros(
            [K, sum(Nall_k)], np.float64
        )  # u_kln is reduced pot. ener. of segment n of temp k evaluated at temp l
        # index = 0
        for k in range(K):
            index = 0
            for l in range(K):
                u_kn[k, index : index + Nall_k[l]] = beta_k[k] * E_kn[l, 0 : Nall_k[l]]
                index = index + Nall_k[l]

        # ------------------------------------------------------------------------
        # Initialize MBAR
        # ------------------------------------------------------------------------

        print("Initializing MBAR:")
        print("--K = number of Temperatures with data = %d" % (originalK))
        print("--L = number of total Temperatures = %d" % (K))
        print("--N = number of Energies per Temperature = %d" % (np.max(Nall_k)))

        mbar = pymbar.MBAR(u_kn, Nall_k, verbose=False, relative_tolerance=1e-12, initial_f_k=None)

        E_kn = u_kn  # not a copy, we are going to write over it, but we don't need it any more.
        for k in range(K):
            E_kn[k, :] *= beta_k[k] ** (
                -1
            )  # get the 'unreduced' potential -- we can't take differences of reduced potentials because the beta is different; math is much more confusing with derivatives of the reduced potentials.

    else:

        E_kn = E_kln
        Temp_k = temperature_list

    if output != None:
        results = mbar.computeExpectations(E_kn, output="differences", state_dependent=True)
        result = results["mu"]
        dresult = results["sigma"]
    else:
        results = mbar.computeExpectations(E_kn, state_dependent=True)
        result = results["mu"]
        dresult = results["sigma"]

    return (mbar, E_kn, result, dresult, Temp_k)
