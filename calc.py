import os
import numpy as np
from foldamers.parameters.reweight import *
import matplotlib.pyplot as pyplot
from openmmtools.multistate import MultiStateReporter
from openmmtools.multistate import ReplicaExchangeAnalyzer
import pymbar
from pymbar import timeseries

kB = (unit.MOLAR_GAS_CONSTANT_R).in_units_of(unit.kilojoule / (unit.kelvin * unit.mole)) # Boltzmann constant in kJ/(mol*K)


def plot_heat_capacity(C_v, dC_v, temperature_list, file_name="heat_capacity.png"):
    """
        Given an array of temperature-dependent heat capacity values and the uncertainties in their estimates, this function plots the heat capacity curve.
    
        :param C_v: The heat capacity data to plot.
        :type C_v: List( float )
    
        :param dC_v: The uncertainties in the heat capacity data
        :type dC_v: List( float )
    
        :param file_name: The name/path of the file where plotting output will be written, default = "heat_capacity.png"
        :type file_name: str
    
        """
    figure = pyplot.figure(1)
    temperature_list = np.array([temperature for temperature in temperature_list])
    C_v = np.array([C_v[i] for i in range(len(C_v))])
    dC_v = np.array([dC_v[i] for i in range(len(dC_v))])
    pyplot.errorbar(temperature_list, C_v, yerr=dC_v, figure=figure)
    pyplot.xlabel("Temperature ( Kelvin )")
    pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
    pyplot.title("Heat capacity for")
    pyplot.savefig(file_name)
    pyplot.close()
    return


def get_heat_capacity(temperature_list, output_data="output.nc", output_directory="output", num_intermediate_states=0,frac_dT=0.05):
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

    reporter = MultiStateReporter(os.path.join(output_directory,output_data), open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)
    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()

    # get the energies, which are already in reduced units
    # test whether we need this.
    Tunit = temperature_list[0].unit
    temps = np.array([temp.value_in_unit(Tunit)  for temp in temperature_list])  # should this just be array to begin with
    beta_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * temps)

    replica_energies = pymbar.utils.kln_to_kn(replica_energies)
    n_samples = len(replica_energies[0,:])
    
    # calculate the number of states we need expectations at.  We want it at all of the original
    # temperatures, each intermediate temperature, and then temperatures +/- from the original
    # to take finite derivatives.


    num_sampled_T = len(temps)
    n_unsampled_states = 3*(num_sampled_T + (num_sampled_T-1)*num_intermediate_states)
    unsampled_state_energies = np.zeros([n_unsampled_states,n_samples])
    full_T_list = np.zeros(n_unsampled_states)

    delta = np.zeros(num_sampled_T-1)
    full_T_list[0] = temps[0]
    t = 0
    for i in range(num_sampled_T-1):
        delta[i] = (temps[i+1] - temps[i])/(num_intermediate_states+1)
        for j in range(num_intermediate_states+1):
            full_T_list[t] = temps[i] + delta[i]*j
            t += 1
    full_T_list[t] = temps[-1]
    n_T_vals = t+1

    # add additional states for finite difference calculation.    
    full_T_list[n_T_vals] = full_T_list[0] - delta[0]*frac_dT
    full_T_list[2*n_T_vals] = full_T_list[0] + delta[0]*frac_dT
    for i in range(1,n_T_vals-1):
        ii = i//(num_intermediate_states+1)
        full_T_list[i + n_T_vals] = full_T_list[i] - delta[ii]*frac_dT
        full_T_list[i + 2*n_T_vals] = full_T_list[i] + delta[ii]*frac_dT
    full_T_list[2*n_T_vals-1] = full_T_list[n_T_vals-1] - delta[-1]*frac_dT
    full_T_list[3*n_T_vals-1] = full_T_list[n_T_vals-1] + delta[-1]*frac_dT        
    
    beta_full_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/Tunit) * full_T_list)

    ti = 0
    N_k = np.zeros(n_unsampled_states)
    for k in range(n_unsampled_states):
        # MBAR is with the 'unreduced' potential -- we can't take differences of reduced potentials
        # because the beta is different; math is much more confusing with derivatives of the reduced potentials.
        unsampled_state_energies[k, :] = replica_energies[0,:]*beta_full_k[k]/beta_k[0]
        if ti < len(temps):
            if full_T_list[k] == temps[ti]:
                ti += 1
                N_k[k] = n_samples//len(temps)  # these are the states that have samples

    # call MBAR to find weights.
    mbarT = pymbar.MBAR(unsampled_state_energies,N_k,verbose=False, relative_tolerance=1e-12);

    for k in range(n_unsampled_states):
        # get the 'unreduced' potential -- we can't take differences of reduced potentials
        # because the beta is different; math is much more confusing with derivatives of the reduced potentials.
        unsampled_state_energies[k, :] /= beta_full_k[k]

    results = mbarT.computeExpectations(unsampled_state_energies, state_dependent=True)
    E_expect = results[0]
    dE_expect = results[1]

    
    # expectations for the differences, which we need for numerical derivatives                                               
    results = mbarT.computeExpectations(unsampled_state_energies, output="differences", state_dependent=True)
    DeltaE_expect = results[0]
    dDeltaE_expect = results[1]

    Cv = np.zeros(n_T_vals)
    dCv = np.zeros(n_T_vals)
    for k in range(n_T_vals):
        im = k+n_T_vals
        ip = k+2*n_T_vals
        Cv[k] = (DeltaE_expect[im, ip]) / (full_T_list[ip] - full_T_list[im])
        dCv[k] = (dDeltaE_expect[im, ip]) / (full_T_list[ip] - full_T_list[im])
        
    import pdb
    pdb.set_trace()


    plot_heat_capacity(Cv, dCv, full_T_list[0:n_T_vals])
    return (Cv, dCv, full_T_list[0:n_T_vals])


def calculate_heat_capacity(
    E_expect,
    E2_expect,
    dE_expect,
    DeltaE_expect,
    dDeltaE_expect,
    df_ij,
    ddf_ij,
    Temp_k,
    originalK,
    numIntermediates,
    ntypes=3,
    dertype="temperature",
):
    """
    Given numerous pieces of thermodynamic data this function calculates the heat capacity by following the `'pymbar' example <https://github.com/choderalab/pymbar/tree/master/examples/heat-capacity>`_ .

    """

    # ------------------------------------------------------------------------
    # Compute Cv for NVT simulations as <E^2> - <E>^2 / (RT^2)
    # ------------------------------------------------------------------------

    # print("")
    # print("Computing Heat Capacity as ( <E^2> - <E>^2 ) / ( R*T^2 ) and as d<E>/dT")

    K = len(Temp_k)

    allCv_expect = np.zeros([K, ntypes], np.float64)
    dCv_expect = np.zeros([K, ntypes], np.float64)
    try:
        Temp_k = np.array([temp._value for temp in Temp_k])
    except:
        Temp_k = np.array([temp for temp in Temp_k])

    allCv_expect[:, 0] = (E2_expect - (E_expect * E_expect)) / (kB * Temp_k ** 2)

    ####################################
    # C_v by fluctuation formula
    ####################################

    N_eff = (
        E2_expect - (E_expect * E_expect)
    ) / dE_expect ** 2  # sigma^2 / (sigma^2/n) = effective number of samples
    dCv_expect[:, 0] = allCv_expect[:, 0] * np.sqrt(2 / N_eff)

    for i in range(originalK, K):

        # Now, calculae heat capacity by T-differences
        im = i - 1
        ip = i + 1
        # print(im,ip)
        if i == originalK:
            im = originalK
        if i == K - 1:
            ip = i
        ####################################
        # C_v by first derivative of energy
        ####################################

        if dertype == "temperature":  # temperature derivative
            # C_v = d<E>/dT
            allCv_expect[i, 1] = (DeltaE_expect[im, ip]) / (Temp_k[ip] - Temp_k[im])
            dCv_expect[i, 1] = (dDeltaE_expect[im, ip]) / (Temp_k[ip] - Temp_k[im])
        elif dertype == "beta":  # beta derivative
            # Cv = d<E>/dT = dbeta/dT d<E>/beta = - kB*T(-2) d<E>/dbeta  = - kB beta^2 d<E>/dbeta
            allCv_expect[i, 1] = (
                kB * beta_k[i] ** 2 * (DeltaE_expect[ip, im]) / (beta_k[ip] - beta_k[im])
            )
            dCv_expect[i, 1] = (
                -kB * beta_k[i] ** 2 * (dDeltaE_expect[ip, im]) / (beta_k[ip] - beta_k[im])
            )

        ####################################
        # C_v by second derivative of free energy
        ####################################

        if dertype == "temperature":
            # C_v = d<E>/dT = d/dT k_B T^2 df/dT = 2*T*df/dT + T^2*d^2f/dT^2

            if (i == originalK) or (i == K - 1):
                # We can't calculate this, set a number that will be printed as NAN
                allCv_expect[i, 2] = -10000000.0
            else:
                allCv_expect[i, 2] = (
                    kB
                    * Temp_k[i]
                    * (
                        2 * df_ij[ip, im] / (Temp_k[ip] - Temp_k[im])
                        + Temp_k[i]
                        * (df_ij[ip, i] - df_ij[i, im])
                        / ((Temp_k[ip] - Temp_k[im]) / (ip - im)) ** 2
                    )
                )

                A = (
                    2 * Temp_k[i] / (Temp_k[ip] - Temp_k[im])
                    + 4 * Temp_k[i] ** 2 / (Temp_k[ip] - Temp_k[im]) ** 2
                )
                B = (
                    2 * Temp_k[i] / (Temp_k[ip] - Temp_k[im])
                    + 4 * Temp_k[i] ** 2 / (Temp_k[ip] - Temp_k[im]) ** 2
                )
                # dCv_expect[i,2,n] = kB* [(A ddf_ij[ip,i])**2 + (B sdf_ij[i,im])**2 + 2*A*B*cov(df_ij[ip,i],df_ij[i,im])
                # This isn't it either: need to figure out that last term.
                dCv_expect[i, 2] = kB * ((A * ddf_ij[ip, i]) ** 2 + (B * ddf_ij[i, im]) ** 2)
                # Would need to add function computing covariance of DDG, (A-B)-(C-D)

        elif dertype == "beta":
            # if beta is evenly spaced, rather than t, we can do 2nd derivative in beta
            # C_v = d<E>/dT = d/dT (df/dbeta) = dbeta/dT d/dbeta (df/dbeta) = -k_b beta^2 df^2/d^2beta
            if (i == originalK) or (i == K - 1):
                # Flag as N/A -- we don't try to compute at the endpoints for now
                allCv_expect[i, 2] = -10000000.0
            else:
                allCv_expect[i, 2] = (
                    kB
                    * beta_k[i] ** 2
                    * (df_ij[ip, i] - df_ij[i, im])
                    / ((beta_k[ip] - beta_k[im]) / (ip - im)) ** 2
                )
            dCv_expect[i, 2] = (
                kB
                * (beta_k[i]) ** 2
                * (ddf_ij[ip, i] - ddf_ij[i, im])
                / ((beta_k[ip] - beta_k[im]) / (ip - im)) ** 2
            )
            # also wrong, need to be fixed.

    return (allCv_expect[:, 0], dCv_expect[:, 0])
