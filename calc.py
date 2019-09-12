import numpy as np
from foldamers.parameters.reweight import *
import matplotlib.pyplot as pyplot

kB = 0.008314462  #Boltzmann constant (Gas constant) in kJ/(mol*K)

def plot_heat_capacity(C_v,dC_v,temperature_list,file_name="heat_capacity.png"):
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
        pyplot.errorbar(temperature_list,C_v,yerr=dC_v,figure=figure)
        pyplot.xlabel("Temperature ( Kelvin )")
        pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
        pyplot.title("Heat capacity for")
        pyplot.savefig(file_name)
        pyplot.close()
        return

def get_heat_capacity(replica_energies,temperature_list,num_intermediate_states=None):
        """
        Given a set of trajectories, a temperature list, and a number of intermediate states to insert for the temperature list, this function calculates and plots the heat capacity profile.

        :param replica_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps 
        :type replica_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

        :param temperature_list: List of temperatures for which to perform replica exchange simulations, default = None
        :type temperature: List( float * simtk.unit.temperature )

        :param num_intermediate_states: The number of states to insert between existing states in 'temperature_list'
        :type num_intermediate_states: int

        :returns:
          - C_v ( List( float ) ) - The heat capacity values for all (including inserted intermediates) states
          - dC_v ( List( float ) ) - The uncertainty in the heat capacity values for intermediate states
          - new_temp_list ( List( float * unit.simtk.temperature ) ) - The temperature list corresponding to the heat capacity values in 'C_v'

        """
        if num_intermediate_states == None:
         num_intermediate_states = 1
        mbar,E_kn,E_expect,dE_expect,new_temp_list = get_mbar_expectation(replica_energies,temperature_list,num_intermediate_states)
        mbar,E_kn,DeltaE_expect,dDeltaE_expect,new_temp_list = get_mbar_expectation(E_kn,new_temp_list,num_intermediate_states,mbar=mbar,output='differences')
        mbar,E_kn,E2_expect,dE2_expect,new_temp_list = get_mbar_expectation(E_kn**2,new_temp_list,num_intermediate_states,mbar=mbar)
        df_ij,ddf_ij = get_free_energy_differences(mbar)
        C_v,dC_v = calculate_heat_capacity(E_expect,E2_expect,dE_expect,DeltaE_expect,dDeltaE_expect,df_ij,ddf_ij,new_temp_list,len(temperature_list),num_intermediate_states)
        plot_heat_capacity(C_v,dC_v,new_temp_list)
        return(C_v,dC_v,new_temp_list)

def calculate_heat_capacity(E_expect,E2_expect,dE_expect,DeltaE_expect,dDeltaE_expect,df_ij,ddf_ij,Temp_k,originalK,numIntermediates,ntypes=3,dertype="temperature"):
    """
    Given numerous pieces of thermodynamic data this function calculates the heat capacity by following the `'pymbar' example <https://github.com/choderalab/pymbar/tree/master/examples/heat-capacity>`_ .

    """

    #------------------------------------------------------------------------
    # Compute Cv for NVT simulations as <E^2> - <E>^2 / (RT^2)
    #------------------------------------------------------------------------

    #print("")
    #print("Computing Heat Capacity as ( <E^2> - <E>^2 ) / ( R*T^2 ) and as d<E>/dT")

    K = len(Temp_k)

    allCv_expect = np.zeros([K,ntypes], np.float64)
    dCv_expect = np.zeros([K,ntypes],np.float64)
    try:
      Temp_k = np.array([temp._value for temp in Temp_k])
    except:
      Temp_k = np.array([temp for temp in Temp_k])


    allCv_expect[:,0] = (E2_expect - (E_expect*E_expect)) / ( kB * Temp_k**2)

    ####################################
    # C_v by fluctuation formula
    ####################################

    N_eff = (E2_expect - (E_expect*E_expect))/dE_expect**2  # sigma^2 / (sigma^2/n) = effective number of samples
    dCv_expect[:,0] = allCv_expect[:,0]*np.sqrt(2/N_eff)

    for i in range(originalK,K):

        # Now, calculae heat capacity by T-differences
        im = i-1
        ip = i+1
        #print(im,ip)
        if (i==originalK):
            im = originalK
        if (i==K-1):
            ip = i
        ####################################
        # C_v by first derivative of energy
        ####################################

        if (dertype == 'temperature'):  # temperature derivative
            # C_v = d<E>/dT
            allCv_expect[i,1] = (DeltaE_expect[im,ip])/(Temp_k[ip]-Temp_k[im])
            dCv_expect[i,1] = (dDeltaE_expect[im,ip])/(Temp_k[ip]-Temp_k[im])
        elif (dertype == 'beta'):  # beta derivative
            #Cv = d<E>/dT = dbeta/dT d<E>/beta = - kB*T(-2) d<E>/dbeta  = - kB beta^2 d<E>/dbeta
            allCv_expect[i,1] = kB * beta_k[i]**2 * (DeltaE_expect[ip,im])/(beta_k[ip]-beta_k[im])
            dCv_expect[i,1] = -kB * beta_k[i]**2 *(dDeltaE_expect[ip,im])/(beta_k[ip]-beta_k[im])

        ####################################
        # C_v by second derivative of free energy
        ####################################

        if (dertype == 'temperature'):
            # C_v = d<E>/dT = d/dT k_B T^2 df/dT = 2*T*df/dT + T^2*d^2f/dT^2

            if (i==originalK) or (i==K-1):
                 # We can't calculate this, set a number that will be printed as NAN
                 allCv_expect[i,2] = -10000000.0
            else:
                allCv_expect[i,2] = kB*Temp_k[i]*(2*df_ij[ip,im]/(Temp_k[ip]-Temp_k[im]) +
                                                    Temp_k[i]*(df_ij[ip,i]-df_ij[i,im])/
                                                    ((Temp_k[ip]-Temp_k[im])/(ip-im))**2)

                A = 2*Temp_k[i]/(Temp_k[ip]-Temp_k[im]) + 4*Temp_k[i]**2/(Temp_k[ip]-Temp_k[im])**2
                B = 2*Temp_k[i]/(Temp_k[ip]-Temp_k[im]) + 4*Temp_k[i]**2/(Temp_k[ip]-Temp_k[im])**2
                #dCv_expect[i,2,n] = kB* [(A ddf_ij[ip,i])**2 + (B sdf_ij[i,im])**2 + 2*A*B*cov(df_ij[ip,i],df_ij[i,im])
                # This isn't it either: need to figure out that last term.
                dCv_expect[i,2] = kB*((A*ddf_ij[ip,i])**2 + (B*ddf_ij[i,im])**2)
                # Would need to add function computing covariance of DDG, (A-B)-(C-D)

        elif (dertype == 'beta'):
            # if beta is evenly spaced, rather than t, we can do 2nd derivative in beta
            # C_v = d<E>/dT = d/dT (df/dbeta) = dbeta/dT d/dbeta (df/dbeta) = -k_b beta^2 df^2/d^2beta
            if (i==originalK) or (i==K-1):
                #Flag as N/A -- we don't try to compute at the endpoints for now
                allCv_expect[i,2] = -10000000.0
            else:
                allCv_expect[i,2] = kB * beta_k[i]**2 *(df_ij[ip,i]-df_ij[i,im])/((beta_k[ip]-beta_k[im])/(ip-im))**2
            dCv_expect[i,2] = kB*(beta_k[i])**2 * (ddf_ij[ip,i]-ddf_ij[i,im])/((beta_k[ip]-beta_k[im])/(ip-im))**2
                # also wrong, need to be fixed.
   
    return(allCv_expect[:,0],dCv_expect[:,0])
