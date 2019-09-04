#!/usr/bin/python

import numpy as np
from math import exp, log
# OpenMM utilities
import mdtraj as md
from simtk import unit
import pymbar

def get_decorrelated_samples(replica_positions,replica_energies,temperature_list):
        """
        """
        configurations = []
        energies = []
        K = len(temperature_list)
        g = np.zeros(K,np.float64)
        for k in range(K):  # subsample the energies
          E_total_all = np.array(np.delete(E_total_all_temp,0,0),dtype=float) # E_total_all stores total energies from NaCl simulation output, after re-typing
          [t0, g[k], Neff_max] = timeseries.detectEquilibration(replica_energies[k][k],nskip=10)
          indices = np.array(pymbar.timeseries.subsampleCorrelatedData(replica_energies[k][k],g=g[k])) # indices of uncorre
          configurations.append(replica_positions[k][k][g[k]])
          energies.append(replica_energies[k][k][g[k]])
          temperatures.append(temperature)
        return(configurations,energies,temperatures)

def get_entropy_differences(mbar):
        """
        """
        results = mbar.computeEntropyAndEnthalpy()
        results = {'Delta_f': results[0], 'dDelta_f': results[1], 'Delta_u': results[2], 'dDelta_u': results[3], 'Delta_s': results[4], 'dDelta_s': results[5]}
        Delta_s = results['Delta_s']
        dDelta_s = results['dDelta_s']
        return(Delta_s,dDelta_s)

def get_enthalpy_differences(mbar):
        """
        """
        results = mbar.computeEntropyAndEnthalpy()
        results = {'Delta_f': results[0], 'dDelta_f': results[1], 'Delta_u': results[2], 'dDelta_u': results[3], 'Delta_s': results[4], 'dDelta_s': results[5]}
        Delta_u = results['Delta_u']
        dDelta_u = results['dDelta_u']
        return(Delta_u,dDelta_u)

def get_free_energy_differences(mbar):
        """
        """

        results = mbar.getFreeEnergyDifferences()
        results = {'Delta_f': results[0], 'dDelta_f': results[1]}
        df_ij = results['Delta_f']
        ddf_ij = results['dDelta_f']
        return(df_ij,ddf_ij)

def calc_temperature_spacing(min_temp,max_temp,num_replicas,replica_index):
        """
        """
        T_replica_index = min_temp * exp( replica_index * log( max_temp._value / min_temp._value )/(num_replicas-1) )
        T_previous_index = min_temp * exp( (replica_index-1) * log( max_temp._value / min_temp._value )/(num_replicas-1) )
        delta = T_replica_index - T_previous_index 
        return(delta)

def get_temperature_list(min_temp,max_temp,num_replicas):
        """
        """
        temperature_list = []
        temperature_list.append(min_temp)
        replica_index = 1
        while len(temperature_list) > 0 and temperature_list[-1].__lt__(max_temp):
          delta = calc_temperature_spacing(min_temp,max_temp,num_replicas,replica_index)
          last_temperature = temperature_list[-1]
          temperature = last_temperature.__add__(delta)
          temperature_list.append(temperature)
          replica_index = replica_index + 1
        return(temperature_list)

def get_intermediate_temperatures(T_from_file,NumIntermediates):
        """
        """
        #------------------------------------------------------------------------
        # Insert Intermediate T's and corresponding blank U's and E's
        #------------------------------------------------------------------------
        kB = unit.Quantity(0.008314462,unit.kilojoule_per_mole)  #Boltzmann constant (Gas constant) in kJ/(mol*K)

        deltas = []
        for i in range(1,len(T_from_file)):
         deltas.append((T_from_file[i]._value-T_from_file[i-1]._value)/(NumIntermediates+1))
         deltas.append((T_from_file[i]._value-T_from_file[i-1]._value)/(NumIntermediates+1))
        originalK = len(T_from_file)

        Temp_k = []
        val_k = []
        current_T = min([T_from_file[i]._value for i in range(len(T_from_file))])

        for delta in deltas:
           current_T = current_T + delta
           Temp_k.append(current_T)

        if len(Temp_k) != (len(T_from_file) + NumIntermediates* (len(T_from_file)-NumIntermediates-1)):
          print("Error: new temperatures are not being assigned correctly.")
          print("There were "+str(len(T_from_file))+" temperatures before inserting intermediates,")
          print(str(NumIntermediates)+" intermediate strucutures were requested,")
          print("and there were "+str(len(Temp_k))+" temperatures after inserting intermediates.")
          exit()

        Temp_k = np.array([temp for temp in Temp_k])
        return(Temp_k)

def get_mbar_expectation(E_kln,temperature_list,NumIntermediates,output=None,mbar=None):
        """
        """

        if mbar == None:
         NumTemps = len(temperature_list) # Last TEMP # + 1 (start counting at 1)

         kB = unit.Quantity(0.008314462,unit.kilojoule_per_mole)  #Boltzmann constant (Gas constant) in kJ/(mol*K)
         T_from_file = np.array([temperature._value for temperature in temperature_list])
         E_from_file = E_kln
         originalK = len(T_from_file)
         N_k = np.zeros(originalK,np.int32)

         g = np.zeros(originalK,np.float64)
         for k in range(originalK):  # subsample the energies
          g[k] = pymbar.timeseries.statisticalInefficiency(E_from_file[k][k])
          indices = np.array(pymbar.timeseries.subsampleCorrelatedData(E_from_file[k][k],g=g[k])) # indices of uncorrelated samples
          N_k[k] = len(indices)
          E_from_file[k,k,0:N_k[k]] = E_from_file[k,k,indices]

         if NumIntermediates > 0:
           Temp_k = get_intermediate_temperatures(temperature_list,NumIntermediates)
         else:
           Temp_k = np.array([temperature._value for temperature in temperature_list])

         # Update number of states
         K = len(Temp_k)
         # Loop, inserting E's into blank matrix (leaving blanks only where new Ts are inserted)

         Nall_k = np.zeros([K], np.int32) # Number of samples (n) for each state (k) = number of iterations/energies

         try:
          E_kn = np.zeros([K,len(E_from_file[0][0])], np.float64)
          for k in range(originalK-1):
              E_kn[k+k*NumIntermediates,0:N_k[k]] = E_from_file[k,k,0:N_k[k]]
              Nall_k[k+k*NumIntermediates] = N_k[k]

          E_kn[-1][0:N_k[-1]] = E_from_file[-1][-1][0:N_k[-1]]
          Nall_k[-1] = N_k[-1]


         except:
          E_kn = np.zeros([K,len(E_from_file[0])], np.float64)
          for k in range(originalK):
              E_kn[k+k*NumIntermediates,0:N_k[k]] = E_from_file[k,0:N_k[k]]
              Nall_k[k+k*NumIntermediates] = N_k[k]

         beta_k = 1 / (kB._value * Temp_k)

         allE_expect = np.zeros([K], np.float64)
         allE2_expect = np.zeros([K], np.float64)
         dE_expect = np.zeros([K],np.float64)
         u_kn = np.zeros([K,sum(Nall_k)], np.float64) # u_kln is reduced pot. ener. of segment n of temp k evaluated at temp l
         #index = 0
         for k in range(K):
           index = 0
           for l in range(K):
              u_kn[k,index:index+Nall_k[l]] = beta_k[k] * E_kn[l,0:Nall_k[l]]
              index = index + Nall_k[l]

         #------------------------------------------------------------------------
         # Initialize MBAR
         #------------------------------------------------------------------------

         print("Initializing MBAR:")
         print("--K = number of Temperatures with data = %d" % (originalK))
         print("--L = number of total Temperatures = %d" % (K))
         print("--N = number of Energies per Temperature = %d" % (np.max(Nall_k)))

         mbar = pymbar.MBAR(u_kn, Nall_k, verbose=False, relative_tolerance=1e-12,initial_f_k = None)

         E_kn = u_kn  # not a copy, we are going to write over it, but we don't need it any more.
         for k in range(K):
               E_kn[k,:]*=beta_k[k]**(-1)  # get the 'unreduced' potential -- we can't take differences of reduced potentials because the beta is different; math is much more confusing with derivatives of the reduced potentials.

        else:

          E_kn = E_kln          
          Temp_k = temperature_list

        if output != None:
               results = mbar.computeExpectations(E_kn,output='differences', state_dependent = True)
               results = {'mu': results[0], 'sigma': results[1]}
               result = results['mu']
               dresult = results['sigma']
        else:
               results = mbar.computeExpectations(E_kn, state_dependent = True)
               results = {'mu': results[0], 'sigma': results[1]}
               result = results['mu']
               dresult = results['sigma']

        return(mbar,E_kn,result,dresult,Temp_k)
