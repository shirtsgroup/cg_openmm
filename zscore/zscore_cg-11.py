import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import matplotlib.pyplot as pyplot
from statistics import mean, stdev
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.parameters.secondary_structure import *
from foldamers.thermo.calc import *
from foldamers.ensembles.ens_build import *
from cg_openmm.simulation.rep_exch import *

# Set value for kB
kB = unit.Quantity(0.008314462,unit.kilojoule_per_mole)

# Provide path to a PDB file containing the coordinates for a 'native' structure for our 1-1 12mer
native_structure_file = str(str(os.getcwd().split('examples')[0])+"ensembles/12_1_1_0/helix.pdb")

# Read the coordinates from the PDB file
native_structure = PDBFile(native_structure_file).getPositions()

# Path to existing replica exchange simulation data
number_replicas = 30
min_temp = 5.0 * unit.kelvin
max_temp = 100.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
print_frequency = 5
simulation_data=str(str(os.getcwd().split('examples')[0])+"/examples/homopolymer_heat_capacity_varying_simulation_time/output/output10.0.nc")

# Build a coarse grained model, which is needed in order to read in the positions data
cgmodel = CGModel()

# Read existing replica exchange simulation data
replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=simulation_data,print_frequency=print_frequency)

# Get uncorrelated samples from the replica exchange trajectories
configurations,energies = get_decorrelated_samples(replica_positions,replica_energies,temperature_list)

# Get "native" and "nonnative" ensembles from the decorrelated samples
native_ensemble,native_ensemble_energies,nonnative_ensemble,nonnative_ensemble_energies = get_ensembles_from_replica_positions(cgmodel,configurations,energies,temperature_list,native_fraction_cutoff=0.95,nonnative_fraction_cutoff=0.9,native_ensemble_size=10,nonnative_ensemble_size=100,decorrelate=False,native_contact_cutoff_distance=None)

e_unfolded_mean = mean(nonnative_ensemble_energies) # unfolded mean energy 
e_unfolded_sigma =  stdev(nonnative_ensemble_energies) # unfolded standard deviation energy
n_unfolded = len(e_unfolded_ensemble) # ratio of number of unfolded states to folded states. We keep number of folded states at 1
e_folded = mean(native_ensemble_energies) # energy of the unfolded state
Tmax = kB.__mul__(max_temp) # temperature max to plot; T in units of kBT
Zscore = (e_folded - e_unfolded_mean)/(e_unfolded_sigma) # Z-score for this folded state

# We will generate data from the specified unfolded Gaussian
# distibution; if we do the calcuation analytically, then we will, at
# low T, always find some structure that is lower than the folded
# state (Gaussian has population everywhere), so the average energy
# will go to zero as T->0, the folded state won't be the lowest state,
# etc.

# generate the unfolded state energies
unfolded = (e_unfolded_sigma * np.random.normal(size=n_unfolded)) - e_unfolded_mean

#stack all the energies together
all = np.concatenate([np.array(e_folded),unfolded],axis=None) 

# now compute thermodynamic properties as a function of T
toplotT = np.array([float(temp._value) for temp in temperature_list])
Tlen = len(toplotT)
A = np.zeros(Tlen)
E = np.zeros(Tlen)
S = np.zeros(Tlen)
C = np.zeros(Tlen)
Pf = np.zeros(Tlen)

for i,T in enumerate(toplotT):
    Q = np.sum(np.exp(-all/T))  # partition function
    A[i] = -T*np.log(Q) # free energy 

    # plot <E> vs. T
    #<E> = kT \ln \int E omega(E) exp(-E/T) / Q.

    E[i] = np.sum(all*np.exp(-all/T))
    E[i] = E[i]/Q

    # plot S vs. T
    # A = E - TS
    # S = (E - A)/T
    S[i] = (E[i]-A[i])/T

    # plot C_V vs T
    #C_V = (<E^2>-<E>)^T^2 =  
    # E^2 =  \int E^2 omega(E) exp(-E/T) / Q
    E2 = np.sum(all**2*np.exp(-all/T))
    E2 = E2/Q
    C[i] = (E2 - E[i]**2)/T**2

    # percent folded
    folded = np.exp(-e_folded/T)
    Pf[i] = folded/Q

print('Zscore = ',Zscore)
toplot = [A,Pf,E,S,C]
titles = ["Helmholtz free energy (A) vs. T",
          "Percent folded vs. T",
          "Energy (E) vs. T",
          "Entropy (S) vs. T",
          "Heat Capacity (C) vs. T"]
ylabels = ['A','% folded','E','S','C']

for p,t,y in zip(toplot,titles,ylabels):
    plt.plot(toplotT,p)
    plt.title(t)
    plt.xlabel('T')
    plt.ylabel(y)
    plt.show()
