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

# Define Boltzmann's constant
kB = unit.Quantity(0.0019872041,unit.kilocalorie_per_mole)

# Use replica exchange trajectory data to build structural ensembles
total_simulation_time = 1.0 * unit.nanosecond
simulation_time_step = 5.0 * unit.femtosecond
output_data=str("output.nc")
print_frequency = 5
number_replicas = 50
min_temp = 1.0 * unit.kelvin
max_temp = 500.0 * unit.kelvin
mean_temp = mean([min_temp._value,max_temp._value])
mean_temp_kT = mean_temp*kB
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)

# Model settings
polymer_length=12
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]
include_bond_forces=False
include_bond_angle_forces=False
include_nonbonded_forces=True
include_torsion_forces=True
constrain_bonds = True

# Bond definitions
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 0 * unit.kilocalorie_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

# Particle definitions
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
r_min =  3.0 * bond_length # Lennard-Jones potential r_min
sigma =  r_min / (2.0**(1/6)) # Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
sigmas = {'bb_sigma': sigma,'sc_sigma': sigma}
# Set the value of epsilon equal to kT at the mean temperature
epsilon = 0.01 * unit.kilocalorie_per_mole
epsilons = {'bb_eps': epsilon,'sc_eps': epsilon}

# Torsion angle definitions
torsion_force_constant = 0.0001 * unit.kilocalorie_per_mole / unit.radian / unit.radian
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant}#,'sc_bb_bb_sc_torsion_k': torsion_force_constant}
bb_bb_bb_bb_equil_torsion_angle = 78.0 * (3.14/180.0) # OpenMM defaults to units of radians for angle definitions
#sc_bb_bb_sc_equil_torsion_angle = 110.0 * (3.14/180.0)
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': bb_bb_bb_bb_equil_torsion_angle}#,'sc_bb_bb_sc_torsion_0': sc_bb_bb_sc_equil_torsion_angle}
torsion_periodicities = {'bb_bb_bb_bb_period': 1}#,'sc_bb_bb_sc_period': 2}

# Initiate cgmodel using positions from local file
positions = PDBFile("init.pdb").getPositions()

# Build a coarse grained model using the positions for the initial structure
cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,torsion_force_constants=torsion_force_constants,equil_torsion_angles=equil_torsion_angles,torsion_periodicities=torsion_periodicities,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

if os.path.exists(output_data):
  # Search for existing data, and read it if possible
  print("Reading replica exchange data")
  replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)
else:
  # Run a replica exchange simulation with this cgmodel
  replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  make_replica_pdb_files(cgmodel.topology,replica_positions)

native_structure = get_native_structure(replica_positions,replica_energies,temperature_list)

# Set parameters for definition/evaluation of native contacts
native_structure_contact_distance_cutoff = 1.05 * cgmodel.get_sigma(0) # This distance cutoff determines which nonbonded interactions are considered 'native' contacts
native_contact_cutoff_ratio = 1.1 # The distance ratio (in comparison with the distance of a contact in the native structure) below which a nonbonded interaction is considered 'native'
native_fraction_cutoff = 0.95 # Poses with a fraction of native contacts greater than or equal to this value will be considered candidates for the native ensemble
nonnative_fraction_cutoff = 0.8 # Poses with a fraction of native contacts lower than or equal to this value will be considered candidates for the nonnative ensemble

native_ensemble_size = 10
nonnative_ensemble_size = 100

# Assemble "native" and "nonnative" ensembles from uncorrelated samples
native_ensemble,native_ensemble_energies,nonnative_ensemble,nonnative_ensemble_energies = get_ensembles_from_replica_positions(cgmodel,replica_positions,replica_energies,temperature_list,decorrelate=True,native_fraction_cutoff=native_fraction_cutoff,nonnative_fraction_cutoff=nonnative_fraction_cutoff,native_structure_contact_distance_cutoff=native_structure_contact_distance_cutoff,native_ensemble_size=native_ensemble_size,nonnative_ensemble_size=nonnative_ensemble_size)

e_unfolded_mean = mean(nonnative_ensemble_energies) # unfolded mean energy 
e_unfolded_sigma =  stdev(nonnative_ensemble_energies) # unfolded standard deviation energy
n_unfolded = len(nonnative_ensemble_energies) # ratio of number of unfolded states to folded states. We keep number of folded states at 1
e_folded = mean(native_ensemble_energies) # energy of the unfolded state
Tmax = kB.__mul__(max_temp) # temperature max to plot; T in units of kBT
Zscore = (e_folded - e_unfolded_mean)/(e_unfolded_sigma) # Z-score for this folded state

# We will generate data from the specified unfolded Gaussian
# distibution; if we do the calculation analytically, then we will, at
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
file_names = file_names = ["CG11_A_v_T.png","CG11_Q_v_T.png","CG11_E_v_T.png","CG11_S_v_T.png","CG11_C_v_T.png"]

for p,t,y,name in zip(toplot,titles,ylabels,file_names):
    figure = plt.figure(1)
    plt.plot(toplotT,p)
    plt.title(t)
    plt.xlabel('T')
    plt.ylabel(y)
    plt.show()
    plt.savefig(name)
    plt.close()
