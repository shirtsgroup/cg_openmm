###
#
# This script can be used to determine suitable
# settings for calculating 'native contacts'.
#
###

import os
import numpy as np
from statistics import mean
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.ensembles.ens_build import *
from foldamers.parameters.secondary_structure import *
from cg_openmm.simulation.rep_exch import *

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
epsilon = 0.1 * unit.kilocalorie_per_mole
epsilons = {'bb_eps': epsilon,'sc_eps': epsilon}

# Torsion angle definitions (Used to establish a scanning range below)
torsion_force_constant = 0.1 * unit.kilocalorie_per_mole / unit.radian / unit.radian
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant}#,'sc_bb_bb_sc_torsion_k': torsion_force_constant}
bb_bb_bb_bb_equil_torsion_angle = 78.0 * (3.14/180.0) # OpenMM defaults to units of radians for angle definitions
#sc_bb_bb_sc_equil_torsion_angle = 110.0 * (3.14/180.0)
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': bb_bb_bb_bb_equil_torsion_angle}#,'sc_bb_bb_sc_torsion_0': sc_bb_bb_sc_equil_torsion_angle}
torsion_periodicities = {'bb_bb_bb_bb_period': 1}#,'sc_bb_bb_sc_period': 2}

# Initiate cgmodel using positions from local file
positions = PDBFile("init.pdb").getPositions()

# Build a coarse grained model for the native structure
cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,torsion_force_constants=torsion_force_constants,equil_torsion_angles=equil_torsion_angles,torsion_periodicities=torsion_periodicities,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

# Use replica exchange trajectory data to build structural ensembles
total_simulation_time = 1.0 * unit.nanosecond
simulation_time_step = 5.0 * unit.femtosecond
output_data=str("output.nc")
print_frequency = 5
number_replicas = 50
min_temp = 1.0 * unit.kelvin
max_temp = 500.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
if os.path.exists(output_data):
  # Search for existing data, and read it if possible
  replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)
else:
  # Run a replica exchange simulation with this cgmodel
  replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  make_replica_pdb_files(cgmodel.topology,replica_positions)


native_structure = get_native_structure(replica_positions,replica_energies,temperature_list)

random_poses,random_pose_energies = get_decorrelated_samples(replica_positions,replica_energies,temperature_list)

# Set parameters for definition/evaluation of native contacts
native_structure_contact_distance_cutoff = 1.05 * cgmodel.get_sigma(0) # This distance cutoff determines which nonbonded interactions are considered 'native' contacts
native_contact_cutoff_ratio = 1.1 # The distance ratio (in comparison with the distance of a contact in the native structure) below which a nonbonded interaction is considered 'native'

cutoff_list = [(1.0+i*.01)*cgmodel.get_sigma(0) for i in range(30)]

list_fraction_native_contacts = []

for cutoff in cutoff_list:
 # Get the number of nonbonded interactions that will be considered 'native'

 # Get the fraction of native contacts for the comparison pose
 #native_ensemble,native_ensemble_energies,nonnative_ensemble,nonnative_ensemble_energies = get_ensembles_from_replica_positions(cgmodel,replica_positions,replica_energies,temperature_list,decorrelate=True,native_fraction_cutoff=native_fraction_cutoff,nonnative_fraction_cutoff=nonnative_fraction_cutoff,native_structure_distance_cutoff=native_structure_distance_cutoff,native_ensemble_size=native_ensemble_size,nonnative_ensemble_size=nonnative_ensemble_size,optimize=False)
 nonnative_ensemble_Q_list = []
 for pose in random_poses:
  Q = fraction_native_contacts(cgmodel,pose,native_structure,native_structure_contact_distance_cutoff=cutoff,native_contact_cutoff_ratio=native_contact_cutoff_ratio)
  nonnative_ensemble_Q_list.append(Q)
 mean_Q = mean(nonnative_ensemble_Q_list)

 list_fraction_native_contacts.append(mean_Q)
 #print("The fraction of native contacts in the comparison pose is: "+str(Q))

x = np.array([float(cutoff.__div__(cgmodel.get_sigma(0))) for cutoff in cutoff_list])
y = np.array([float(fraction_native_contacts) for fraction_native_contacts in list_fraction_native_contacts])

pyplot.xlabel(r"Native contact cutoff distance (in units of $\sigma$)")
pyplot.ylabel("<Q> (Fraction native contacts)")
pyplot.plot(x,y)
pyplot.savefig("Q_vs_cutoff_distance.png")
pyplot.show()

exit()
