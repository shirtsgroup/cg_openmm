####
#
# Garrett developed this example on 9/12/19, for Connor.
#
####

####
#
# Import all of the packages that we need.
#
####
import os
from statistics import mean
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import get_mbar_expectation, get_free_energy_differences, get_temperature_list
from foldamers.ensembles.ens_build import *
from cg_openmm.simulation.rep_exch import *

# Make a directory for output files
output_directory = 'output'
if not os.path.exists(output_directory):
  os.mkdir(output_directory)

print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 500.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond

###
#
# MODEL SETTINGS
#
# The following lines provide all possible 
# settings for the coarse grained model you
# will be working with.
#
# Try to become comfortable with the meaning
# of the variables, from both a software and
# standpoint as well as a scientific standpoint.
# 
# Next time we meet, let's talk about the meaning
# of the variables in the first section, 
# 'Model settings'.  We'll go
# through the meaning of any that are unclear.
# From there, I can gauge your understanding to
# decide how much depth to use when discussing the
# other variables.
#
###

# Model settings
polymer_length=12
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]
include_bond_forces=False
include_bond_angle_forces=True
include_nonbonded_forces=True
include_torsion_forces=True
constrain_bonds = True

# Particle properties
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}

# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

sigma = bond_length * 2.5
sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma}
epsilon = 0.5 * unit.kilocalorie_per_mole
epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': 0.5 * epsilon}

# Bond angle properties
bond_angle_force_constant = 200 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': bond_angle_force_constant,'bb_sc_sc_angle_k': bond_angle_force_constant,'sc_sc_sc_angle_k': bond_angle_force_constant, 'sc_bb_sc_angle_k': bond_angle_force_constant, 'sc_sc_bb_angle_k': bond_angle_force_constant}
equil_bond_angle = 1.61
equil_bond_angles = {'bb_bb_bb_angle_0': equil_bond_angle,'bb_bb_sc_angle_0': 0.0}

# Torsion properties
torsion_force_constant = 200
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': 0,'bb_bb_sc_sc_torsion_k': 0,'bb_sc_sc_sc_torsion_k': 0,'sc_bb_bb_sc_torsion_k': torsion_force_constant,'bb_sc_sc_bb_torsion_k': 0,'sc_sc_sc_sc_torsion_k': 0,'sc_bb_bb_bb_torsion_k': 0,'bb_bb_bb_sc_torsion_k': 0}

###
#
# These are the main parameters we are focusing on with this scan/example
# Let's talk about what these angles are, visually, when we meet next.
#
###

# This parameter is known to give stable helix, we probably don't need to change it too much until we are modeling something new.
bb_bb_bb_bb_equil_torsion_angle = 0.91
# This parameter was chosen randomly, more or less.  So I'm not at all confident that it has a reasonable value to allow helix formation in our model.  This is where I would like your help.
sc_bb_bb_sc_equil_torsion_angle = 0.48
###
#
# We want to identify values for 
# 'sc_bb_bb_sc_equil_torsion_angle' 
# that give "better helices".
#
# By "better", we mean "more cooperative".
# By "more cooperative" we mean (for now)
# that the folded structure has a significantly
# lower energy than unfolded structures.
# By "helix" we mean the model exhibits
# a rotational axis, about which the polymer
# folds (forms nonbonded interactions).
#
# How can we go about identifying "better helices"?
#
###

bb_bb_bb_bb_equil_torsion_angles = [float(bb_bb_bb_bb_equil_torsion_angle+i*0.01) for i in range(-5,5,1)]
sc_bb_bb_sc_equil_torsion_angles = [float(sc_bb_bb_sc_equil_torsion_angle+i*0.01) for i in range(-5,5,1)]

# We build an array to store data we will read from the output
dQ_list = []

# This is where we start iterating over (scanning) different torsion angles
for bb_bb_bb_bb_equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angles: 
 for sc_bb_bb_sc_equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angles:
  print("Performing simulations for a coarse grained model")
  print("with bb_bb_bb_bb torsion angles of "+str(round(bb_bb_bb_bb_equil_torsion_angle,1))+" radians.")
  print("and sc_bb_bb_sc torsion angles of "+str(round(sc_bb_bb_sc_equil_torsion_angle,1))+" radians.")
  # Set the torsion angles for this grid point.
  equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': bb_bb_bb_bb_equil_torsion_angle, 'sc_bb_bb_sc_torsion_0': sc_bb_bb_sc_equil_torsion_angle,'bb_bb_bb_sc_torsion_0':0.0,'sc_bb_bb_bb_torsion_0':0.0}
  # This is where we build a coarse grained model that has the torsion parameters for this grid point.
  positions = PDBFile("helix.pdb").getPositions()
  cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

  output_data = str(str(output_directory)+"/torsions_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+".nc")
  # We will probably end up analyzing this dataset over and over again,
  # and it takes a long time to generate new data each time,
  # so this script will search for existing data before performing new
  # simulations.  To perform a fresh run you will want to delete the
  # 'output_data' folder.
  file_name = str(str(output_directory)+"/re_min_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+".pdb")
  if os.path.exists(output_data):
    replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)
  # Find the "native structure", or lowest energy pose for our polymer 
    native_structure = get_native_structure(replica_positions,replica_energies,temperature_list)
  else:
    # Run a replica exchange simulation with this cgmodel
    replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
    native_structure = PDBFile(file_name).getPositions()

  native_contact_cutoff_distance = 0.8 * cgmodel.get_sigma(0)
  native_ensemble,native_ensemble_energies,nonnative_ensemble,nonnative_ensemble_energies = get_ensembles_from_replica_positions(cgmodel,replica_positions,replica_energies,temperature_list,decorrelate=False,nonnative_fraction_cutoff=0.9,native_contact_cutoff_distance=native_contact_cutoff_distance)

  nonnative_ensemble_directory = str(str(output_directory)+"/ens_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+"_nonnative")
  native_ensemble_directory = str(str(output_directory)+"/ens_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+"_native")
  # We build an ensemble of nonnative poses for energetic comparison with the native pose.
  if os.path.exists(nonnative_ensemble_directory):
    nonnative_ensemble,nonnative_ensemble_energies = get_ensemble_data(cgmodel,nonnative_ensemble_directory)
    if len(nonnative_ensemble_energies) == 0:
     print("ERROR: the nonnative ensemble energies were collected incorrectly.")
     exit()
  else:
    os.mkdir(nonnative_ensemble_directory)
    for pose in nonnative_ensemble:
      cgmodel.positions = pose
      write_ensemble_pdb(cgmodel,ensemble_directory=nonnative_ensemble_directory)

  nonnative_ensemble_fraction_native_contacts = []
  for pose in nonnative_ensemble:
    Q = fraction_native_contacts(cgmodel,pose,native_structure)
    nonnative_ensemble_fraction_native_contacts.append(Q)
  nonnative_ensemble_fraction_native_contacts = np.array([Q for Q in nonnative_ensemble_fraction_native_contacts])
  mean_contacts = mean(nonnative_ensemble_fraction_native_contacts)

  # We build an ensemble of native poses in order to understand the energy distribution around the folded state.
  if os.path.exists(native_ensemble_directory):
    native_ensemble,native_ensemble_energies = get_ensemble_data(cgmodel,native_ensemble_directory)
  else:
    os.mkdir(native_ensemble_directory)
    for pose in native_ensemble:
      cgmodel.positions = pose
      write_ensemble_pdb(cgmodel,ensemble_directory=native_ensemble_directory)

  if len(nonnative_ensemble_energies) == 0:
    print("ERROR: the nonnative ensemble energies were collected incorrectly.")
    exit()
  if len(native_ensemble_energies) == 0:
    print("ERROR: the native ensemble energies were collected incorrectly.")
    exit()

  # Get the average change in the fraction of native contacts during folding (dQ),
  # calculated as the difference between the average fraction of native contacts
  # in the nonnative ensemble.
  # A large dQ means the model/structure has a stable folded state.
  # A small dQ means the model/structure does not have a very stable folded state.
  dQ = 1.0 - mean_contacts
  dQ_list.append(dQ)

file_name = "dQ_for_variable_equil_torsion_angles.png"
figure = pyplot.figure(1)
bb_bb_bb_bb_equil_torsion_angles = np.array([float(equil_torsion_angle) for equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angle_range])
sc_bb_bb_sc_equil_torsion_angles = np.array([float(equil_torsion_angle) for equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angle_range])

x=np.unique(bb_bb_bb_bb_equil_torsion_angles)
y=np.unique(sc_bb_bb_sc_equil_torsion_angles)
X,Y = np.meshgrid(x,y)
Z=dQ_list.reshape(len(x),len(y))

pyplot.xlabel(r"$ \alpha_{0}^{BB-BB-BB-BB} $ ( Degrees )")
pyplot.ylabel(r"$ \alpha_{0}^{SC-BB-BB-SC} $ ( Degrees )")
pyplot.title("dQ (Change in native contacts during folding)")
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()


exit()
