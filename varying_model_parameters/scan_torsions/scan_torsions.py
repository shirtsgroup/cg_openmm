import os
from statistics import mean
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.ensembles.ens_build import *
from cg_openmm.simulation.rep_exch import *

# Job settings
output_directory = 'output'
if not os.path.exists(output_directory):
  os.mkdir(output_directory)
# Number of grid points to scan (around initial angle definitions)
grid_points = 3

# Configure Yank (replica exchange) simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 50.0 * unit.picosecond
simulation_time_step = 5.0 * unit.femtosecond
number_replicas = 30
min_temp = 1.0 * unit.kelvin
max_temp = 500.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)

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
epsilon = 0.05 * unit.kilocalorie_per_mole
epsilons = {'bb_eps': epsilon,'sc_eps': epsilon}

# Bond angle definitions
bond_angle_force_constant = 0.005 * unit.kilocalorie_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': bond_angle_force_constant}
bb_bb_bb_equil_bond_angle = 120.0 * (3.14/180.0) # OpenMM expects angle definitions in units of radians
bb_bb_sc_equil_bond_angle = 120.0 * (3.14/180.0)
equil_bond_angles = {'bb_bb_bb_angle_0': bb_bb_bb_equil_bond_angle,'bb_bb_sc_angle_0': bb_bb_sc_equil_bond_angle}

# Torsion angle definitions (Used to establish a scanning range below)
torsion_force_constant = 0.005 * unit.kilocalorie_per_mole / unit.radian / unit.radian
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'sc_bb_bb_sc_torsion_k': torsion_force_constant}
bb_bb_bb_bb_equil_torsion_angle = 78.0 * (3.14/180.0) # OpenMM defaults to units of radians for angle definitions
sc_bb_bb_sc_equil_torsion_angle = 110.0 * (3.14/180.0)
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': bb_bb_bb_bb_equil_torsion_angle,'sc_bb_bb_sc_torsion_0': sc_bb_bb_sc_equil_torsion_angle}
torsion_periodicities = {'bb_bb_bb_bb_period': 1,'sc_bb_bb_sc_period': 2}

# Get initial positions from local file
positions = PDBFile("helix.pdb").getPositions()

# Build a coarse grained model
cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,torsion_periodicities=torsion_periodicities,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

bb_bb_bb_bb_equil_torsion_angles = [float(bb_bb_bb_bb_equil_torsion_angle+i*0.05) for i in range(-grid_points,grid_points,1)]
sc_bb_bb_sc_equil_torsion_angles = [float(sc_bb_bb_sc_equil_torsion_angle+i*0.05) for i in range(-grid_points,grid_points,1)]

# We build an array to store data we will read from the output
dQ_list = []

# This is where we start evaluating the properties of models with different equilibrium torsion angles
for bb_bb_bb_bb_equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angles: 
 for sc_bb_bb_sc_equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angles:
  print("Performing simulations for a coarse grained model")
  print("with bb_bb_bb_bb torsion angles of "+str(round(bb_bb_bb_bb_equil_torsion_angle*(180.0/3.14),1))+" degrees.")
  print("and sc_bb_bb_sc torsion angles of "+str(round(sc_bb_bb_sc_equil_torsion_angle*(180.0/3.14),1))+" degrees.\n")
  # Set the torsion angles for this grid point.
  equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': bb_bb_bb_bb_equil_torsion_angle, 'sc_bb_bb_sc_torsion_0': sc_bb_bb_sc_equil_torsion_angle}
  # Build a coarse grained model that has the torsion parameters for this grid point.
  positions = PDBFile("helix.pdb").getPositions()
  cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,torsion_periodicities=torsion_periodicities,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

  output_data = str(str(output_directory)+"/torsions_"+str(round(bb_bb_bb_bb_equil_torsion_angle*(180.0/3.14),1))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle*(180.0/3.14),1))+".nc")
  file_name = str(str(output_directory)+"/re_min_"+str(round(bb_bb_bb_bb_equil_torsion_angle*(180.0/3.14),1))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle*(180.0/3.14),1))+".pdb")
  if os.path.exists(file_name):
    # Search for existing data, and read it if possible
    replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)
    # Find the lowest energy pose for this model
    native_structure = PDBFile(file_name).getPositions()
  else:
    # Run a replica exchange simulation with this cgmodel
    replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
    native_structure = get_native_structure(replica_positions,replica_energies,temperature_list)
    file = open(file_name,"w")
    PDBFile.writeFile(cgmodel.topology,native_structure,file=file)
    file.close()

  native_structure_contact_distance_cutoff = 1.15 * cgmodel.get_sigma(0) # This distance cutoff determines which nonbonded interactions are considered 'native' contacts
  native_fraction_cutoff = 0.95 # The threshold fraction of native contacts above which a pose is considered 'native'
  nonnative_fraction_cutoff = 0.95 # The threshold fraction of native contacts below which a pose is considered 'nonnative'
  native_ensemble_size = 10
  nonnative_ensemble_size = 10
  decorrelate=False
  native_ensemble,native_ensemble_energies,nonnative_ensemble,nonnative_ensemble_energies = get_ensembles_from_replica_positions(cgmodel,replica_positions,replica_energies,temperature_list,decorrelate=decorrelate,native_fraction_cutoff=native_fraction_cutoff,nonnative_fraction_cutoff=nonnative_fraction_cutoff,native_structure_contact_distance_cutoff=native_structure_contact_distance_cutoff,native_ensemble_size=native_ensemble_size,nonnative_ensemble_size=nonnative_ensemble_size)
  if len(native_ensemble_energies) != native_ensemble_size or len(nonnative_ensemble_energies) != nonnative_ensemble_size:
    print("ERROR: attempt to generate native and nonnative ensembles was unsuccessful.")
    print(str(len(native_ensemble_energies))+" native ensemble members were generated ("+str(native_ensemble_size)+" were requested),")
    print("and "+str(len(nonnative_ensemble_energies))+" non-native ensemble members were generated ("+str(nonnative_ensemble_size)+" were requested).")
    print("Try adjusting the 'native_structure_distance_cutoff' parameter (current value="+str(native_structure_contact_distance_cutoff.__div__(cgmodel.get_sigma(0)))+"*'bb_sigma'),")
    print("and the 'nonnative_fraction_cutoff' parameter (current value="+str(nonnative_fraction_cutoff)+")")
    print("to see if either of these approaches fixes the problem.")
    exit()
  nonnative_ensemble_directory = str(str(output_directory)+"/ens_"+str(round(bb_bb_bb_bb_equil_torsion_angle*(180.0/3.14),1))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle*(180.0/3.14),1))+"_nonnative")
  native_ensemble_directory = str(str(output_directory)+"/ens_"+str(round(bb_bb_bb_bb_equil_torsion_angle*(180.0/3.14),1))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle*(180.0/3.14),1))+"_native")
  # We build an ensemble of nonnative poses for energetic comparison with the native pose.
  if os.path.exists(nonnative_ensemble_directory):
    nonnative_ensemble,nonnative_ensemble_energies = get_ensemble_data(cgmodel,nonnative_ensemble_directory)
    if len(nonnative_ensemble) != nonnative_ensemble_size:
      print("ERROR: "+str(len(nonnative_ensemble_energies))+" nonnative poses were found in existing output folders, but "+str(nonnative_ensemble_size)+" poses were requested.")
      print("This probably means that the requested ensemble size changed since the script was last run.")
      exit()
  else:
    os.mkdir(nonnative_ensemble_directory)
    for pose in nonnative_ensemble:
      cgmodel.positions = pose
      write_ensemble_pdb(cgmodel,ensemble_directory=nonnative_ensemble_directory)

  nonnative_ensemble_Q = []
  for pose in nonnative_ensemble:
    Q = fraction_native_contacts(cgmodel,pose,native_structure)
    nonnative_ensemble_Q.append(Q)
  nonnative_ensemble_Q = np.array([Q for Q in nonnative_ensemble_Q])
  mean_nonnative_contacts = mean(nonnative_ensemble_Q)

  # We build an ensemble of native poses in order to understand the energy distribution around the folded state.
  if os.path.exists(native_ensemble_directory):
    native_ensemble,native_ensemble_energies = get_ensemble_data(cgmodel,native_ensemble_directory)
    if len(native_ensemble_energies) != native_ensemble_size:
      print("ERROR: "+str(len(native_ensemble_energies))+" native poses were found in existing output folders, but "+str(native_ensemble_size)+" poses were requested.")
      print("This probably means that the requested ensemble size changed since the script was last run.")
      exit()
  else:
    os.mkdir(native_ensemble_directory)
    for pose in native_ensemble:
      cgmodel.positions = pose
      write_ensemble_pdb(cgmodel,ensemble_directory=native_ensemble_directory)

  # Get the average change in the fraction of native contacts during folding (dQ),
  # calculated as the difference between the average fraction of native contacts
  # in the nonnative ensemble.
  # A large dQ means the model/structure has a stable folded state.
  # A small dQ means the model/structure does not have a stable folded state.
  dQ = 1.0 - mean_nonnative_contacts
  dQ_list.append(dQ)

file_name = "dQ_for_variable_equil_torsion_angles.png"
figure = pyplot.figure(1)
bb_bb_bb_bb_equil_torsion_angles = np.array([float(equil_torsion_angle) for equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angle_range])
sc_bb_bb_sc_equil_torsion_angles = np.array([float(equil_torsion_angle) for equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angle_range])

x=np.unique(bb_bb_bb_bb_equil_torsion_angles*(180.0/3.14))
y=np.unique(sc_bb_bb_sc_equil_torsion_angles*(180.0/3.14))
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
