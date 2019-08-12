import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.parameters.reweight import get_mbar_expectation, get_free_energy_differences, get_temperature_list
from foldamers.src.ensembles.ens_build import *
from cg_openmm.src.simulation.rep_exch import *

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 100.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
number_replicas = 100
min_temp = 10.0 * unit.kelvin
max_temp = 100.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

###
#
# Coarse grained model settings
#
###

# Global definitions
polymer_length=8
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
equil_bond_angle = 120
equil_bond_angles = {'bb_bb_bb_angle_0': equil_bond_angle,'bb_bb_sc_angle_0': equil_bond_angle,'bb_sc_sc_angle_0': equil_bond_angle,'sc_sc_sc_angle_0': equil_bond_angle, 'sc_bb_sc_angle_0': equil_bond_angle,'sc_sc_bb_angle_0': equil_bond_angle}

# Torsion properties
torsion_force_constant = 200
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': 0,'bb_bb_sc_sc_torsion_k': 0, 'bb_sc_sc_sc_torsion_k': 0, 'sc_bb_bb_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_bb_torsion_k': 0, 'sc_sc_sc_sc_torsion_k': 0,  'sc_bb_bb_bb_torsion_k': 0}

bb_bb_bb_bb_equil_torsion_angle_range = range(-45,50,5)
bb_bb_bb_bb_equil_torsion_angles = [float(equil_torsion_angle*3.1415/180.0) for equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angle_range]
sc_bb_bb_sc_equil_torsion_angle_range = range(-45,50,5)
sc_bb_bb_sc_equil_torsion_angles = [float(equil_torsion_angle*3.1415/180.0) for equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angle_range]
equil_torsion_angle = 0.0

z_scores = open(str(top_directory)+"/z_scores.dat","w")
z_scores.write("bb_bb_bb_bb_equil_torsion_angle,sc_bb_bb_sc_equil_torsion_angle,z_score\n")
z_scores.close()

for bb_bb_bb_bb_equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angles: 
 for sc_bb_bb_sc_equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angles:
  print("Performing simulations for a coarse grained model")
  print("with bb_bb_bb_bb torsion angles of "+str(round(bb_bb_bb_bb_equil_torsion_angle*180.0/3.1415,1))+" degrees")
  print("and sc_bb_bb_sc torsion angles of "+str(round(sc_bb_bb_sc_equil_torsion_angle*180.0/3.1415,1))+" degrees.")
  equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': bb_bb_bb_bb_equil_torsion_angle,'bb_bb_bb_sc_torsion_0': equil_torsion_angle,'bb_bb_sc_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_sc_torsion_0': sc_bb_bb_sc_equil_torsion_angle, 'bb_sc_sc_bb_torsion_0': equil_torsion_angle, 'sc_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_bb_torsion_0': equil_torsion_angle}
  cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds)

  # Run a replica exchange simulation with this cgmodel
  output_data = str(str(top_directory)+"/torsions_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+".nc")
  if os.path.exists(output_data):
    replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)
  else:
    replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  file_name = str(str(top_directory)+"/re_min_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+".pdb")
  native_structure = get_minimum_energy_pose(cgmodel.topology,replica_energies,replica_positions,file_name=file_name)

  nonnative_ensemble_directory = str(str(top_directory)+"/ens_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+"_nonnative")
  native_ensemble_directory = str(str(top_directory)+"/ens_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+"_native")
  if os.path.exists(nonnative_ensemble_directory):
    nonnative_ensemble,nonnative_ensemble_energies = get_ensemble_data(cgmodel,nonnative_ensemble_directory)
    if len(nonnative_ensemble_energies) == 0:
    print("ERROR: the nonnative ensemble energies were collected incorrectly.")
    exit()
  else:
    os.mkdir(nonnative_ensemble_directory)
    nonnative_ensemble,nonnative_ensemble_energies = get_nonnative_ensemble(cgmodel,native_structure)
    for pose in nonnative_ensemble:
      cgmodel.positions = pose
      write_ensemble_pdb(cgmodel,ensemble_directory=nonnative_ensemble_directory)
  if os.path.exists(native_ensemble_directory):
    native_ensemble,native_ensemble_energies = get_ensemble_data(cgmodel,native_ensemble_directory)
  else:
    os.mkdir(native_ensemble_directory)
    native_ensemble,native_ensemble_energies = get_native_ensemble(cgmodel,native_structure)
    for pose in native_ensemble:
      cgmodel.positions = pose
      write_ensemble_pdb(cgmodel,ensemble_directory=native_ensemble_directory)

  if len(nonnative_ensemble_energies) == 0:
    print("ERROR: the nonnative ensemble energies were collected incorrectly.")
    exit()
  if len(native_ensemble_energies) == 0:
    print("ERROR: the native ensemble energies were collected incorrectly.")
    exit()

  z = z_score(cgmodel.topology,cgmodel.system,nonnative_ensemble_energies,native_ensemble_energies)

  z_scores = open(str(top_directory)+"/z_scores.dat","a")
  z_scores.write(str(round(bb_bb_bb_bb_equil_torsion_angle,2))+","+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+","+str(round(z,2))+"\n")
  z_scores.close()

file_name = "Torsion_equil_angle_Z_scores.png"
figure = pyplot.figure(1)
bb_bb_bb_bb_equil_torsion_angles = np.array([float(equil_torsion_angle) for equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angle_range])
sc_bb_bb_sc_equil_torsion_angles = np.array([float(equil_torsion_angle) for equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angle_range])
z_scores = np.array([float(z) for z in z_scores])
#pyplot.plot(torsion_angles,z_scores,figure=figure)
print(z_scores)

x=np.unique(bb_bb_bb_bb_equil_torsion_angles)
y=np.unique(sc_bb_bb_sc_equil_torsion_angles)
X,Y = np.meshgrid(x,y)
Z=z_scores.reshape(len(y),len(x))

pyplot.xlabel(r"$ \alpha_{0}^{BB-BB-BB-BB} $ ( Degrees )")
pyplot.ylabel(r"$ \alpha_{0}^{SC-BB-BB-SC} $ ( Degrees )")
pyplot.title("$Z_{score}$ vs equil. torsion angles")
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()


exit()
