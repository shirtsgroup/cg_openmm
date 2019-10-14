import os, shutil
import numpy as np
import matplotlib.pyplot as pyplot
from statistics import mean
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.thermo.calc import *
from foldamers.parameters.reweight import *
from foldamers.ensembles.ens_build import *
from foldamers.utilities.util import *
from foldamers.utilities.plot import *
from cg_openmm.build.cg_build import *
from cg_openmm.simulation.tools import *
from cg_openmm.simulation.rep_exch import *

# Job settings
output_directory = 'output'
if not os.path.exists(output_directory):
  os.mkdir(output_directory)

# OpenMM simulation settings
print_frequency = 1 # Number of steps to skip when printing output
total_simulation_time = 10.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 1.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(output_directory)+"/output.nc")
number_replicas = 30
min_temp = 5.0 * unit.kelvin
max_temp = 100.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

polymer_length = 15
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]
constrain_bonds = True
include_bond_forces=False
include_bond_angle_forces=True
include_nonbonded_forces=True
include_torsion_forces=False

mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
bond_length = 1.0 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 1000.0 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

epsilon = 0.5 * unit.kilocalorie_per_mole
epsilons = {'bb_bb_eps': epsilon,'sc_sc_eps': epsilon}
sigma = bond_length / (2.0**(1/6))
sigmas = {'bb_bb_sigma': sigma,'sc_sc_sigma': sigma}

bond_angle_force_constant = 0.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': 0.0}
equil_bond_angle = 120.0*(3.14/180.0)
equil_bond_angles = {'bb_bb_bb_angle_0': equil_bond_angle, 'bb_bb_sc_angle_0': 0.0}

positions = PDBFile("init.pdb").getPositions()
cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,equil_bond_angles=equil_bond_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

cgmodel.simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=1.0 * unit.kelvin,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency)

model_angle_list = cgmodel.bond_angle_list
angle_list = []
for angle in model_angle_list:
  if all([cgmodel.get_particle_type(i) == "backbone" for i in angle]):
    angle_list.append(angle)
bond_angles = []
trajectory = md.load("init.pdb")
for angle in angle_list:
      traj_angles = md.compute_angles(trajectory,[angle])
      for sample in traj_angles:
       bond_angles.append(sample[0])

run_simulation(cgmodel,output_directory,total_simulation_time,simulation_time_step,1.0*unit.kelvin,20,output_pdb="output.pdb",output_data="output.dat")

exit()

model_angle_list = cgmodel.bond_angle_list
angle_list = []
for angle in model_angle_list:
  if all([cgmodel.get_particle_type(i) == "backbone" for i in angle]):
    angle_list.append(angle)

init_pose = False
attempts = 0
max_attempts = 100
while not os.path.exists("native_structure.pdb"):
 try:
  print("reading in a structure")
  cgmodel.positions = PDBFile("init.pdb").getPositions()
  print("minimize the structure")
  cgmodel.positions,energy,simulation = minimize_structure(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=1.0*unit.kelvin,simulation_time_step=simulation_time_step,output_data="output.dat",output_pdb="output.pdb")
  print("minimization succeeded.")
  if not os.path.exists(output_data):
     print("performing replica exchange simulations")
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
     make_replica_pdb_files(cgmodel.topology,replica_positions)
  else:
     print("reading replica exchange simulation data")
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)
  native_structure = get_native_structure(replica_positions,replica_energies,temperature_list)

  cgmodel.positions = get_native_structure(replica_positions,replica_energies,temperature_list)


  native_structure,native_structure_energy,simulation = minimize_structure(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=1.0*unit.kelvin,simulation_time_step=simulation_time_step)

  cgmodel.positions = native_structure

  write_pdbfile_without_topology(cgmodel,"native_structure.pdb",energy=native_structure_energy)

 except:
  if os.path.exists(output_directory):
   shutil.rmtree(output_directory)
  if attempts > max_attempts:
   print("Error: failed to identify a suitable initial configuration for this model.")
   exit()
  else:
   attempts = attempts + 1

bond_angles = []
for replica_index in range(len(replica_positions)):
     trajectory = md.load(str("replica_"+str(replica_index+1)+".pdb"))
     for angle in angle_list:
      traj_angles = md.compute_angles(trajectory,[angle])
      for sample in traj_angles:
       bond_angles.append(sample)

bond_angles = np.array([float(angle) for angle in bond_angles])

n_bond_angle_bins = 50
bond_angle_bin_counts = np.zeros((n_bond_angle_bins+1),dtype=int)
min_bond_angle = bond_angles[np.argmin(bond_angles)]
max_bond_angle = bond_angles[np.argmax(bond_angles)]
bond_angle_step = (max_bond_angle-min_bond_angle)/(n_bond_angle_bins+1)
bond_angle_ranges = [[min_bond_angle+bond_angle_step*i,min_bond_angle+bond_angle_step*(i+1)] for i in range(n_bond_angle_bins+1)]

for index in range(len(bond_angles)):
   for bond_angle_range_index in range(len(bond_angle_ranges)):
    bond_angle_range = bond_angle_ranges[bond_angle_range_index]
    if bond_angle_range_index == 0:
     if float(bond_angles[index]) >= float(bond_angle_range[0]) and float(bond_angles[index]) <= float(bond_angle_range[1]):
      bond_angle_bin_index = bond_angle_range_index
    else:
     if float(bond_angles[index]) > float(bond_angle_range[0]) and float(bond_angles[index]) <= float(bond_angle_range[1]):
      bond_angle_bin_index = bond_angle_range_index
   bond_angle_bin_counts[bond_angle_bin_index] = bond_angle_bin_counts[bond_angle_bin_index] + 1

x = np.array([mean(np.array([float(bond_angle*(180.0/3.14)) for bond_angle in bond_angle_range])) for bond_angle_range in bond_angle_ranges])
y = np.array([float(count) for count in bond_angle_bin_counts])

plot_distribution(x,y,plot_type="Angles",multiple=False,plot_title=str(r"$\theta_{BBB}$ distribution"))

exit()
