import os
import csv
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from statistics import mean
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.parameters.reweight import *
from foldamers.src.parameters.secondary_structure import *
from foldamers.src.thermo.calc import *
from foldamers.src.utilities.plot import plot_distribution
from foldamers.src.utilities.util import random_positions
from cg_openmm.src.build.cg_build import build_topology
from cg_openmm.src.simulation.rep_exch import *

grid_size = 5

native_structure_file = str(str(os.getcwd().split('examples/')[0])+"ensembles/12_1_1_0/helix.pdb")

native_structure = PDBFile(native_structure_file).getPositions()

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas = 30
min_temp = 50.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
print("Using "+str(len(temperature_list))+" replicas.")

torsion_list = []
bond_angle_list = []

angle_bins = []
angle_bin_counts = []
torsion_bins = []
torsion_bin_counts = []

torsion_force_constant_list = [ 0.0001 * 10.0**float(i) for i in range(grid_size)]
for torsion_force_constant in torsion_force_constant_list:
  torsion_force_constants={'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': 0.0,'bb_bb_sc_sc_torsion_k': 0.00, 'bb_sc_sc_sc_torsion_k': 0.0, 'sc_bb_bb_sc_torsion_k': 0.0, 'sc_sc_sc_sc_torsion_k': 0.0, 'sc_bb_bb_bb_torsion_k': 0.0}
  print("Calculating the free energy as a function of RMSD to the idyllic helical structure.")
  print("for a coarse grained model with torsion force constants of "+str(torsion_force_constant))
  cgmodel = CGModel(torsion_force_constants=torsion_force_constants)

  output_data = str(str(top_directory)+"/tfc_"+str(torsion_force_constant)+".nc")
  if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  else:
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  model_torsion_list = cgmodel.torsion_list
  backbone_torsion_list = []
  for torsion in model_torsion_list:
   if all([cgmodel.get_particle_type(particle) == "backbone" for particle in torsion]):
    backbone_torsion_list.append(torsion)
  model_angle_list = cgmodel.bond_angle_list
  backbone_bond_angle_list = []
  for angle in model_angle_list:
   if all([cgmodel.get_particle_type(particle) == "backbone" for particle in angle]):
    backbone_bond_angle_list.append(angle)

  model_torsion_list = backbone_torsion_list
  model_angle_list = backbone_bond_angle_list

  torsions = []
  bond_angles = []
  make_replica_pdb_files(cgmodel.topology,replica_positions)
  for replica_index in range(len(replica_positions)):
     trajectory = md.load(str("replica_"+str(replica_index+1)+".pdb"))
     traj_torsions = md.compute_dihedrals(trajectory,model_torsion_list)
     for torsion_list in traj_torsions:
      for torsion in torsion_list:
       torsions.append(torsion)
     traj_angles = md.compute_angles(trajectory,model_angle_list)
     for angle_list in traj_angles:
      for angle in angle_list:
       bond_angles.append(angle)

  torsions = np.array([float(torsion) for torsion in torsions])
  bond_angles = np.array([float(angle) for angle in bond_angles])

  n_bond_angle_bins = 100
  n_torsion_bins = 100
  min_bond_angle = bond_angles[np.argmin(bond_angles)]
  max_bond_angle = bond_angles[np.argmax(bond_angles)]
  min_torsion = torsions[np.argmin(torsions)]
  max_torsion = torsions[np.argmax(torsions)]
  bond_angle_step = (max_bond_angle-min_bond_angle)/(n_bond_angle_bins+1)
  torsion_step = (max_torsion-min_torsion)/(n_torsion_bins+1)
  bond_angle_ranges = [[min_bond_angle+bond_angle_step*i,min_bond_angle+bond_angle_step*(i+1)] for i in range(n_bond_angle_bins+1)]
  torsion_ranges = [[min_torsion+torsion_step*i,min_torsion+torsion_step*(i+1)] for i in range(n_torsion_bins+1)]

  bond_angle_bin_counts = np.zeros((n_bond_angle_bins+1),dtype=int)
  backbone_torsion_bin_counts = np.zeros((n_torsion_bins+1),dtype=int)

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
  for index in range(len(torsions)):
   for torsion_range_index in range(len(torsion_ranges)):
    torsion_range = torsion_ranges[torsion_range_index]
    if torsion_range_index == 0:
     if float(torsions[index]) >= float(torsion_range[0]) and float(torsions[index]) <= float(torsion_range[1]):
      torsion_bin_index = torsion_range_index
    else:
     if float(torsions[index]) > float(torsion_range[0]) and float(torsions[index]) <= float(torsion_range[1]):
      torsion_bin_index = torsion_range_index
   backbone_torsion_bin_counts[torsion_bin_index] = backbone_torsion_bin_counts[torsion_bin_index] + 1

  angle_bin_counts.append(bond_angle_bin_counts)
  torsion_bin_counts.append(backbone_torsion_bin_counts)

  bond_angle_bins = np.array([mean(np.array([float(bond_angle*(180.0/3.14)) for bond_angle in bond_angle_range])) for bond_angle_range in bond_angle_ranges])
  torsion_angle_bins = np.array([mean(np.array([float(torsion*(180.0/3.14)) for torsion in torsion_range])) for torsion_range in torsion_ranges])
  angle_bins.append(bond_angle_bins)
  torsion_bins.append(torsion_angle_bins)

plot_distribution(torsion_bins,torsion_bin_counts,plot_type="Torsions",multiple=True,legend=torsion_force_constant_list,legend_title='$k_{torsion}$')
exit()

