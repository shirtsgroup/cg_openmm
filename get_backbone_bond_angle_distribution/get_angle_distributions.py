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

bond_angle_list = []

angle_bins = []
angle_bin_counts = []

bond_angle_force_constant_list = [ 0.0001 * 10.0**float(i) for i in range(grid_size)]
for bond_angle_force_constant in bond_angle_force_constant_list:
  bond_angle_force_constants={'bb_bb_bb_bond_angle_k': bond_angle_force_constant,'bb_bb_sc_bond_angle_k': 0.0,'bb_sc_sc_bond_angle_k': 0.00}
  print("Calculating the free energy as a function of RMSD to the idyllic helical structure.")
  print("for a coarse grained model with bond_angle force constants of "+str(bond_angle_force_constant))
  cgmodel = CGModel(bond_angle_force_constants=bond_angle_force_constants)

  output_data = str(str(top_directory)+"/bafc_"+str(bond_angle_force_constant)+".nc")
  if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  else:
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  model_bond_angle_list = cgmodel.bond_angle_list
  backbone_bond_angle_list = []
  for bond_angle in model_bond_angle_list:
   if all([cgmodel.get_particle_type(particle) == "backbone" for particle in bond_angle]):
    backbone_bond_angle_list.append(bond_angle)

  model_bond_angle_list = backbone_bond_angle_list

  bond_angles = []
  make_replica_pdb_files(cgmodel.topology,replica_positions)
  for replica_index in range(len(replica_positions)):
     trajectory = md.load(str("replica_"+str(replica_index+1)+".pdb"))
     traj_bond_angles = md.compute_angles(trajectory,model_bond_angle_list)
     for bond_angle_list in traj_bond_angles:
      for bond_angle in bond_angle_list:
       bond_angles.append(bond_angle)

  bond_angles = np.array([float(bond_angle) for bond_angle in bond_angles])

  n_bond_angle_bins = 100
  min_bond_angle = bond_angles[np.argmin(bond_angles)]
  max_bond_angle = bond_angles[np.argmax(bond_angles)]
  bond_angle_step = (max_bond_angle-min_bond_angle)/(n_bond_angle_bins+1)
  bond_angle_ranges = [[min_bond_angle+bond_angle_step*i,min_bond_angle+bond_angle_step*(i+1)] for i in range(n_bond_angle_bins+1)]
  bond_angle_bin_counts = np.zeros((n_bond_angle_bins+1),dtype=int)

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

  angle_bin_counts.append(bond_angle_bin_counts)

  bond_angle_bins = np.array([mean(np.array([float(bond_angle*(180.0/3.14)) for bond_angle in bond_angle_range])) for bond_angle_range in bond_angle_ranges])
  angle_bins.append(bond_angle_bins)

plot_distribution(angle_bins,angle_bin_counts,plot_type="Angles",multiple=True,legend=bond_angle_force_constant_list,legend_title='$k_{bond_angle}$')
exit()

