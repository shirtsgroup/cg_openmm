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

grid_size = 4

native_structure_file = str(str(os.getcwd().split('examples/')[0])+"ensembles/12_1_1_0/helix.pdb")

native_structure = PDBFile(native_structure_file).getPositions()

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 5.0 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas = 30
min_temp = 10.0 * unit.kelvin
max_temp = 200.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
print("Using "+str(len(temperature_list))+" replicas.")

cgmodel = CGModel()
model_torsion_list = cgmodel.torsion_list

torsion_list = []
for torsion in model_torsion_list:
  if all([cgmodel.get_particle_type(i) == "backbone" for i in torsion]):
    torsion_list.append(torsion)

torsions_list = []
bin_counts_list = []

torsion_force_constant_list = [ 0.001*10**i for i in range(grid_size)]
for constant in torsion_force_constant_list:
  torsion_force_constants={'bb_bb_bb_bb_torsion_k': constant}
  cgmodel = CGModel(torsion_force_constants=torsion_force_constants)

  output_data = str(str(top_directory)+"/"+str(round(constant,3))+".nc")
  if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  else:
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  model_angle_list = cgmodel.torsion_list
  torsions = []
  make_replica_pdb_files(cgmodel.topology,replica_positions)
  for replica_index in range(len(replica_positions)):
     trajectory = md.load(str("replica_"+str(replica_index+1)+".pdb"))
     for torsion in torsion_list:
      traj_angles = md.compute_dihedrals(trajectory,[torsion])
      for sample in traj_angles:
       torsions.append(sample)

  torsions = np.array([float(angle) for angle in torsions])

  n_torsion_bins = 100
  torsion_bin_counts = np.zeros((n_torsion_bins+1),dtype=int)
  min_torsion = torsions[np.argmin(torsions)]
  max_torsion = torsions[np.argmax(torsions)]
  torsion_step = (max_torsion-min_torsion)/(n_torsion_bins+1)
  torsion_ranges = [[min_torsion+torsion_step*i,min_torsion+torsion_step*(i+1)] for i in range(n_torsion_bins+1)]

  for index in range(len(torsions)):
   for torsion_range_index in range(len(torsion_ranges)):
    torsion_range = torsion_ranges[torsion_range_index]
    if torsion_range_index == 0:
     if float(torsions[index]) >= float(torsion_range[0]) and float(torsions[index]) <= float(torsion_range[1]):
      torsion_bin_index = torsion_range_index
    else:
     if float(torsions[index]) > float(torsion_range[0]) and float(torsions[index]) <= float(torsion_range[1]):
      torsion_bin_index = torsion_range_index
   torsion_bin_counts[torsion_bin_index] = torsion_bin_counts[torsion_bin_index] + 1

  bin_counts_list.append(torsion_bin_counts)

x = [np.array([mean(np.array([float(torsion*(180.0/3.14)) for torsion in torsion_range])) for torsion_range in torsion_ranges]) for i in range(len(torsion_force_constant_list))]
plot_distribution(x,bin_counts_list,plot_type="Torsions",legend=[str(round(constant,3)) for constant in torsion_force_constant_list],multiple=True,legend_title=str(r"$k_{\alpha}$"),plot_title=str(r"$\alpha_{BBBB}$ distribution for variable $k_{\alpha}$ (unitless)"))

exit()

