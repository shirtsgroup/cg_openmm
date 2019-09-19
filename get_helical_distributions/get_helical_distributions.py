import os
import csv
import numpy as np
import matplotlib.pyplot as pyplot
from statistics import mean
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.parameters.secondary_structure import *
from foldamers.thermo.calc import *
from cg_openmm.simulation.rep_exch import *

n_bins = 50

native_structure_file = str(str(os.getcwd().split('examples')[0])+"ensembles/12_1_1_0/helix.pdb")

native_structure = PDBFile(native_structure_file).getPositions()

# Job settings
output_directory = 'output'
if not os.path.exists(output_directory):
  os.mkdir(output_directory)
output_data=str(str(os.getcwd().split('examples')[0])+"examples/homopolymer_heat_capacity_varying_simulation_time/output/output5.0.nc")

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 15.0 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
number_replicas = 30
min_temp = 5.0 * unit.kelvin
max_temp = 100.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)

cgmodel = CGModel()

if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
else:
     print("Reading simulation data.")
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

configurations,energies = get_decorrelated_samples(replica_positions,replica_energies,temperature_list)

max_num_samples = 0
for traj in configurations:
  if len(traj) > max_num_samples:
    max_num_samples = len(traj)

pitch_kn = np.zeros((len(configurations),max_num_samples))
radius_kn = np.zeros((len(configurations),max_num_samples))
mpt_kn = np.zeros((len(configurations),max_num_samples))
residual_kn = np.zeros((len(configurations),max_num_samples))

for traj_index in range(len(configurations)):
  for pose_index in range(len(configurations[traj_index])):
    cgmodel.positions = configurations[traj_index][pose_index]
    pitch,radius,mpt,residual = get_helical_parameters(cgmodel)
    pitch_kn[traj_index][pose_index] = pitch
    radius_kn[traj_index][pose_index] = radius
    mpt_kn[traj_index][pose_index] = mpt
    residual_kn[traj_index][pose_index] = residual
    
max_pitch,min_pitch=None,None
max_radius,min_radius=None,None
max_mpt,min_mpt=None,None
max_residual,min_residual=None,None

for index in range(len(pitch_kn)):
  for pitch in pitch_kn[index]:
     if max_pitch == None: max_pitch = pitch
     if min_pitch == None: min_pitch = pitch
     if pitch > max_pitch: max_pitch = pitch
     if pitch < min_pitch: min_pitch = pitch

for radius in data['radius']:
     if max_radius == None: max_radius = radius
     if min_radius == None: min_radius = radius
     if radius > max_radius: max_radius = radius
     if radius < min_radius: min_radius = radius
   for mpt in data['monomers_per_turn']:
     if max_mpt == None: max_mpt = mpt
     if min_mpt == None: min_mpt = mpt
     if mpt > max_mpt: max_mpt = mpt
     if mpt < min_mpt: min_mpt = mpt
   for residual in data['residual']:
     if max_residual == None: max_residual = residual
     if min_residual == None: min_residual = residual
     if residual > max_residual: max_residual = residual
     if residual < min_residual: min_residual = residual

pitch_step_size = (max_pitch-min_pitch)/n_bins
radius_step_size = (max_radius-min_radius)/n_bins
mpt_step_size = (max_mpt-min_mpt)/n_bins
residual_step_size = (max_residual-min_residual)/n_bins
pitch_bins = np.array([[min_pitch+i*pitch_step_size,min_pitch+(i+1)*pitch_step_size] for i in range(n_bins)])
radius_bins = np.array([[min_radius+i*radius_step_size,min_radius+(i+1)*radius_step_size] for i in range(n_bins)])
mpt_bins = np.array([[min_mpt+i*mpt_step_size,min_mpt+(i+1)*mpt_step_size] for i in range(n_bins)])
residual_bins = np.array([[min_residual+i*residual_step_size,min_residual+(i+1)*residual_step_size] for i in range(n_bins)])
pitch_bin_counts = np.zeros((len(temperature_list),len(pitch_bins)),dtype=int)
radius_bin_counts = np.zeros((len(temperature_list),len(radius_bins)),dtype=int)
mpt_bin_counts = np.zeros((len(temperature_list),len(mpt_bins)),dtype=int)
residual_bin_counts = np.zeros((len(temperature_list),len(residual_bins)),dtype=int)

for data_index in range(len(pitch_kn)):
     data = pitch_kn[data_index]
     for pitch in data:
       for bin_index in range(len(pitch_bins)):
         bin = pitch_bins[bin_index]
         if pitch > bin[0] and pitch <= bin[1]:
           pitch_bin_counts[data_index][bin_index] = pitch_bin_counts[data_index][bin_index] + 1

     data = radius_kn[data_index]
     for radius in data:
       for bin_index in range(len(radius_bins)):
         bin = radius_bins[bin_index]
         if radius > bin[0] and radius <= bin[1]:
           radius_bin_counts[data_index][bin_index] = radius_bin_counts[data_index][bin_index] + 1

     data = mpt_kn[data_index]
     for mpt in data:
       for bin_index in range(len(mpt_bins)):
         bin = mpt_bins[bin_index]
         if mpt > bin[0] and mpt <= bin[1]:
           mpt_bin_counts[data_index][bin_index] = mpt_bin_counts[data_index][bin_index] + 1

     data = residual_kn[data_index]
     for residual in data:
       for bin_index in range(len(residual_bins)):
         bin = residual_bins[bin_index]
         if residual > bin[0] and residual < bin[1]:
           residual_bin_counts[data_index][bin_index] = residual_bin_counts[data_index][bin_index] + 1

pitch_bins = np.array([mean([min_pitch+i*pitch_step_size,min_pitch+(i+1)*pitch_step_size]) for i in range(n_bins)])
radius_bins = np.array([mean([min_radius+i*radius_step_size,min_radius+(i+1)*radius_step_size]) for i in range(n_bins)])
mpt_bins = np.array([mean([min_mpt+i*mpt_step_size,min_mpt+(i+1)*mpt_step_size]) for i in range(n_bins)])
residual_bins = np.array([mean([min_residual+i*residual_step_size,min_residual+(i+1)*residual_step_size]) for i in range(n_bins)])

figure = pyplot.figure(0)
for T_index in range(len(pitch_kn)):
  num_samples = len(configurations[T_index])
  pyplot.hist(pitch_kn[T_index][:num_samples],pitch_bins)

pyplot.xlabel("Pitch (Angstroms)")
pyplot.ylabel("Counts")
pyplot.title("Pitch distribution for 1-1-homo-12-mer")
pyplot.legend([round(temperature._value,1) for temperature in temperature_list[:9]],loc='center left', bbox_to_anchor=(1, 0.5),title='T (K)')
pyplot.savefig("pitch.png")
pyplot.show()

pyplot.close()
