import os
import csv
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from statistics import mean
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.parameters.secondary_structure import *
from foldamers.thermo.calc import *
from foldamers.utilities.util import random_positions
from cg_openmm.build.cg_build import build_topology
from cg_openmm.simulation.rep_exch import *

grid_size = 1

native_structure_file = str(str(os.getcwd().split('examples/rmsd_from_helical')[0])+"ensembles/12_1_1_0/helix.pdb")

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

RMSD_list = []
F_list = []
T_list = []

if not os.path.exists("F_RMSD_T.dat"):

 epsilon_list = [ unit.Quantity((0.05 + i*0.25),unit.kilocalorie_per_mole) for i in range(grid_size)]
 for epsilon in epsilon_list:
  print("Calculating the free energy as a function of RMSD to the idyllic helical structure.")
  print("for a coarse grained model with epsilon values of "+str(epsilon))
  epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon}
  cgmodel = CGModel(epsilons=epsilons)

  output_data = str(str(top_directory)+"/eps_"+str(epsilon._value)+".nc")
  if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  else:
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  rmsd_to_helical = np.zeros((len(replica_positions),len(replica_positions[0])))
  make_replica_pdb_files(cgmodel.topology,replica_positions)
  native_structure = md.load(str(str(os.getcwd().split('examples/rmsd_from_helical')[0])+"ensembles/12_1_1_0/helix.pdb"))
  for replica_index in range(len(replica_positions)):
     trajectory = md.load(str("replica_"+str(replica_index+1)+".pdb"))
     rmsds = md.rmsd(trajectory,native_structure)
     rmsd_to_helical[replica_index] = rmsds

  mbar,E_kn,result,dresult,new_temp_list = get_mbar_expectation(replica_energies,temperature_list,0)

  F_expect,dF_expect = get_free_energy_differences(mbar)  
#  Q_expect,dQ_expect = mbar.computeExpectations(Q_kn)

  for k in range(len(F_expect)):
    for l in range(len(F_expect[k])):
      F_list.append(F_expect[k][l])
      RMSD_list.append(rmsd_to_helical[k][l])
      T_list.append(temperature_list[k]._value)

 data_file = open("F_RMSD_T.dat","w")
 data_file.write("Temperature (Kelvin),RMSD to Native (Angstroms),Free Energy (Dimensionless)\n")
 for index in range(len(T_list)):
  data_file.write(str(T_list[index])+","+str(RMSD_list[index])+","+str(F_list[index])+"\n")
 data_file.close()
else:
  T_list = []
  F_list = []
  RMSD_list = []
  with open("F_RMSD_T.dat",newline='') as csvfile:
          reader = csv.reader(csvfile, delimiter=',')
          next(reader)
          for row in reader:
           T_list.append(row[0])
           RMSD_list.append(row[1])
           F_list.append(row[2])

file_name = "F_v_RMSD_v_T.png"
figure = pyplot.figure(1)
temperatures = np.array([temperature for temperature in temperature_list])

x = np.array([float(T) for T in T_list])
y = np.array([float(R) for R in RMSD_list])
z = np.array([float(F) for F in F_list])

n_bins = 20
max_Q = 1.0
min_Q = y[np.argmin(y)]
min_T = x[np.argmin(x)]
max_T = x[np.argmax(x)]
Q_step = (max_Q-min_Q)/(n_bins+1)
T_step = (max_T-min_T)/(n_bins+1)
Q_ranges = [[min_Q+Q_step*i,min_Q+Q_step*(i+1)] for i in range(n_bins+1)]
T_ranges = [[min_T+T_step*i,min_T+T_step*(i+1)] for i in range(n_bins+1)]
bins = [[[] for j in range(n_bins+1)] for i in range(n_bins+1)]

bin_counts = np.zeros((n_bins+1,n_bins+1),dtype=int)

Q_list = RMSD_list

for index in range(len(z)):
  for Q_range_index in range(len(Q_ranges)):
   Q_range = Q_ranges[Q_range_index]
   if Q_range_index == 0:
     if float(Q_list[index]) >= float(Q_range[0]) and float(Q_list[index]) <= float(Q_range[1]):
      Q_index = Q_range_index
   else:
     if float(Q_list[index]) > float(Q_range[0]) and float(Q_list[index]) <= float(Q_range[1]):
      Q_index = Q_range_index
  for T_range_index in range(len(T_ranges)):
   T_range = T_ranges[T_range_index]
   if T_range_index == 0:
     if float(T_list[index]) >= float(T_range[0]) and float(T_list[index]) <= float(T_range[1]):
      T_index = T_range_index
   else:
     if float(T_list[index]) > float(T_range[0]) and float(T_list[index]) <= float(T_range[1]):
      T_index = T_range_index

  bins[T_index][Q_index].append(F_list[index])

total_samples = 0
for Q_index in range(len(Q_ranges)):
 for T_index in range(len(T_ranges)):
  bin_counts[T_index][Q_index] = len(bins[T_index][Q_index])
  total_samples = total_samples + bin_counts[T_index][Q_index]
y = np.array([mean(np.array([float(Q) for Q in Q_range])) for Q_range in Q_ranges])
x = np.array([mean(np.array([float(T) for T in T_range])) for T_range in T_ranges])
X,Y = np.meshgrid(x,y)
Z = bin_counts.reshape(len(y),len(x))

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("RMSD to Native ( Angstroms )")
pyplot.title("Bin Distributions ("+str(total_samples)+" total samples)")

pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig("bin_distributions.png")
pyplot.show()
pyplot.close()
exit()


ensemble_avg_F = np.zeros((n_bins+1,n_bins+1))
for i in range(n_bins+1):
 for j in range(n_bins+1):
  print(bins[i][j])
  #if len(bins[i][j]) > 1:
    #ensemble_avg_F[i][j] = mean(np.array([float(F) for F in bins[i][j]]))
  #else:
    #print(i)
    #print(j)
    #print(bins[i][j])
    #ensemble_avg_F[i][j] = float(bins[i][j][0])
exit()

x = np.array([mean(np.array([Q_range])) for Q_range in Q_ranges])
y = np.array([mean(np.array([T_range])) for T_range in T_ranges])
X,Y = np.meshgrid(x,y)
Z = ensemble_avg_F.reshape(len(y),len(x))

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("Fraction of Native Contacts")
pyplot.title("Free Energy vs. Q")

pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig("native_contacts.png")
pyplot.show()
pyplot.close()

exit()
