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

native_structure_file = str(str(os.getcwd().split('examples/native_contacts')[0])+"ensembles/12_1_1_0/helix.pdb")

native_contacts_file = "F_Q_T.dat"

native_structure = PDBFile(native_structure_file).getPositions()

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 2.0 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas = 100
min_temp = 10.0 * unit.kelvin
max_temp = 200.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
print("Using "+str(len(temperature_list))+" replicas.")

Q_list = []
F_list = []
T_list = []

if not os.path.exists(native_contacts_file):

 cgmodel = CGModel()

 output_data = str(str(top_directory)+"/output.nc")
 if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
 else:
     print("Reading simulation data.")
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)


 if not os.path.exists("T_Q.dat"):
  print("Building an array of the fraction of native contacts for each pose.")
  Q_kn = np.zeros((len(replica_positions),len(replica_positions[0])))
  for replica_index in range(len(replica_positions)):
     trajectory = replica_positions[replica_index]
     for pose_index in range(len(trajectory)):
       pose = trajectory[pose_index]
       Q_kn[replica_index][pose_index] = fraction_native_contacts(cgmodel,pose,native_structure)

  data_file = open("T_Q.dat","w")
  data_file.write("Temperature (Kelvin),Fraction of Native Contacts\n")
  for index in range(len(T_list)):
   data_file.write(str(T_list[index])+","+str(Q_list[index])+"\n")
  data_file.close()
 else:
  with open("T_Q.dat",newline='') as csvfile:
          reader = csv.reader(csvfile, delimiter=',')
          next(reader)
          for row in reader:
           T_list.append(row[0])
           Q_list.append(row[1])


 print("Reweighting results with MBAR.")
 mbar,E_kn,result,dresult,new_temp_list = get_mbar_expectation(replica_energies,temperature_list,0)

 print("Calculating free energy differences with MBAR.")
 F_expect,dF_expect = get_free_energy_differences(mbar)
 print("Calculating the expecation value for Q with MBAR.") 
 Q_expect,dQ_expect = mbar.computeExpectations(Q_kn)

 for k in range(len(F_expect)):
    for l in range(len(F_expect[k])):
      F_list.append(F_expect[k][l])
      Q_list.append(Q_expect[k][l])
      T_list.append(temperature_list[k]._value)

 data_file = open("F_Q_T.dat","w")
 data_file.write("Temperature (Kelvin),Fraction of Native Contacts,Free Energy (Dimensionless)\n")
 for index in range(len(T_list)):
  data_file.write(str(T_list[index])+","+str(Q_list[index])+","+str(F_list[index])+"\n")
 data_file.close()

T_list = []
F_list = []
Q_list = []
with open("F_Q_T.dat",newline='') as csvfile:
          reader = csv.reader(csvfile, delimiter=',')
          next(reader)
          for row in reader:
           T_list.append(row[0])
           Q_list.append(row[1])
           F_list.append(row[2])

file_name = "F_v_Q_v_T.png"
figure = pyplot.figure(1)
temperatures = np.array([temperature._value for temperature in temperature_list])

T_ranges = []
for T_index in range(1,len(temperatures)-1):
 this_T_step = temperatures[T_index] - temperatures[T_index-1]
 next_T_step = temperatures[T_index+1] - temperatures[T_index]
 this_middle = temperatures[T_index-1] + this_T_step/2.0
 next_middle = temperatures[T_index] + next_T_step/2.0
 T_ranges.append([this_middle,next_middle])

n_bins = 50
max_Q = 1.0
min_Q = float(Q_list[np.argmin(Q_list)])
min_T = temperatures[0]
max_T = temperatures[1]
Q_step = (max_Q-min_Q)/(n_bins+1)
Q_ranges = [[min_Q+Q_step*i,min_Q+Q_step*(i+1)] for i in range(n_bins)]

bins = [[[] for j in range(len(T_ranges))] for i in range(len(Q_ranges))]
bin_counts = np.zeros((len(Q_ranges),len(T_ranges)),dtype=int)

print("Binning samples.")
for index in range(len(F_list)):
  Q_index = None
  T_index = None
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
  if Q_index != None and T_index != None:
    #print("q = "+str(Q_list[index]))
    #print("T = "+str(T_list[index]))
    #print("T index = "+str(T_index))
    #print("Q range = "+str(Q_ranges[Q_index]))
    #print("T range = "+str(T_ranges[T_index]))
    bins[Q_index][T_index].append(F_list[index])
  if Q_index == 1:
    print("Houston")
    exit()

total_samples = 0
for Q_index in range(len(Q_ranges)):
 for T_index in range(len(T_ranges)):
  bin_counts[Q_index][T_index] = len(bins[Q_index][T_index])
  total_samples = total_samples + bin_counts[Q_index][T_index]
x = np.array([mean(np.array([float(T) for T in T_range])) for T_range in T_ranges])
y = np.array([mean(np.array([float(Q) for Q in Q_range])) for Q_range in Q_ranges])
X,Y = np.meshgrid(x,y)
Z = bin_counts

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("Fraction of Native Contacts")
pyplot.title("Bin Distributions ("+str(total_samples)+" total samples)")

pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig("bin_distributions.png")
pyplot.show()
pyplot.close()

ensemble_avg_F = np.zeros((n_bins+1,n_bins+1))
for i in range(n_bins+1):
 for j in range(n_bins+1):
  #print(bins[i][j])
  if len(bins[i][j]) > 1:
    ensemble_avg_F[i][j] = mean(np.array([float(F) for F in bins[i][j]]))
  else:
    print(i)
    print(j)
    print(bins[i][j])
    ensemble_avg_F[i][j] = float(bins[i][j][0])

y = np.array([mean(np.array([Q_range])) for Q_range in Q_ranges])
x = np.array([mean(np.array([T_range])) for T_range in T_ranges])
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
