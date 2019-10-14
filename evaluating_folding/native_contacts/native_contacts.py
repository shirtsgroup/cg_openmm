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

 print("Reweighting results with MBAR.")
 mbar,E_kn,result,dresult,new_temp_list = get_mbar_expectation(replica_energies,temperature_list,0)

 print("Calculating free energy differences with MBAR.")
 F_expect,dF_expect = get_free_energy_differences(mbar)
 print("Calculating the expecation value for Q with MBAR.") 
 Q_expect,dQ_expect = mbar.computeExpectations(Q_kn)

 for k in range(len(F_expect)):
    for l in range(len(F_expect[k])):
      F_list.append(F_expect[k][l])
      Q_list.append(Q_expect[l])
      T_list.append(temperature_list[l]._value)

 data_file = open("F_Q_T.dat","w")
 data_file.write("Temperature (Kelvin),Fraction of Native Contacts,Free Energy (Dimensionless)\n")
 for index in range(len(T_list)):
  data_file.write(str(T_list[index])+","+str(Q_list[index])+","+str(F_list[index])+"\n")
 data_file.close()

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

x = np.array([mean(np.array([float(T) for T in T_range])) for T_range in T_ranges])
y = np.array([float(Q) for Q in Q_expect])
X,Y = np.meshgrid(x,y)
Z = np.array([float(F) for F in F_list]).reshape(len(x),len(y))

pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.xlabel("T ( Kelvin )")
pyplot.ylabel("Q (Fraction native contacts)")
pyplot.title("F v Q for 1-1 12-mer homopolymer")
pyplot.savefig("native_contacts.png")
pyplot.show()
pyplot.close()
exit()
