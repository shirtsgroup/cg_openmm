import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.thermo.calc import *
from foldamers.utilities.util import random_positions
from cg_openmm.build.cg_build import build_topology
from cg_openmm.simulation.rep_exch import *

grid_size = 10

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)
run_simulations = True

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 0.2 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas = 30
min_temp = 50.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
print("Using "+str(len(temperature_list))+" replicas.")
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

file = open(str("Cv_data.dat"),"w")
file.write("Epsilon(kCal/mol) T(Kelvin) Cv(kCal/mol*Kelvin)\n")
file.close()

file = open(str("dCv_data.dat"),"w")
file.write("Epsilon(kCal/mol) T(Kelvin) dCv(kCal/mol*Kelvin)\n")
file.close()

C_v_list = []
dC_v_list = []
folding_T_list = []

epsilon_list = [ unit.Quantity((0.25 + i*0.25),unit.kilocalorie_per_mole) for i in range(grid_size)]
for epsilon in epsilon_list:
  print("Performing simulations and heat capacity analysis for a coarse grained model")
  print("with epsilon values of "+str(epsilon))
  epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon}
  cgmodel = CGModel(epsilons=epsilons)

  output_data = str(str(top_directory)+"/eps_"+str(epsilon._value)+".nc")
  if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  else:
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  C_v,dC_v,new_temp_list = get_heat_capacity(replica_energies,temperature_list,num_intermediate_states=1)

  C_v_list.append(C_v)
  dC_v_list.append(dC_v)
  folding_T_list.append(new_temp_list[np.argmax(C_v)])

  file = open(str("Cv_data.dat"),"a")
  for temp in range(len(new_temp_list)):
   file.write(str(round(epsilon._value,2))+" "+str(round(new_temp_list[temp],2))+" "+str(round(C_v[temp],4))+"\n")
  file.close
  file = open(str("dCv_data.dat"),"a")
  for temp in range(len(new_temp_list)):
   file.write(str(round(epsilon._value,2))+" "+str(round(new_temp_list[temp],2))+" "+str(round(dC_v[temp],4))+"\n")
  file.close

file_name = "heat_capacity.png"
figure = pyplot.figure(1)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
temperatures = np.array([temperature for temperature in new_temp_list])
legend_labels = [ str("$\epsilon$= "+str(round(epsilon._value,2))) for epsilon in epsilon_list]

for C_v,dC_v in zip(C_v_list,dC_v_list):
 C_v = np.array([C_v[i] for i in range(len(C_v))])
 dC_v = np.array([dC_v[i] for i in range(len(dC_v))])
 pyplot.errorbar(temperatures,C_v,yerr=dC_v,figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
pyplot.title("Heat capacity for variable $\epsilon$")
pyplot.legend(legend_labels)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()


exit()
