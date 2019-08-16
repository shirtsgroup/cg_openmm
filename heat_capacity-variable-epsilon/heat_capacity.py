import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.parameters.reweight import *
from foldamers.src.thermo.calc import calculate_heat_capacity
from cg_openmm.src.build.cg_build import build_topology
from cg_openmm.src.simulation.rep_exch import *

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

number_replicas = 20
min_temp = 100.0 * unit.kelvin
max_temp = 150.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)

# Yank (replica exchange) simulation settings
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

epsilon_list = [unit.Quantity(0.5+i*0.25,unit.kilocalorie_per_mole) for i in range(10)]

for epsilon in epsilon_list:
  print("Epsilon = "+str(epsilon))
  epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon}
  cgmodel = CGModel(epsilons=epsilons)

  output_data = str(str(top_directory)+"/eps_"+str(epsilon._value)+".nc")
  if not os.path.exists(output_data):
    replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
    steps_per_stage = round(total_steps/exchange_attempts)
    plot_replica_exchange_energies(replica_energies,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)
    plot_replica_exchange_summary(replica_states,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)
  else:
    replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  num_intermediate_states = 1
  mbar,E_kn,E_expect,dE_expect,new_temp_list = get_mbar_expectation(replica_energies,temperature_list,num_intermediate_states)

  mbar,E_kn,DeltaE_expect,dDeltaE_expect,new_temp_list = get_mbar_expectation(E_kn,temperature_list,num_intermediate_states,mbar=mbar,output='differences')

  mbar,E_kn,E2_expect,dE2_expect,new_temp_list = get_mbar_expectation(E_kn**2,temperature_list,num_intermediate_states,mbar=mbar)

  df_ij,ddf_ij = get_free_energy_differences(mbar)


  C_v,dC_v = calculate_heat_capacity(E_expect,E2_expect,dE_expect,DeltaE_expect,dDeltaE_expect,df_ij,ddf_ij,new_temp_list,len(temperature_list),num_intermediate_states)
  C_v_list.append(C_v)
  dC_v_list.append(dC_v)


file_name = "heat_capacity.png"
figure = pyplot.figure(1)

legend_labels = [ str("$\epsilon$= "+str(round(epsilon._value,2))) for epsilon in epsilon_list]

for C_v,dC_v in zip(C_v_list,dC_v_list):
 C_v = np.array([C_v[i][0] for i in range(len(C_v))])
 dC_v = np.array([dC_v[i][0] for i in range(len(dC_v))])
 pyplot.errorbar(temperatures,C_v,yerr=dC_v,figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
pyplot.title("Heat capacity for variable $\sigma$")
pyplot.legend(legend_labels)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()


exit()
