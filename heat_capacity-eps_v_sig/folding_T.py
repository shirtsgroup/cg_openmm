import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.parameters.reweight import get_mbar_expectation, get_free_energy_differences, get_temperature_list
from foldamers.src.thermo.calc import calculate_heat_capacity
from cg_openmm.src.build.cg_build import build_topology
from cg_openmm.src.simulation.rep_exch import *

grid_size = 10

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
number_replicas = 20
min_temp = 100.0 * unit.kelvin
max_temp = 200.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

###
#
# Coarse grained model settings
#
###

# Global definitions
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer

C_v_list = []
dC_v_list = []

sigma_list = [ (1.5 + i*0.1) * bond_length for i in range(grid_size)]
epsilon_list = [ unit.Quantity((0.5 + i*0.1),unit.kilocalorie_per_mole) for i in range(grid_size)]
for sigma in sigma_list:
 for epsilon in epsilon_list:
  print("Sigma = "+str(sigma))
  print("Epsilon = "+str(epsilon))
  output_data = str(str(top_directory)+'/eps_'+str(round(epsilon._value,1))+'_sig_'+str(round(sigma._value,1))+'.nc')
  sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma}
  epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon}
  cgmodel = CGModel(sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths)
  if not os.path.exists(output_data):
   success = False
   while not success:
    try:
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
     success = True
    except:
     os.remove(output_data)
  else:
    replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  steps_per_stage = round(total_steps/exchange_attempts)

  minimum_energy_structure = get_minimum_energy_pose(cgmodel.topology,replica_energies,replica_positions)

  plot_replica_exchange_energies(replica_energies,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)

  plot_replica_exchange_summary(replica_states,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)

  num_intermediate_states = 1
  mbar,E_kn,E_expect,dE_expect,new_temp_list = get_mbar_expectation(replica_energies,temperature_list,num_intermediate_states)

  mbar,E_kn,DeltaE_expect,dDeltaE_expect,new_temp_list = get_mbar_expectation(E_kn,temperature_list,num_intermediate_states,mbar=mbar,output='differences')

  mbar,E_kn,E2_expect,dE2_expect,new_temp_list = get_mbar_expectation(E_kn**2,temperature_list,num_intermediate_states,mbar=mbar)

  df_ij,ddf_ij = get_free_energy_differences(mbar)

  C_v,dC_v = calculate_heat_capacity(E_expect,E2_expect,dE_expect,DeltaE_expect,dDeltaE_expect,df_ij,ddf_ij,new_temp_list,len(temperature_list),num_intermediate_states)

  C_v_list.append(C_v)

file_name = str(str(top_directory)+"/heat_capacity.png")
figure = pyplot.figure(1)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
try:
  temperatures = np.array([temperature._value for temperature in new_temp_list])
except:
  temperatures = np.array([temperature for temperature in new_temp_list])

folding_temperature = []
for C_v in C_v_list:
 C_v = np.array([C_v[i][0] for i in range(len(C_v))])
 folding_temperature.append(max(C_v))

folding_temperature = np.array([temp for temp in folding_temperature])
x = np.unique([sigma._value for sigma in sigma_list])
y = np.unique([epsilon._value for epsilon in epsilon_list])
X,Y = np.meshgrid(x,y)
Z = folding_temperature.reshape(len(y),len(x))

pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Folding Temperature ( Kelvin )")
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

exit()
