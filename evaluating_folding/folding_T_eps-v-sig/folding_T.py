import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import * 
from foldamers.thermo.calc import *
from cg_openmm.simulation.rep_exch import *

grid_size = 10

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.nanosecond # Units = picoseconds
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

bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}

C_v_list = []
dC_v_list = []
folding_T_list = []

sigma_list = [ (2.5 + i*0.1) * bond_length for i in range(grid_size)]
epsilon_list = [ unit.Quantity((0.05 + i*0.025),unit.kilocalorie_per_mole) for i in range(grid_size)]
for sigma in sigma_list:
 for epsilon in epsilon_list:
  print("Sigma = "+str(sigma))
  print("Epsilon = "+str(epsilon))
  output_data = str(str(top_directory)+'/eps_'+str(round(epsilon._value,1))+'_sig_'+str(round(sigma._value,1))+'.nc')
  sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma}
  epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon}
  cgmodel = CGModel(sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths)
  if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
     minimum_energy_structure = get_minimum_energy_pose(cgmodel.topology,replica_energies,replica_positions)
  else:
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  C_v,dC_v,new_temperature_list = get_heat_capacity(replica_energies,temperature_list,num_intermediate_states=1)
  C_v_list.append(C_v)
  dC_v_list.append(dC_v)
  folding_T_list.append(new_temperature_list[np.argmax(np.array([float(i) for i in C_v]))])

file_name = str(str(top_directory)+"/folding_T_eps_sig.png")
figure = pyplot.figure(1)
temperature_list = np.array([temperature for temperature in new_temperature_list])
x = np.unique([sigma.in_units_of(unit.nanometer)._value for sigma in sigma_list])
y = np.unique([epsilon.in_units_of(unit.kilocalorie_per_mole)._value for epsilon in epsilon_list])
X,Y = np.meshgrid(x,y)
Z = folding_T_list.reshape(len(y),len(x))

pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Folding Temperature ( Kelvin )")
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

exit()
