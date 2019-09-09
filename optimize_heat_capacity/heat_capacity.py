import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.thermo.calc import *
from foldamers.parameters.optimize import *
from foldamers.parameters.reweight import *
from cg_openmm.simulation.rep_exch import *

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 2.0 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas = 20
min_temp = 50.0 * unit.kelvin
max_temp = 300.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
print("Using "+str(len(temperature_list))+" replicas.")
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

cgmodel = CGModel()

if not os.path.exists(output_data):
  replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  make_replica_pdb_files(cgmodel.topology,replica_positions)
else:
  replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

C_v,dC_v,new_temperature_list = get_heat_capacity(replica_energies,temperature_list,num_intermediate_states=1)

calculate_C_v_fitness(C_v,new_temperature_list)

exit()
