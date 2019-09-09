import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import get_mbar_expectation, get_free_energy_differences,get_temperature_list
from foldamers.parameters.secondary_structure import get_helical_parameters
from foldamers.thermo.calc import calculate_heat_capacity
from foldamers.utilities.iotools import write_pdbfile_without_topology
from cg_openmm.build.cg_build import build_topology
from cg_openmm.simulation.rep_exch import *

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)
output_data = 'output_data.nc'

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 200.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
number_replicas = 100
min_temp = 100.0 * unit.kelvin
max_temp = 300.0 * unit.kelvin
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
polymer_length=8
cgmodel = CGModel(polymer_length=polymer_length)
  # Run a replica exchange simulation with this cgmodel
if not os.path.exists(output_data):
  replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  steps_per_stage = round(total_steps/exchange_attempts)
  plot_replica_exchange_energies(replica_energies,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)
  plot_replica_exchange_summary(replica_states,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)

if not os.path.exists('re_min.pdb'):
  get_minimum_energy_pose(cgmodel.topology,replica_energies,replica_positions)

cgmodel.positions = PDBFile('re_min.pdb').getPositions()
pitch,radius,monomers_per_turn = get_helical_parameters(cgmodel)
print(pitch)
print(radius)
print(monomers_per_turn)

exit()
