#!/Users/mrshirts/anaconda3/bin/python

import os
import pdb
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
import numpy as np

###
#
# This example demonstrates how to run a Yank replica exchange simulation
# using a "CGModel" object built with the 'foldamers' software package.
#
###

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
output_data = os.path.join(output_directory, "output.nc")

# Replica exchange simulation settings
print_frequency = 10  # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.nanosecond
simulation_time_step = 2.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))
output_data = os.path.join(output_directory, "output.nc")
number_replicas = 12
min_temp = 600.0 * unit.kelvin
max_temp = 1000.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
exchange_attempts = 1000
replica_energies, replica_positions, replica_states = process_replica_exchange_data(
    temperature_list=temperature_list,
    output_data=output_data,
    output_directory="output",
    print_frequency=print_frequency,
    time_interval = 2.0*unit.picoseconds
    )
import pdb
pdb.set_trace()
