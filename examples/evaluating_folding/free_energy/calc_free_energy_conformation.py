# This example script demonstrates the calculation of free energy differences
# between designated conformational states from a replica exchange simulation.

import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.parameters.free_energy import *
from cg_openmm.parameters.reweight import *

output_data = "output.nc"
output_directory = "../native_contacts/output"
number_replicas = 12
min_temp = 50.0 * unit.kelvin
max_temp = 600.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

# Add some function to classify conformational states.
# For now randomly assign state 0,1, or 2
array_folded_states = np.random.randint(3,size=(6012,1))

num_intermediate_states = 1

full_T_list, deltaF_values, deltaF_uncertainty = expectations_free_energy(
    array_folded_states,
    temperature_list,
    output_directory,
    output_data,
    num_intermediate_states)
    
plot_free_energy_results(full_T_list, deltaF_values, deltaF_uncertainty,plotfile="free_energy2")