import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.thermo.calc import *
from foldamers.parameters.reweight import get_temperature_list

# Replica exchange simulation settings.
#These must match the simulations that are being analyzed.
number_replicas = 12
min_temp = 50 * unit.kelvin
max_temp = 600.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

# Read the simulation coordinates for individual temperature replicas                                                                     
C_v, dC_v, new_temperature_list = get_heat_capacity(temperature_list, output_data="output.nc",
                                                    output_directory="output", num_intermediate_states=3,plot_file='heat_capacity.pdf')
print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0].unit})")
for i, C in enumerate(C_v):
    print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[i]._value:>10.4f}")
