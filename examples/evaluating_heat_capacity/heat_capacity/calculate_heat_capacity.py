import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.thermo.calc import *

# Replica exchange simulation settings.
# Temps are extracted from the .nc file.

# number_replicas = 12
# min_temp = 50 * unit.kelvin
# max_temp = 600.0 * unit.kelvin

output_directory = "output"
output_data = os.path.join(output_directory, "output.nc")

# Read the simulation coordinates for individual temperature replicas                                                                     
C_v, dC_v, new_temperature_list = get_heat_capacity(
    output_data=output_data,
    num_intermediate_states=3,
    plot_file='heat_capacity.pdf'
)

print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0].unit})")
for i, C in enumerate(C_v):
    print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[i]._value:>10.4f}")

# Fit curve to spline and evaluate 1st, 2nd derivatives    
derC_v, der2C_v, spline_tck = get_heat_capacity_derivative(
    C_v,
    new_temperature_list,
    plotfile='dCv_dT.pdf'
)
