import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.thermo.calc import *
import pickle


# This example demonstrates how to calculate heat capacity as a function of temperature from
# replica exchange energies, with uncertainties calculated from bootstrapping.

# Note: process_replica_exchange_data should first be run to determine the determine the start
# of the production region and energy decorrelation time.


# Job settings
output_directory = '../run_replica_exchange/output'
output_data = os.path.join(output_directory, "output.nc")

# Load in trajectory stats:
analysis_stats = pickle.load(open("../run_replica_exchange/analysis_stats_discard_20ns.pkl","rb"))

# Read the simulation coordinates for individual temperature replicas                                                                     
(new_temperature_list, C_v, dC_v, Tm_value, Tm_uncertainty, 
Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty) = bootstrap_heat_capacity(
    output_data=output_data,
    frame_begin=analysis_stats["production_start"],
    sample_spacing=analysis_stats["energy_decorrelation"],
    num_intermediate_states=3,
    n_trial_boot=200,
    conf_percent='sigma',
    plot_file=f"heat_capacity_boot.pdf",
)

print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0][0].unit})")
for i, C in enumerate(C_v):
    print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[0][i]._value:>10.4f} {dC_v[1][i]._value:>10.4f}")

print(f"Tmelt (K)")
print(f"{Tm_value._value:8.2f} {Tm_uncertainty[0]._value:8.2f} {Tm_uncertainty[1]._value:8.2f}")    

print(f"Cv_height (kJ/mol/K)")
print(f"{Cv_height_value._value:8.2f} {Cv_height_uncertainty[0]._value:8.2f} {Cv_height_uncertainty[1]._value:8.2f}")   

print(f"FWHM (K)")
print(f"{FWHM_value._value:8.2f} {FWHM_uncertainty[0]._value:8.2f} {FWHM_uncertainty[1]._value:8.2f}")   
