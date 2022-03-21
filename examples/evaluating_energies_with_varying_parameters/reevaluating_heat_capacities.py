#!~/anaconda3/bin/python

import os
import pickle

from cg_openmm.parameters.evaluate_energy import *
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.thermo.calc import *
from openmm import unit

# This example demonstrates how to use configurational sampling in combination with MBAR reweighting
# to evaluate heat capacity at a new set of force field parameters.

# Job settings
output_directory = '../run_replica_exchange/output'
output_data = os.path.join(output_directory, "output.nc")

# Load in trajectory stats:
analysis_stats = pickle.load(open("../run_replica_exchange/analysis_stats_discard_20ns.pkl","rb"))

# Load in CGModel
cgmodel = pickle.load(open("../run_replica_exchange/stored_cgmodel.pkl","rb"))

# Get temperature list (logarithmic spacing) corresponding to trajectory file list:
n_replicas = 12
min_temp = 200.0 * unit.kelvin
max_temp = 600.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, n_replicas)

# Create list of trajectory files to use for evaluating energies
rep_traj_file_list = []

for i in range(n_replicas):
    rep_traj_file_list.append(f'{output_directory}/replica_{i+1}.dcd')
    
# Specify force field parameters to change
param_dict = {}
param_dict['sc_sigma'] = 2.5 * unit.angstrom # Was 3.5

frame_begin = 20000 #analysis_stats["production_start"]
frame_stride = 10 # To get the best mean values for Cv curve
                  # This is not valid for getting the uncertainty right
                   
# Reevaluate energies at new force field parameters:
U_eval, simulation = eval_energy(
    cgmodel,
    rep_traj_file_list,
    temperature_list,
    param_dict,
    frame_begin=frame_begin,
    frame_stride=frame_stride,
)

# U_eval is a 3d energy array organized by replica/evaluated_state/frame

# Now, calculate the heat capacity for the new force field parameters:
C_v, dC_v, new_temperature_list = get_heat_capacity_reeval(
    U_eval,
    temperature_list,
    frame_begin=frame_begin,
    sample_spacing=frame_stride,
    num_intermediate_states=3,
    plot_file="heat_capacity_sc_sigma_2_5.pdf",
)
