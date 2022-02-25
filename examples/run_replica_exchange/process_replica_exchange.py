#!~/anaconda3/bin/python

import os
import pickle

from cg_openmm.simulation.rep_exch import *
from openmm import unit

# This example demonstrates how to post-process OpenMM replica exchange simulation energies,
# and generate individual dcd trajectories from the .nc output files.

# Replica exchange analysis data
analysis_stats = {}

# Job settings
output_directory = "output"
output_data = os.path.join(output_directory, "output.nc")

# Load in the cgmodel:
cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))

# Process replica exchange energy data:
replica_energies, replica_states, production_start, sample_spacing, n_transit, mixing_stats = process_replica_exchange_data(
    output_data=output_data,
    output_directory=output_directory,
    frame_begin=20000,
    write_data_file=False,
    print_timing=True,
)

# Starting frame of production region:
analysis_stats["production_start"] = production_start

# Decorrelation time for state energy timeseries: 
analysis_stats["energy_decorrelation"] = sample_spacing

# Number of half transitions from temperature state 0-->n or n-->0:
analysis_stats["n_transit"] = n_transit

# Replica transition matrix:
analysis_stats["transition_matrix"] = mixing_stats[0]

# Eigenvalues of transition matrix:
analysis_stats["eigenvalues"] = mixing_stats[1]

# Statistical inefficiency from transition matrix analysis:
analysis_stats["statistical_inefficiency"] = mixing_stats[2]

# Save analysis stats to pickle file:
pickle_out = open("analysis_stats_discard_20ns.pkl", "wb")
pickle.dump(analysis_stats, pickle_out)
pickle_out.close()

#--------------------#
# Write trajectories #
#--------------------#

# With constant replica (continuous):
make_replica_dcd_files(
    cgmodel.topology,
    timestep=5*unit.femtosecond,
    time_interval=200,
    output_dir=output_directory
)

# With constant temperature state (discontinuous):
make_state_dcd_files(
    cgmodel.topology,
    timestep=5*unit.femtosecond,
    time_interval=200,
    output_dir=output_directory
)    

