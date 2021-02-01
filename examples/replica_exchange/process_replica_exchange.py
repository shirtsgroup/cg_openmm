#!/Users/mrshirts/anaconda3/bin/python

import os
import pdb
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.simulation.rep_exch import *
import numpy as np
import pickle

###
#
# This example demonstrates how to run a OpenMM replica exchange simulation
# using a "CGModel" object built with the 'foldamers' software package.
#
###

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
output_data = os.path.join(output_directory, "output.nc")

# Replica exchange simulation settings
cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))
replica_energies, replica_states, production_start, decorrelation_spacing, n_transit, mixing_stats = process_replica_exchange_data(
    output_data=output_data,
    output_directory="output",
)

print(production_start)
print(decorrelation_spacing)

# make_replica_dcd_files(
    # cgmodel.topology,
    # timestep=10*unit.femtosecond,
    # time_interval=100,
    # output_dir=output_directory,
# )
# make_state_dcd_files(
    # cgmodel.topology,
    # timestep=10*unit.femtosecond,
    # time_interval=100,
    # output_dir=output_directory,
# )

make_replica_pdb_files(
    cgmodel.topology,
    output_dir=output_directory,
)
make_state_pdb_files(
    cgmodel.topology,
    output_dir=output_directory,
)