#!~/anaconda3/bin/python

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
# using a "CGModel" object built with the cg_openmm software package.
#
###

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
output_data = os.path.join(output_directory, "output.nc")

# Replica exchange simulation settings
cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))
replica_energies, replica_positions, replica_states, production_start, decorrelation_spacing = \
    process_replica_exchange_data(
        output_data=output_data,
        output_directory="output",
    )

# Write trajectory files:
# make_replica_pdb_files(cgmodel.topology, replica_positions, output_dir=output_directory)
# make_state_pdb_files(cgmodel.topology, replica_positions, replica_states, output_dir=output_directory)


