import os
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import get_temperature_list
from cg_openmm.simulation.physical_validation import *
from cg_openmm.simulation.rep_exch import *
import numpy as np
import pickle

# This example shows how to run an ensemble check on replica exchange simulation output
# using the physical validation package.

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
output_data = os.path.join(output_directory, "output.nc")

# Get replica exchange temperature list
ft = open("stored_topology.pkl", "rb")
(temperature_list, time_interval, stored_topology) = pickle.load(ft)

quantiles = physical_validation_ensemble(
    temperature_list=temperature_list,
    ref_state_index=0,
    output_data=output_data,
    output_directory="output",
    plotfile='ensemble_check'
)

print(quantiles)