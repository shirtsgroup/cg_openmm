import os
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.simulation.physical_validation import *
import numpy as np

# This example shows how to run an ensemble check on replica exchange simulation output
# using the physical validation package.

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
output_data = os.path.join(output_directory, "output.nc")

# Temperature list is now determined from the .nc file internally.

quantiles = physical_validation_ensemble(
    output_data=output_data,
    output_directory="output",
    plotfile='ensemble_check',
    pairs='adjacent',
    ref_state_index=0
)

