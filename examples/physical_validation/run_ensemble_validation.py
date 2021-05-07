import os
from cg_openmm.simulation.physical_validation import *

# This example shows how to run an ensemble check on replica exchange simulation output
# using the physical validation package.

# Job settings
output_directory = "../run_replica_exchange/output"
output_data = os.path.join(output_directory, "output.nc")

# Run ensemble check on all adjacent temperature pairs:
quantiles = physical_validation_ensemble(
    output_data=output_data,
    output_directory="output",
    plotfile='ensemble_check',
    pairs='adjacent',
    ref_state_index=0
)

