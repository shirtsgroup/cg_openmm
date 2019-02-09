#!/bin/bash

#
# p-xylene binding to T4 lysozyme L99A example run script (serial mode)
#

# Run the simulation
echo "Running simulation..."
yank script --yaml=CG_octamer.yaml

# Analyze the data
echo "Analyzing data..."
yank analyze --store=CG_octamer-output
