#!/usr/bin/python

import os, timeit
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.util import random_positions
from cg_openmm.src.simulation.tools import build_mm_simulation, run_simulation
from cg_openmm.src.utilities.iotools import read_mm_energies

# OpenMM simulation settings
temperature = 300.0 * unit.kelvin
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.picosecond
simulation_time_step = 5.0 * unit.femtosecond

# Polymer properties

top_directory = 'generate_ensembles'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

for polymer_length in [1,10,20,30,50,100]:
 print("Building models with 'polymer_length' ="+str(polymer_length))
 for backbone_lengths in [[1],[2],[3],[4],[5]]:
  print("Building models with backbone lengths ="+str(backbone_lengths))
  for sidechain_lengths in [[0],[1],[2],[3],[4],[5]]:
   print("Building models with sidechain lengths ="+str(sidechain_lengths))
   if sidechain_lengths[0] != 0:
    for sidechain_positions in [[i] for i in range(backbone_lengths[0])]:
     print("Building models with sidechain positions ="+str(sidechain_positions))
     cg_model = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,include_torsion_forces=False,use_structure_library=True)
     cg_model.simulation.step(100)
exit()
