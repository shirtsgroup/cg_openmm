import os
import numpy as np
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from cg_openmm.src.simulation.rep_exch import replica_exchange
from cg_openmm.src.simulation.tools import run_simulation, build_mm_simulation
import pymbar

temperature = 300.0 * unit.kelvin
print_frequency = 1 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 1.0 * unit.femtosecond
kB = 0.008314462  #Boltzmann constant (Gas constant) in kJ/(mol*K)

cg_model = CGModel(include_torsion_forces=False)
simulation = build_mm_simulation(cg_model.topology,cg_model.system,cg_model.positions,simulation_time_step=simulation_time_step,print_frequency=1)
positions = simulation.context.getState(getPositions=True).getPositions()
potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
forces = simulation.context.getState(getForces=True).getForces()
for step in range(1000):
 simulation.step(1)
 forces = simulation.context.getState(getForces=True).getForces()
 print(forces)
#run_simulation(cg_model,os.getcwd(),total_simulation_time,simulation_time_step,temperature,print_frequency)
exit()
