from foldamers.src.cg_model.cgmodel import CGModel
from cg_openmm.src.simulation.tools import run_simulation
import os
from simtk import unit

top_directory = os.getcwd()
total_simulation_time = 0.1 * unit.nanosecond
simulation_time_step = 5.0 * unit.femtosecond
cgmodel = CGModel()
temperature = 300.0 * unit.kelvin
print_frequency = 20
run_simulation(cgmodel,top_directory,total_simulation_time,simulation_time_step,temperature,print_frequency)
