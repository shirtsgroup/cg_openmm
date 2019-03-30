## No default python environment

# This script performs replica exchange simulations in Yank/OpenMM,
# followed by parameter re-weighting and structural analysis with
# pymbar and MDTraj.

# The script has been successfully tested with Python3.7

# The following external packages ( and all of their dependencies ) are required to run this script:
#
# 1) OpenMM
# 2) Yank
# 3) pymbar
# 4) OpenMMTools
# 5) MDTraj
#

# =======================================================================
#
# SCRIPT STRUCTURE
#
# =======================================================================

# 1) PYTHON PACKAGE IMPORTS
# 2) ENVIRONMENT/JOB SETTINGS
# 3) BUILD COARSE-GRAINED MODEL
# 4) SETUP AND RUN REPLICA-EXCHANGE SIMULATIONS
# 5) RE-WEIGHT REPLICA-EXCHANGE DATA WITH PYMBAR
# 6) PLOT RESULTS

# =============================================================================================
# 1) PYTHON PACKAGE IMPORTS
# =============================================================================================

# System packages
import os, sys, timeit
from io import StringIO
import numpy as np
import math, random
# Local protocol functions
from include.replica_exchange import *
from include.build_cg_model import *
from include.get_random_coordinates import *
# OpenMM utilities
from simtk import openmm as mm
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
# MBAR
from pymbar import MBAR, timeseries
# Plotting functions
import matplotlib.pyplot as pyplot

# =============================================================================================
# 2) ENVIRONMENT/JOB SETTINGS
# =============================================================================================

# Job settings
output_directory = "output"
input_directory = "input"
if not os.path.exists(output_directory): os.makedirs(output_directory)
if not os.path.exists(input_directory): os.makedirs(input_directory)
initial_coordinates_pdb = str(input_directory+"/coordinates.pdb")
initial_coordinates_xyz = str(input_directory+"/coordinates.xyz")
replica_exchange_storage_file = str(output_directory+"/replica_exchange.nc")

# OpenMM/Yank simulation settings
box_size = 10.00 # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.002 # Units = picoseconds
kB = 0.008314462  #Boltzmann constant (Gas constant) in kJ/(mol*K)
simulation_steps = 1000 # Number of steps used in individual Langevin dynamics simulations
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds
exchange_attempts = 5 # The number of times that a replica exchange will be attempted within the simulation
num_replicas = 10 # Number of discrete temperatures at which we will run simulations
t_min = 300.0 # Minimum temperature for replicas
t_max = 500.0 # Maximum temperature for replicas
#temperatures = [t_min + i * temp_increment for i in range(0,num_replicas)] * unit.kelvin # Temperatures for individual replicas
temperatures = [t_min + (t_max - t_min) * (math.exp(float(i) / float(num_replicas-1)) - 1.0) / (math.e - 1.0) for i in range(0,num_replicas)] * unit.kelvin
simulation_settings = [temperatures,simulation_time_step,simulation_steps,print_frequency,total_simulation_time,exchange_attempts,replica_exchange_storage_file,input_directory,output_directory]

# Coarse-grained model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead on which the side chains are placed
polymer_length = 8 # Number of monomers in the polymer
num_particles = (backbone_length + sidechain_length) * polymer_length
model_settings = [box_size,polymer_length,backbone_length,sidechain_length,sidechain_positions]
mass = 12.0 * unit.amu # Mass of beads
sigma = 1.0 # unit.angstroms Lennard-Jones interaction distance, and bond length
epsilon = 0.2 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
q = 0.0 * unit.elementary_charge # Charge of beads
particle_properties = [mass,q,sigma,epsilon]

start_time = timeit.default_timer()

# =============================================================================================
# 3) BUILD COARSE-GRAINED MODEL
# =============================================================================================

get_low_energy_structure(simulation_settings,model_settings,particle_properties)
exit()
#positions = assign_random_initial_coordinates(model_settings)
system,topology = build_cg_model(model_settings,particle_properties,positions)
system = assign_default_box_vectors(system,box_size)

# =============================================================================================
# 4) SETUP AND RUN REPLICA-EXCHANGE SIMULATIONS
# =============================================================================================

run_replica_exchange(system,simulation_settings,model_settings,particle_properties)
all_replica_energies = get_replica_energies(simulation_settings)

# =============================================================================================
# 5) RE-WEIGHT REPLICA-EXCHANGE DATA WITH MBAR
# =============================================================================================



# =============================================================================================
# 6) PLOT RESULTS AND SHOW MINIMUM ENERGY STRUCTURES
# =============================================================================================

figure_index = 1
figure = pyplot.figure(figure_index)
pyplot.xlabel("Simulation Time (fs)")
pyplot.ylabel("Potential Energy (kcal/mol)")
pyplot.title("Replica exchange summary")
pyplot.savefig(str(figures_directory+str("/replica_exchange_summary.png")))
pyplot.close()
figure_index = figure_index + 1

#####
#
# END
#
#####

stop_time = timeit.default_timer()
print("Calculation time was: "+str(stop_time-start_time)+" seconds.")
exit()
