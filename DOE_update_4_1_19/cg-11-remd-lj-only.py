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
# 2) ENVIRONMENT SETTINGS
# 3) BUILD COARSE-GRAINED MODEL
# 4) SETUP AND RUN REPLICA-EXCHANGE SIMULATIONS
# 5) ANALYZE REPLICA-EXCHANGE DATA WITH MDTRAJ
# 6) RE-WEIGHT REPLICA-EXCHANGE DATA WITH PYMBAR
# 7) PLOT RESULTS AND SHOW MINIMUM ENERGY STRUCTURES

# =============================================================================================
# 1) PYTHON PACKAGE IMPORTS
# =============================================================================================

# System packages
import os, sys, timeit, socket
from io import StringIO
import numpy as np
import math, random
# OpenMM packages
from simtk import openmm as mm
from simtk.openmm.app import *
from simtk.openmm import LangevinIntegrator
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
# Yank (replica exchange) packages
from yank import mpi, analyze
from yank.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler, ParallelTemperingSampler, SAMSSampler
from yank.multistate import ReplicaExchangeAnalyzer, SAMSAnalyzer
from yank.multistate.multistatereporter import _DictYamlLoader
from yank.utils import config_root_logger
# OpenMMTools
import openmmtools as mmtools
from openmmtools import testsystems
# MDTraj
import mdtraj as md

# =============================================================================================
# 2) ENVIRONMENT/JOB SETTINGS
# =============================================================================================

# Environment settings
if socket.gethostname() == "Louie":
 pdb_file="/mnt/d/Foldamers/OpenMM_CG_polymers/structure_files/CG_8-mer.pdb"

# Job settings

# OpenMM/Yank simulation settings
simulation_time_step = 0.002 # Units = picoseconds
kB = 0.0019872041 #Boltzmann constant (Gas constant) in kCal/(mol K)
simulation_steps = 100 # Number of steps used in individual Langevin dynamics simulations
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds
box_size = 200.00 * unit.angstroms  # box width
cutoff = box_size / 2.0 * 0.99

# Coarse-grained model settings
backbone_length = 1
sidechain_length = 1
sidechain_positions = [0]
chain_length = 8
mass = 1.0 * unit.amu
sigma = 1.0 # unit.angstroms
epsilon = 0.2 * unit.kilocalorie_per_mole

# =============================================================================================
# 3) BUILD COARSE-GRAINED MODEL
# =============================================================================================

def assign_default_box_vectors(system,box_size):
 a = unit.Quantity(np.zeros([3]), unit.angstroms)
 a[0] = box_size
 b = unit.Quantity(np.zeros([3]), unit.angstroms)
 b[1] = box_size
 c = unit.Quantity(np.zeros([3]), unit.angstroms)
 c[2] = box_size
 system.setDefaultPeriodicBoxVectors(a, b, c)
 return(system)

def get_distance(bead_1,bead_2):
 sqrt_arg = 0.0
 for direction in range(0,len(bead_1)):
  sqrt_arg = sqrt_arg + ( bead_2[direction] - bead_1[direction] ) ** 2.0
 distance = math.sqrt(sqrt_arg)
 return(distance)

def assign_position(parent_atom_coordinates):
   move = np.zeros([3])
   new_coordinates = np.zeros([3])
   move[0] = math.sqrt(random.uniform(0.0,1.0))
   move[1] = math.sqrt(random.uniform(0.0,1.0-(move[0]**2.0)))
   move[2] = math.sqrt(1.0-(move[0]**2.0 + move[1]**2.0))
   new_coordinates[0] = parent_atom_coordinates[0] + move[0]
   new_coordinates[1] = parent_atom_coordinates[1] + move[1]
   new_coordinates[2] = parent_atom_coordinates[2] + move[2]
   return(new_coordinates)

def assign_random_initial_coordinates(chain_length,backbone_length,sidechain_length):
# Define array for initial Cartesian coordinates
 positions = []
 for monomer in range(0,chain_length):
# Assign all backbone bead positions
  for bead in range(0,backbone_length):
   monomer_start = len(positions)
   if monomer == 0:
    new_coordinates = np.zeros([3])
   else:
    new_coordinates = assign_position(positions[monomer_start - 1])
    print(get_distance(new_coordinates,positions[len(positions)-1]))
   positions.append(new_coordinates)
# Assign all side-chain bead positions
   if bead in sidechain_positions:
    backbone_parent = len(positions)
    for sidechain in range(0,sidechain_length):
     new_coordinates = assign_position(positions[backbone_parent - 1])
     print(get_distance(positions[len(positions)-2],positions[len(positions)-1]))
     positions.append(new_coordinates)
 return(positions)

def add_position_units(positions):
 positions = unit.Quantity(positions, unit.angstroms)
 return(positions)

def write_coordinates_to_pdb(coordinates):
 pdb_object = open("coordinates.pdb","w")
 for bead in coordinates:
  pdb_object.write(str("CG1 "+str(bead[0])+" "+str(bead[1])+" "+str(bead[2])+"\n"))
 pdb_object.write(str("END"))
 pdb_object.close()
 return

def build_cg_1_1_model(epsilon,sigma):
 system = mm.System()
 return()

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Configure logger.
    config_root_logger(False)

    start_time = timeit.default_timer()
    positions = assign_random_initial_coordinates(chain_length,backbone_length,sidechain_length)
#    positions = add_position_units(assign_random_initial_coordinates(chain_length,backbone_length,sidechain_length))
    write_coordinates_to_pdb(positions)
    stop_time = timeit.default_timer()
    print("Calculation time was: "+str(stop_time-start_time)+" seconds.")
    print(positions)
