###!/usr/lccal/bin/env python

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

# Non-scientific python packages needed for this protocol

import os
import sys
import timeit
import socket
from io import StringIO
import numpy as np
from simtk import openmm as mm
from simtk.openmm.app import *
from simtk.openmm import LangevinIntegrator
import openmmtools as mmtools
from openmmtools import testsystems
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
import mdtraj as md
# This is where replica exchange utilities are imported from Yank

from yank import mpi, analyze
from yank.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler, ParallelTemperingSampler, SAMSSampler
from yank.multistate import ReplicaExchangeAnalyzer, SAMSAnalyzer
from yank.multistate.multistatereporter import _DictYamlLoader
from yank.utils import config_root_logger

if socket.gethostname() == "Louie":
 pdb_file="/mnt/d/Foldamers/OpenMM_CG_polymers/structure_files/CG_8-mer.pdb"
temperature = 300.0 
simulation_time_step = 0.002 # Units = picoseconds
kB = 0.008314462  #Boltzmann constant (Gas constant) in kJ/(mol*K)
simulation_steps = 100 # Number of steps used in individual Langevin dynamics simulations
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds

mass_CG1 = 1.0 * unit.amu
mass_CG2 = 1.0 * unit.amu
q_CG1 = 0.0 * unit.elementary_charge
q_CG2 = 0.0 * unit.elementary_charge
sigma_CG1 = 3.5 * unit.angstrom
sigma_CG2 = 4.5 * unit.angstrom
epsilon_CG1 = 0.2 * unit.kilocalorie_per_mole
epsilon_CG2 = 0.1 * unit.kilocalorie_per_mole

box_size = 200.00 * unit.angstroms  # box width
a = unit.Quantity(np.zeros([3]), unit.angstroms)
a[0] = box_size
b = unit.Quantity(np.zeros([3]), unit.angstroms)
b[1] = box_size
c = unit.Quantity(np.zeros([3]), unit.angstroms)
c[2] = box_size
# Create nonbonded force term.
force = mm.NonbondedForce()

# Set interactions to be periodic Ewald.
force.setNonbondedMethod(mm.NonbondedForce.Ewald)

# Set cutoff to be less than one half the box length.
cutoff = box_size / 2.0 * 0.99
force.setCutoffDistance(cutoff)


# ==============================================================================
# RUN COARSE-GRAINED SIMULATION
# ==============================================================================


def run_cg_simulation(verbose=False, verbose_simulation=False):
    # Create thermodynamic state and save positions.
    pdb_object = PDBFile(file=pdb_file)
    positions = pdb_object.getPositions()
    positions_test = testsystems.SodiumChlorideCrystal().positions
    mm_positions = []
    for particle in range(0,len(positions)):
     mm_positions.append(np.array([positions[particle][0], positions[particle][1], positions[particle][2]]))
    topology = pdb_object.getTopology()
    system = mm.System()
#    system.positions = positions
#    system.topology = topology
    system.setDefaultPeriodicBoxVectors(a, b, c)
    for particle in positions:
     system.addParticle(mass_CG1)
     force.addParticle(q_CG1,sigma_CG1,epsilon_CG1)
    system.addForce(force)
    integrator = LangevinIntegrator(temperature, total_simulation_time, simulation_time_step) # Define Langevin integrator
    # Create and configure simulation object.
    simulation = Simulation(topology,system,integrator)
    simulation.reporters.append(PDBReporter(str(os.getcwd()+"/coordinates.pdb"),print_frequency)) # Write simulation PDB coordinates
    simulation.reporters.append(StateDataReporter(str(os.getcwd()+"/sim_data.dat"), print_frequency, \
    step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True)) # Write simulation data
    print(mm_positions)
    print(positions_test)
    simulation.context.setPositions(mm_positions) # Assign particle positions for this context
    simulation.minimizeEnergy() # Set the simulation type to energy minimization
    simulation.step(simulation_steps) # Run the simulation 
    config_root_logger(verbose_simulation)

    # Clean up.

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Configure logger.
    config_root_logger(False)

    start_time = timeit.default_timer()
    run_cg_simulation()
    stop_time = timeit.default_timer()
    print("Calculation time was: "+str(stop_time-start_time)+" seconds.")
	
