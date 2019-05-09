#!/usr/bin/python

import os, sys, timeit
from io import StringIO
import numpy as np
import math, random
import matplotlib.pyplot as pyplot
# OpenMM utilities
import simtk.openmm.app.element as elem
from simtk.openmm.app.pdbfile import PDBFile
import mdtraj as md
from simtk import openmm as mm
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm.vec3 import Vec3
# foldamers utilities
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.iotools import *
from cg_openmm.src.cg_mm_tools.cg_openmm import *

box_size = 10.00 * unit.nanometer # box width
cutoff = box_size / 2.0 * 0.99
simulation_time_step = 0.002 * unit.picosecond # Units = picoseconds
temperature = 300.0 * unit.kelvin
print_frequency = 10 # Number of steps to skip when printing output
# Change this:
total_simulation_time = 1.0 * unit.picosecond # Units = picoseconds

# Model settings
backbone_length = 1 # Number of backbone beads
sidechain_length = 1 # Number of sidechain beads
sidechain_positions = [0] # Index of backbone bead on which the side chains are placed
polymer_length = 12 # Number of monomers in the polymer
mass = 12.0 * unit.amu # Mass of beads
sigma = 8.4 * unit.angstrom # Lennard-Jones interaction distance
bond_length = 1.0 * unit.angstrom # bond length
bond_force_constant = 9.9e5 # Units = kJ/mol/A^2
constrain_bonds = False
epsilon = 0.5 * unit.kilocalorie_per_mole # Lennard-Jones interaction strength
charge = 0.0 * unit.elementary_charge # Charge of beads

sigma = sigma.in_units_of(unit.nanometer)._value
charge = charge._value
epsilon = epsilon.in_units_of(unit.kilojoule_per_mole)._value
bond_length = bond_length.in_units_of(unit.nanometer)._value

cgmodel = CGModel(polymer_length=polymer_length,backbone_length=backbone_length, sidechain_length=sidechain_length, sidechain_positions = sidechain_positions, mass = mass, sigma = sigma, epsilon = epsilon, bond_length = bond_length, bond_force_constant = bond_force_constant, charge = charge,constrain_bonds=constrain_bonds)
pdb_file = "test.pdb"
write_pdbfile(cgmodel,pdb_file)
pdb_mm_obj = PDBFile(pdb_file)
topology = pdb_mm_obj.getTopology()
simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=temperature,simulation_time_step=simulation_time_step,total_simulation_time=simulation_time_step*print_frequency,output_pdb="dihedrals.pdb",output_data="dihedrals_sim_data.dat",print_frequency=print_frequency)
energies=[]
number_stages = round((total_simulation_time._value/simulation_time_step._value)/print_frequency)
print("Number of simulation stages is: "+str(number_stages))
for stage in range(number_stages):
 simulation.step(print_frequency)
 energy = round(simulation.context.getState(getEnergy=True).getPotentialEnergy()._value,2)
 energies.append(energy)
trajectory = md.load("dihedrals.pdb")
dihedral_list = get_dihedral_angles()
dihedrals = []
for dihedral in dihedral_list:
 angles = md.compute_dihedrals(trajectory,[dihedral])
 dihedrals.append(angles)
figure = pyplot.figure(0)
colors = pyplot.cm.rainbow(np.linspace(0, 1, len(dihedral_list)))
for dihedral, c in zip(dihedrals, colors):
 pyplot.scatter(dihedral, energies, color=c, figure=figure)
pyplot.xlabel("Dihedral Angle (Degrees)")
pyplot.ylabel("Potential Energy (kJ/mol)")
pyplot.title("Dihedral distribution data for simulation of 1,1-CG model")
pyplot.savefig("/CG-11-distribution.png")
pyplot.legend(dihedral_list)
pyplot.show()
pyplot.close()
exit()
