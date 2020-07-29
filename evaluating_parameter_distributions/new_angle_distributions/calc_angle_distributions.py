import os
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
import pickle
from foldamers.parameters.angle_distributions import *

# This script calculates and plots angle distributions from a CGModel object and pdb trajectory

# Load in a trajectory pdb file:
traj_file = "simulation.pdb"

# Load in a CGModel:
cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))

angle_hist_data = calc_bond_angle_distribution(cgmodel,traj_file)
torsion_hist_data = calc_torsion_distribution(cgmodel,traj_file)

