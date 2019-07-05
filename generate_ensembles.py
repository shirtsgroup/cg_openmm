#!/usr/bin/python

import os
from simtk import unit
from foldamers.src.cg_model.cgmodel import basic_cgmodel
from foldamers.src.ensembles.ens_build import get_ensemble

# Coarse grained model settings
polymer_length=8
backbone_length=1
sidechain_length=1
sidechain_positions=[0]
mass = unit.Quantity(10.0,unit.amu)
sigma = unit.Quantity(2.4,unit.angstrom)
bond_length = unit.Quantity(1.0,unit.angstrom)
epsilon = unit.Quantity(0.5,unit.kilocalorie_per_mole)

# Build a basic 1-1 coarse grained model (homopolymer)
cgmodel = basic_cgmodel(polymer_length=polymer_length,backbone_length=backbone_length,sidechain_length=sidechain_length,sidechain_positions=sidechain_positions,mass=mass,bond_length=bond_length,sigma=sigma,epsilon=epsilon)

# Get a general ensemble for this model.
ensemble_positions = get_ensemble(cgmodel)
# Get a low energy ensemble for this model.
low_energy_ensemble_positions = get_ensemble(cgmodel,low_energy=True)
high_energy_ensemble_positions = get_ensemble(cgmodel,high_energy=True)

exit()
