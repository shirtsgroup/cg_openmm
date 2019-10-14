import os, timeit
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.secondary_structure import get_helical_parameters, get_helical_data

positions = PDBFile(str(str(os.getcwd().split('examples')[0])+"ensembles/12_1_1_0/helix.pdb")).getPositions()

cgmodel = CGModel(positions=positions)
pitch,radius,monomers_per_turn,residual = get_helical_parameters(cgmodel)
print(pitch,radius,monomers_per_turn,residual)

exit()
