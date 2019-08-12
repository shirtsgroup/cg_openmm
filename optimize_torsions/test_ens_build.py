from simtk.openmm.app.pdbfile import PDBFile
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.ensembles.ens_build import *
native_structure = PDBFile('/home/gmeek/Foldamers/foldamers/examples/optimize_torsions/output/re_min_-0.79_-0.79.pdb').getPositions()
cgmodel = CGModel()
ensemble,ensemble_energies = get_nonnative_ensemble(cgmodel,native_structure)
exit()
