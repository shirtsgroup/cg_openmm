cg_openmm
==============================
[//]: # (Badges)
[![GH Actions Status](https://github.com/shirtsgroup/cg_openmm/workflows/CI/badge.svg)](https://github.com/shirtsgroup/cg_openmm/actions?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/shirtsgroup/cg_openmm/branch/master/graph/badge.svg)](https://codecov.io/gh/shirtsgroup/cg_openmm/branch/master)
[![Documentation Status](https://readthedocs.org/projects/cg-openmm/badge/?version=latest)](https://cg-openmm.readthedocs.io/en/latest/?badge=latest)

`CG_openmm` is a python package for setup and thermodynamic analysis of coarse-grained foldamer replica exchange molecular dynamics simulations run using OpenMM.

For further details refer to the web-based documentation:

https://cg-openmm.readthedocs.io/en/latest/

## Installation
To install `CG_openmm`, run `python setup.py install` in the base directory.

To use structural analysis features, including native structure determination and calculation of bonded distributions, also install [analyze_foldamers](https://github.com/shirtsgroup/analyze_foldamers).

## Dependencies:

[mdtraj](https://mdtraj.org/1.9.4/index.html) | [mpi4py](https://mpi4py.readthedocs.io/en/stable/) | [numpy](https://numpy.org) | [openmm](http://openmm.org/) | [openmmtools](https://openmmtools.readthedocs.io/en/latest/) | [physical_validation](https://physical-validation.readthedocs.io/en/latest/index.html) | [pymbar](https://pymbar.readthedocs.io/en/master/) | [scikit-learn](https://scikit-learn.org/stable/index.html) | [scipy](https://scipy.org) 

## Testing your installation:

To test your installation, open a new Python session and type the command `import cg_openmm`.
