Installation
===============

Currently, only the development version of ``cg_openmm`` is available.

Download cg_openmm from GitHub using:

``git clone https://github.com/shirtsgroup/cg_openmm.git``

In the base directory of cg_openmm, install using:

``python setup.py install``

To use structural analysis features such as:

1) predicting a native structure by clustering REMD trajectories
2) calculating bonded distributions

you will also need to install ``analyze_foldamers``, available through GitHub:

``git clone https://github.com/shirtsgroup/analyze_foldamers.git``

In the base directory of ``analyze_foldamers``, install using:

``python setup.py install

Conda environment
-----------------

``cg_openmm`` is currently tested and maintained on python 3.6, 3.7, and 3.8.

Create an anaconda environment for running cg_openmm by specifying the following dependencies:
``conda create -n cg_openmm_env python=3.X mdtraj mpi4py numpy openmm openmmtools physical_validation pymbar
scikit-learn scipy``

Alternatively, an environment can be created from one of the .yml files provided with ``cg_openmm``.

