.. cg_openmm documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cg_openmm
=========================================================

``cg_openmm`` is a Python package for setup and thermodynamic analysis of coarse-grained
oligomer replica exchange molecular dynamics (REMD) simulations, with a focus on identifying
and analyzing cooperative folding transitions.

 - Build coarse-grained hetero-oligomer topology and force field definitions using the highly customizable
   potentials in OpenMM
 - Set up GPU-accelerated REMD simulations with a ``CGModel`` and a few lines of python code
 - Compute heat capacity as a near-continuous function of temperature using MBAR and a bootstrapping scheme
 - Compute native contact fraction expectation and free energy of folding as functions of temperature using
   MBAR and a bootstrapping scheme (requires a native structure determined using ``analyze_foldamers``)
 - Predict helix-stabilizing force field parameters using a helical geometry optimization tool 
 
.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   examples
   
.. toctree::
   :maxdepth: 1
   :caption: API documentation:
   
   build
   cgmodel
   ensembles
   parameters
   simulation
   thermo
   utilities
    
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
