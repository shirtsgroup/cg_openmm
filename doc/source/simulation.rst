OpenMM simulation tools for coarse grained modeling 
===================================================

Building OpenMM Simulation() objects
------------------------------------

OpenMM simulations are propagated using a `Simulation() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1simulation_1_1Simulation.html>`_ object.

Shown below are the main tools needed to build OpenMM Simulaton() objects for coarse grained modeling.

.. automodule:: simulation.tools
    :members: build_mm_simulation, run_simulation

.. raw:: latex

    \newpage

Running Yank replica exchange simulations
-----------------------------------------

The `Yank <http://getyank.org/0.23.4/index.html>`_ python package is used to perform replica exchange sampling with OpenMM simulations.

Shown below are the main functions and tools necessary to conduct Yank replica exchange simulations with a coarse grained model in OpenMM.

.. automodule:: simulation.rep_exch
    :members: run_replica_exchange, read_replica_exchange_data, make_replica_pdb_files

.. raw:: latex

    \newpage

Tools to plot coarse grained model simulation results
-----------------------------------------------------

The 'cg_openmm' package contains multiple functions to plot OpenMM and Yank simulation results using `matplotlib <https://matplotlib.org/>`_ .

Shown below is example output from a Yank replica exchange simulation run:

.. image:: replica_exchange_energies.png
.. image:: replica_exchange_transitions.png

Shown below are functions which allow plotting of simulation results.

.. automodule:: simulation.rep_exch
    :members: plot_replica_exchange_energies, plot_replica_exchange_summary

.. automodule:: simulation.tools
    :members: plot_simulation_data, plot_simulation_results

.. raw:: latex

    \newpage

Other simulation tools
----------------------

Shown below are other tools which aid the building and verification of OpenMM simulation objects.

.. automodule:: simulation.tools
    :members:
    :exclude-members: build_mm_simulation, run_simulation, plot_simulation_data, plot_simulation_results

.. automodule:: simulation.rep_exch
    :members:
    :exclude-members: plot_replica_exchange_energies, plot_replica_exchange_summary, run_replica_exchange, read_replica_exchange_data, make_replica_pdb_files

