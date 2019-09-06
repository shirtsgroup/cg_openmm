Building OpenMM objects for coarse grained modeling
===================================================

All `OpenMM <http://openmm.org>`_ simulations require a `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_ and a `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_.  This chapter details procedures for building those objects for coarse grained models with user-defined properties.

Building an OpenMM System() for a coarse grained model.
-------------------------------------------------------

An OpenMM `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_ object contains force definitions for a molecular model.  The 'cg_openmm' repository applies default definitions for all forces (see `OpenMM user guide  <http://docs.openmm.org/development/userguide/theory.html>`_ for detailed definitions).

Shown below are tools and functions needed to build and verify an OpenMM `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_ .

.. automodule:: build.cg_build
    :members: build_system, verify_system

.. raw:: latex

    \newpage

Building an OpenMM Topology() for a coarse grained model.
---------------------------------------------------------

An OpenMM `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_ object contains structural definitions for a molecular model (bond assignments, residue assignments, etc.).

Shown below are tools and functions needed to build and verify an OpenMM `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_ .

.. automodule:: build.cg_build
    :members: build_topology, verify_topology

.. raw:: latex

    \newpage

Configuring OpenMM Forces() for a coarse grained model.
-------------------------------------------------------

The 'cg_openmm' package contains multiple tools for verifying the validity of forces that are added to an OpenMM System().  These tools are shown below:

.. automodule:: build.cg_build
    :members: add_force, test_force, test_forces, get_num_forces

.. raw:: latex

    \newpage

Other tools for building and verifying the OpenMM System() and Topology()
-------------------------------------------------------------------------

Shown below are other utilities to build and verify a coarse grained model System()/Topology() for OpenMM:

.. automodule:: build.cg_build
    :members:
    :exclude-members: build_topology, verify_topology, add_force, test_force, test_forces, get_num_forces, build_system, verify_system

