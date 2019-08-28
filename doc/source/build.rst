Building OpenMM objects for coarse grained modeling
===================================================

Building an OpenMM System() and Topology()
------------------------------------------

All OpenMM simulations require a `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_ and a `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_.

Listed below are functions and classes that aid the building of OpenMM System() and Topology() class objects for coarse grained models with user-defined properties:

.. automodule:: build.cg_build
    :members: build_system, build_topology

Other tools for building and verifying the OpenMM System() and Topology()
-------------------------------------------------------------------------

Shown below are other functions/tools to build and verify the System/Topology:

.. automodule:: build.cg_build
    :members:
    :exclude-members: build_system, build_topology

