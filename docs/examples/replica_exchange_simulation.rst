Replica exchange MD simulations
===============================

Setting up a replica exchange MD simulation
-------------------------------------------

Once a CGModel has been created, a replica exchange MD simulation can be launched on a GPU with run_replica_exchange
after specifying range of temperatures, number of replicas, Langevin integrator parameters, and frequency of exchanges
between replicas. The setup of the REMD simulation is facilitated by the ``openmmtools`` package. A Langevin integrator
is used, with exchanges attempted between neighboring temperatures at each iteration.

The following example loads in a previously built CGModel and launches a REMD simulation.

.. code-block:: python

    import os

    import numpy as np
    import simtk.openmm as openmm
    from cg_openmm.cg_model.cgmodel import CGModel
    from cg_openmm.parameters.reweight import get_temperature_list
    from cg_openmm.simulation.rep_exch import *
    from openmmtools.cache import global_context_cache
    from simtk import unit
    import pickle

    # Set output directory:
    output_directory = "output"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Use CUDA platform for GPU acceleration:
    global_context_cache.platform = openmm.Platform.getPlatformByName("CUDA")

    # Replica exchange simulation settings:
    total_simulation_time = 50.0 * unit.nanosecond
    simulation_time_step = 5.0 * unit.femtosecond
    total_steps = int(np.floor(total_simulation_time / simulation_time_step))
    output_data = os.path.join(output_directory, "output.nc")

    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    # Use logarithmic temperature spacing:
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

    exchange_frequency = 200 # Number of steps between exchange attempts
    collision_frequency = 5 / unit.picosecond

    # Load in a cgmodel:
    cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))

    run_replica_exchange(
        cgmodel.topology,
        cgmodel.system,
        cgmodel.positions,
        friction=collision_frequency,
        temperature_list=temperature_list,
        simulation_time_step=simulation_time_step,
        total_simulation_time=total_simulation_time,
        exchange_frequency=exchange_frequency,
        output_data=output_data,
    )
    
Processing replica exchange output
----------------------------------

Replica exchange simulations produce an output.nc file, which contains energies of each replica over
time, and an output_checkpoint.nc file, which contains particle coordinate trajectories. To perform
thermodynamic analysis, the energies in the output.nc file need to be decorrelated, which is 
accomplished in cg_openmm using the pyMBAR timeseries module.

This following decorrelates the replica energies, computes mixing statistics of the simulation,
and plots the energy timeseries of each thermodynamic state.

.. code-block:: python

    # Specify location of the .nc files:
    output_directory = "output"
    output_data = os.path.join(output_directory, "output.nc")

    # Process replica exchange energy data:
    (replica_energies, replica_states, production_start,
    sample_spacing, n_transit, mixing_stats) = process_replica_exchange_data(
        output_data=output_data,
        output_directory=output_directory,
        write_data_file=True,
    )
    
It is also useful to generate individual replica trajectories (continuous) or individual state
trajectories (discontinuous at constant temperature) from the output_checkpoint.nc file. These
individual trajectories are more manageable for tasks such as trajectory visualization and 
calculation of bonded distributions at each thermodynamic state. Trajectories can be written
in either .pdb or .dcd format.

:Writing constant replica trajectories:
.. code-block:: python

    make_replica_dcd_files(
        cgmodel.topology,
        timestep=5*unit.femtosecond,
        time_interval=200,
        output_dir=output_directory
    )

:Writing constant thermodynamic state trajectories:
.. code-block:: python
    
    make_state_dcd_files(
        cgmodel.topology,
        timestep=5*unit.femtosecond,
        time_interval=200,
        output_dir=output_directory
    )
