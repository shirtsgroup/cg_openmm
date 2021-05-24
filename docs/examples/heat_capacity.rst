Calculating heat capacity
=========================

Heat capacity as a function of temperature is a useful indicator of a folding
transition. This example calculates the constant volume heat capacity curve and uncertainty from
decorrelated replica energies. MBAR reweighting is used to computate the expectation
of heat capacity at each temperature of interest, including intermediate states
not sampled directly in the replica exchange MD simulation. A bootstrapping scheme
is used to compute uncertainty in not only the heat capacity values, but also the
melting point (of folding), height of the heat capacity curve relative to the minimum
in the temperature range studied, and full-width half-maximum (FWHM) of the curve.

.. code-block:: python

    import os
    import pickle
    
    from cg_openmm.thermo.calc import *
    from simtk import unit    
    
    # Specify location of the .nc files:
    output_directory = "output"
    output_data = os.path.join(output_directory, "output.nc")

    # Specify starting frame and frames between each uncorrelated energy data point:
    # These should be determined using process_replica_exchange_data
    frame_begin = 20000
    sample_spacing = 10

    # Calculate heat capacity using mbar and bootstrapping:                                                                    
    (new_temperature_list, C_v, dC_v, Tm_value, Tm_uncertainty, 
    Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty) = bootstrap_heat_capacity(
        output_data=output_data,
        frame_begin=frame_begin
        sample_spacing=sample_spacing,
        num_intermediate_states=3,
        n_trial_boot=200,
        conf_percent='sigma',
        plot_file="heat_capacity_boot.pdf",
    )