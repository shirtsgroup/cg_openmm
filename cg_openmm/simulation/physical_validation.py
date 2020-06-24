import os
import physical_validation as pv
from physical_validation.data.simulation_data import SimulationData
from physical_validation.data.observable_data import ObservableData
from physical_validation.data.ensemble_data import EnsembleData
from physical_validation.data.unit_data import UnitData
import matplotlib.pyplot as pyplot
from simtk import unit
import numpy as np
from openmmtools.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler
from openmmtools.multistate import ReplicaExchangeAnalyzer

kB = (unit.MOLAR_GAS_CONSTANT_R).in_units_of(unit.kilojoule / (unit.kelvin * unit.mole))

def physical_validation_ensemble(
    temperature_list, ref_state_index=0, output_data="output.nc", output_directory="ouput", plotfile='ensemble_check'
):
    """
    Run ensemble physical validation test for 2 states in replica exchange simulation

    :param temperature_list: List of temperatures that will be used to define different replicas (thermodynamics states)
    :type temperature_list: List( `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_ * number_replicas )

    :param ref_state_index: Index in temperature_list to use as one of the states in the ensemble check. The other state will be chosen based on the energy standard deviation at the reference state
    :type ref_state_index: int
    
    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data
    :type output_data: str
    
    :param plotfile: Filename for outputting ensemble check plot
    :type plotfile: str
    
    """

    # Read the energies for individual temperature replicas
    reporter = MultiStateReporter(output_data, open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)

    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()

    n_particles = np.shape(reporter.read_sampler_states(iteration=0)[0].positions)[0]
    temps = np.array([temp._value for temp in temperature_list])
    beta_k = 1 / (kB * temps)
    n_replicas = len(temperature_list)
    for k in range(n_replicas):
        replica_energies[:, k, :] *= beta_k[k] ** (-1)

    total_steps = len(replica_energies[0][0])
    state_energies = np.zeros([n_replicas, total_steps])

    for step in range(total_steps):
        for state in range(n_replicas):
            state_energies[state, step] = replica_energies[
                np.where(replica_state_indices[:, step] == state)[0], 0, step
            ]

    state_energies *= unit.kilojoule_per_mole


    # Find optimal state pair for ensemble check:
    # Compute standard deviations of each energy distribution:
    state_energies_std = np.std(state_energies,axis=1)

    # Select reference state point
    T_ref = temperature_list[ref_state_index]
    std_ref = state_energies_std[ref_state_index]

    # Compute optimal spacing:
    deltaT = 2*kB*T_ref**2/std_ref
    #print("DeltaT: %r" %deltaT) 

    # Find closest match
    T_array = np.zeros(len(temperature_list))
    for i in range(len(temperature_list)):
        T_array[i] = temperature_list[i]._value

    T_diff = np.abs(T_ref._value-T_array)

    T_opt_index = np.argmin(np.abs(deltaT._value - T_diff))
    T_opt = temperature_list[T_opt_index]
    #print(T_opt)

    # Set SimulationData for physical validation

    # State 1
    state1_index = ref_state_index

    sim_data1 = SimulationData()

    sim_data1.observables = ObservableData(
        potential_energy=state_energies[state1_index,:],
        )
        
    sim_data1.ensemble = EnsembleData(
        ensemble='NVT',
        energy=state_energies[state1_index,:],
        temperature=T_array[state1_index]
        )
        
    # Check that these are correct units and conversions to Gromacs units reference set
    sim_data1.units = UnitData(
        kb=kB._value,
        energy_conversion=1,
        length_conversion=1,
        volume_conversion=1,
        temperature_conversion=1,
        pressure_conversion=1,
        time_conversion=1,
        energy_str='KJ/mol',
        length_str='nm',
        volume_str='nm^3',
        temperature_str='K',
        pressure_str='bar',
        time_str='ps'
        )
        
    # State 2
    state2_index = T_opt_index
    sim_data2 = SimulationData()

    sim_data2.observables = ObservableData(
        potential_energy=state_energies[state2_index,:],
        )
        
    sim_data2.ensemble = EnsembleData(
        ensemble='NVT',
        energy=state_energies[state2_index,:],
        temperature=T_array[state2_index]
        )
        
    # Check that these are correct units and conversions to Gromacs units reference set
    sim_data2.units = UnitData(
        kb=kB._value,
        energy_conversion=1,
        length_conversion=1,
        volume_conversion=1,
        temperature_conversion=1,
        pressure_conversion=1,
        time_conversion=1,
        energy_str='KJ/mol',
        length_str='nm',
        volume_str='nm^3',
        temperature_str='K',
        pressure_str='bar',
        time_str='ps'
        )
        
    # Run physical validation
    quantiles = pv.ensemble.check(
        sim_data1,
        sim_data2,
        total_energy=False,
        filename=plotfile
        )

    return quantiles