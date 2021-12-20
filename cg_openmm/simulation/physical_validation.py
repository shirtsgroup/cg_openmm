import os
import physical_validation as pv
from physical_validation.data.simulation_data import SimulationData
from physical_validation.data.observable_data import ObservableData
from physical_validation.data.ensemble_data import EnsembleData
from physical_validation.data.unit_data import UnitData
from physical_validation.util.error import InputError
import matplotlib.pyplot as pyplot
import simtk.unit as unit
import numpy as np
from openmmtools.multistate import MultiStateReporter, MultiStateSampler, ReplicaExchangeSampler
from openmmtools.multistate import ReplicaExchangeAnalyzer

kB = unit.MOLAR_GAS_CONSTANT_R # Boltzmann constant

def physical_validation_ensemble(
    output_data="output.nc", output_directory="ouput", plotfile='ensemble_check', 
    pairs='single',ref_state_index=0
):
    """
    Run ensemble physical validation test for 2 states in replica exchange simulation

    :param output_data: Path to the output data for a NetCDF-formatted file containing replica exchange simulation data
    :type output_data: str
    
    :param plotfile: Filename for outputting ensemble check plot
    :type plotfile: str
    
    :param pairs: Option for running ensemble validation on all replica pair combinations ('all'), adjacent pairs ('adjacent'), or single pair with optimal spacing ('single')
    
    :param ref_state_index: Index in temperature_list to use as one of the states in the ensemble check. The other state will be chosen based on the energy standard deviation at the reference state. Ignored if pairs='all'
    :type ref_state_index: int
    
    """

    # Get temperature list and read the energies for individual temperature replicas
    reporter = MultiStateReporter(output_data, open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)

    states = reporter.read_thermodynamic_states()[0]
    temperature_list = []
    for s in states:
        temperature_list.append(s.temperature)
    
    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()

    n_particles = np.shape(reporter.read_sampler_states(iteration=0)[0].positions)[0]
    
    # Close the data file:
    reporter.close()     
    
    T_unit = temperature_list[0].unit
    temps = np.array([temp.value_in_unit(T_unit) for temp in temperature_list])
    beta_k = 1 / (kB.value_in_unit(unit.kilojoule_per_mole/T_unit) * temps)
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
    
    T_array = np.zeros(len(temperature_list))
    for i in range(len(temperature_list)):
        T_array[i] = temperature_list[i].value_in_unit(T_unit)
    
    if pairs.lower() != 'single' and pairs.lower() != 'adjacent' and pairs.lower() != 'all':
        print(f"Error: Pair option '{pairs}' not recognized, using default option 'single'")
        pairs = 'single'

    if pairs.lower() == 'single': 
    # Run ensemble validation on one optimally spaced temperature pair
        quantiles = {}
    
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
        T_diff = np.abs(T_ref.value_in_unit(T_unit)-T_array)

        T_opt_index = np.argmin(np.abs(deltaT.value_in_unit(T_unit) - T_diff))
        T_opt = temperature_list[T_opt_index]

        # Set SimulationData for physical validation
        state1_index = ref_state_index
        state2_index = T_opt_index

        sim_data1, sim_data2 = set_simulation_data(
            state_energies,
            T_array,
            state1_index,
            state2_index
            )
        
        # Run physical validation
        try:
            quantiles_ij = pv.ensemble.check(
                sim_data1,
                sim_data2,
                total_energy=False,
                filename=plotfile
                )
            
            quantiles[f"state{state1_index}_state{state2_index}"] = quantiles_ij[0]
            
        except InputError:
            print(f"Insufficient overlap between trajectories for states {state1_index} and {state2_index}. Skipping...")

        
    elif pairs.lower() == 'adjacent':
    # Run ensemble validation on all adjacent temperature pairs
        quantiles = {}

        for i in range(len(temperature_list)-1):
            # Set SimulationData for physical validation
            state1_index = i
            state2_index = i+1
            
            sim_data1, sim_data2 = set_simulation_data(
                state_energies,
                T_array,
                state1_index,
                state2_index
                )
            
            # Run physical validation
            try:
                quantiles_ij = pv.ensemble.check(
                    sim_data1,
                    sim_data2,
                    total_energy=False,
                    filename=f"{plotfile}_{state1_index}_{state2_index}"
                    )
                    
                quantiles[f"state{state1_index}_state{state2_index}"] = quantiles_ij[0]
                
            except InputError:
                print(f"Insufficient overlap between trajectories for states {state1_index} and {state2_index}. Skipping...")
    
    elif pairs.lower() == 'all':
    # Run ensemble validation on all combinations of temperature pairs
        quantiles = {}
        for i in range(len(temperature_list)):
            for j in range(i+1,len(temperature_list)):
                # Set SimulationData for physical validation
                state1_index = i
                state2_index = j

                sim_data1, sim_data2 = set_simulation_data(
                    state_energies,
                    T_array,
                    state1_index,
                    state2_index
                    )
                
                # Run physical validation
                try:              
                    quantiles_ij = pv.ensemble.check(
                        sim_data1,
                        sim_data2,
                        total_energy=False,
                        filename=f"{plotfile}_{state1_index}_{state2_index}"
                        )
                        
                    quantiles[f"state{state1_index}_state{state2_index}"] = quantiles_ij[0]
                
                except InputError:
                    print(f"Insufficient overlap between trajectories for states {state1_index} and {state2_index}. Skipping...")
                    
                    
    return quantiles
    
def set_simulation_data(
    state_energies, T_array, state1_index, state2_index
):
    """
    Create and set SimulationData objects for a pair of specified states
    
    """
    
    # Set default UnitData object
    default_UnitData = UnitData(
        kb=kB.value_in_unit(unit.kilojoule_per_mole/unit.kelvin),
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
    
    # State 1
    sim_data1 = SimulationData()

    sim_data1.observables = ObservableData(
        potential_energy=state_energies[state1_index,:],
        )
        
    sim_data1.ensemble = EnsembleData(
        ensemble='NVT',
        energy=state_energies[state1_index,:],
        temperature=T_array[state1_index]
        )
        
    sim_data1.units = default_UnitData
        
    # State 2
    sim_data2 = SimulationData()

    sim_data2.observables = ObservableData(
        potential_energy=state_energies[state2_index,:],
        )
        
    sim_data2.ensemble = EnsembleData(
        ensemble='NVT',
        energy=state_energies[state2_index,:],
        temperature=T_array[state2_index]
        )
        
    sim_data2.units = default_UnitData
    
    return sim_data1, sim_data2