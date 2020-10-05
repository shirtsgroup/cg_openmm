"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed
import os
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.parameters.free_energy import *
from cg_openmm.parameters.reweight import get_temperature_list
from cg_openmm.utilities.util import fit_sigmoid
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))
structures_path = os.path.join(current_path, 'test_structures')
data_path = os.path.join(current_path, 'test_data')

def test_native_contacts_pdb(tmpdir):
    """See if we can determine native contacts and classify folded vs. unfolded states"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 300.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of pdb trajectories to analyze
    # For fraction_native_contacts vs. T, we use state trajectories.
    # However, we can test with the replica pdbs:
    pdb_file_list = []
    for i in range(len(temperature_list)):
        pdb_file_list.append(f"{data_path}/replica_{i+1}.pdb")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.pdb"
        
    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 3.5* unit.angstrom

    # Cutoff for current trajectory distances, as a multiple of native_contact_cutoff
    native_contact_cutoff_ratio = 1.25
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff
    )
    
    # Determine native contact fraction of current trajectories:
    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        pdb_file_list,
        native_contact_list,
        native_contact_distances,
        native_contact_cutoff_ratio=native_contact_cutoff_ratio
    )   

    plot_native_contact_fraction(
        temperature_list,
        Q_avg,
        Q_stderr,
        plotfile=f"{output_directory}/Q_vs_T.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_vs_T.pdf")
    

def test_native_contacts_dcd(tmpdir):
    """See if we can determine native contacts and classify folded vs. unfolded states"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 300.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of pdb trajectories to analyze
    # For fraction_native_contacts vs. T, we use state trajectories.
    # However, we can test with the replica pdbs:
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"
    
    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 3.5* unit.angstrom

    # Cutoff for current trajectory distances, as a multiple of native_contact_cutoff
    native_contact_cutoff_ratio = 1.25
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff
    )
    
    # Determine native contact fraction of current trajectories:
    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        native_contact_cutoff_ratio=native_contact_cutoff_ratio
    )   

    plot_native_contact_fraction(
        temperature_list,
        Q_avg,
        Q_stderr,
        plotfile=f"{output_directory}/Q_vs_T.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_vs_T.pdf")
    
    
def test_expectations_fraction_contacts_pdb(tmpdir):
    """See if we can determine native contacts expectations as a function of T"""

    output_directory = tmpdir.mkdir("output")

    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 300.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    pdb_file_list = []
    for i in range(len(temperature_list)):
        pdb_file_list.append(f"{data_path}/replica_{i+1}.pdb")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.pdb"

    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 3.5* unit.angstrom
    
    # Cutoff for current trajectory distances, as a multiple of native_contact_cutoff
    native_contact_cutoff_ratio = 1.25

    # Get native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff
    )

    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        pdb_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=100,
        native_contact_cutoff_ratio=native_contact_cutoff_ratio
    )
    
    # Determine how many folding transitions each replica underwent:
    # plot Q_avg vs. frame
    plot_native_contact_timeseries(
        Q,
        frame_begin=100,
        time_interval=1*unit.picosecond,
        plot_per_page=3,
        plotfile=f"{output_directory}/Q_vs_time.pdf",
        figure_title="Native contact fraction",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_vs_time.pdf")
    
    output_data = os.path.join(data_path, "output.nc")
    num_intermediate_states=1

    results = expectations_fraction_contacts(
        Q,
        temperature_list,
        frame_begin=100,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
    )
    
    plot_native_contact_fraction(
        results["T"],
        results["Q"],
        results["dQ"],
        plotfile=f"{output_directory}/Q_expect_vs_T.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_expect_vs_T.pdf")
    
    
    # Test free energy of folding:
    
    # Cutoff for native contact fraction folded vs. unfolded states:
    Q_folded = 0.9
    
    # Array folded states can be all frames, or only selected frames.
    # It is trimmed to the correct size in expectations_free_energy.
    array_folded_states = np.zeros((len(Q[:,0]),len(pdb_file_list)))
    
    for rep in range(len(pdb_file_list)):
        # Classify into folded/unfolded states:
        for frame in range(len(Q[:,rep])):
            if Q[frame,rep] >= Q_folded:
                # Folded
                array_folded_states[frame,rep] = 1
            else:
                # Unfolded
                array_folded_states[frame,rep] = 0

    full_T_list, deltaF_values, deltaF_uncertainty = expectations_free_energy(
        array_folded_states,
        temperature_list,
        frame_begin=100,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
    )
    
    plot_free_energy_results(
        full_T_list,
        deltaF_values,
        deltaF_uncertainty,
        plotfile=f"{output_directory}/free_energy.pdf"
    )

    assert os.path.isfile(f"{output_directory}/free_energy.pdf")
    
    # Test free energy fitting / derivative calculation:
    ddeltaF_out, d2deltaF_out, spline_tck = get_free_energy_derivative(
        deltaF_values['state0_state1'],
        full_T_list,
        plotfile=f"{output_directory}/ddeltaF_dT.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/ddeltaF_dT.pdf")


def test_expectations_fraction_contacts_dcd(tmpdir):
    """See if we can determine native contacts expectations as a function of T"""

    output_directory = tmpdir.mkdir("output")

    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 300.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"

    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 3.5* unit.angstrom
    
    # Cutoff for current trajectory distances, as a multiple of native_contact_cutoff
    native_contact_cutoff_ratio = 1.25

    # Get native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff
    )

    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=100,
        native_contact_cutoff_ratio=native_contact_cutoff_ratio
    )
    
    # Determine how many folding transitions each replica underwent:
    # plot Q_avg vs. frame
    plot_native_contact_timeseries(
        Q,
        frame_begin=100,
        time_interval=1*unit.picosecond,
        plot_per_page=3,
        plotfile=f"{output_directory}/Q_vs_time.pdf",
        figure_title="Native contact fraction",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_vs_time.pdf")
    
    output_data = os.path.join(data_path, "output.nc")
    num_intermediate_states=1

    results = expectations_fraction_contacts(
        Q,
        temperature_list,
        frame_begin=100,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
    )
    
    plot_native_contact_fraction(
        results["T"],
        results["Q"],
        results["dQ"],
        plotfile=f"{output_directory}/Q_expect_vs_T.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_expect_vs_T.pdf")    
    
    # Test sigmoid fitting function on Q_expect_vs_T data:
    param_opt, param_cov = fit_sigmoid(
        results["T"],
        results["Q"],
        plotfile=f"{output_directory}/Q_vs_T_fit.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_vs_T_fit.pdf")  
    