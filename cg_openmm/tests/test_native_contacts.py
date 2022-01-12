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
    max_temp = 600.0 * unit.kelvin
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
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )
    
    # Determine native contact fraction of current trajectories:
    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        pdb_file_list,
        native_contact_list,
        native_contact_distances,
        native_contact_tol=native_contact_tol,
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
    max_temp = 600.0 * unit.kelvin
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
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )
    
    # Determine native contact fraction of current trajectories:
    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        native_contact_tol=native_contact_tol,
    )   

    plot_native_contact_fraction(
        temperature_list,
        Q_avg,
        Q_stderr,
        plotfile=f"{output_directory}/Q_vs_T.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/Q_vs_T.pdf")
    
    
def test_native_contacts_dcd_homopolymer_sym_linear(tmpdir):
    """
    Test native contact fraction calculation with homopolymer
    end-to-end symmetry checks.
    (linear homopolymer with no sidechains)
    """
    
    output_directory = tmpdir.mkdir("output")
    
    data_path_linear = os.path.join(data_path, 'linear_24mer')
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path_linear}/stored_cgmodel_24mer_linear.pkl", "rb" ))
    
    # Create list of dcd trajectories to analyze
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path_linear}/replica_{i+1}.dcd")
        
    # Set path to native structure file:    
    native_structure_file=f"{data_path_linear}/native_medoid_min.dcd"
    
    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )
    
    # Determine native contact fraction of current trajectories:
    # With end-to-end symmetry check:
    Q_sym, Q_avg_sym, Q_stderr_sym, decorrelation_spacing_sym = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        native_contact_tol=native_contact_tol,
        homopolymer_sym=True,
    )   
    
    # Without end-to-end symmetry check:
    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        native_contact_tol=native_contact_tol,
        homopolymer_sym=False,
    )      

    # End-to-end symmetry check should increase the contact fraction in all cases:
    assert Q_sym.all() >= Q.all()


def test_native_contacts_dcd_homopolymer_sym_sidechain(tmpdir):
    """
    Test native contact fraction calculation with homopolymer
    end-to-end symmetry checks.
    (1-1 homopolymer model with sidechains)
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of dcd trajectories to analyze
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Set path to native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"
    
    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )
    
    # Determine native contact fraction of current trajectories:
    # With end-to-end symmetry check:
    Q_sym, Q_avg_sym, Q_stderr_sym, decorrelation_spacing_sym = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        native_contact_tol=native_contact_tol,
        homopolymer_sym=True,
    )   
    
    # Without end-to-end symmetry check:
    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        native_contact_tol=native_contact_tol,
        homopolymer_sym=False,
    )      

    # End-to-end symmetry check should increase the contact fraction in all cases:
    assert Q_sym.all() >= Q.all()


def test_helix_contacts_dcd(tmpdir):
    """See if we can determine native contacts for helix backbone sequences"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of pdb trajectories to analyze
    # For fraction_native_contacts vs. T, we use state trajectories.
    # However, we can test with the replica pdbs:
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
        cgmodel,
        native_structure_file,
        backbone_type_name='bb',
    )
    
    
def test_expectations_fraction_contacts_pdb(tmpdir):
    """See if we can determine native contacts expectations as a function of T"""

    output_directory = tmpdir.mkdir("output")

    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
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
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5

    # Set starting frame:
    frame_begin = 200

    # Get native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )

    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        pdb_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=frame_begin,
        native_contact_tol=native_contact_tol,
    )
    
    # Determine how many folding transitions each replica underwent:
    # plot Q vs. frame
    plot_native_contact_timeseries(
        Q,
        frame_begin=frame_begin,
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
        frame_begin=frame_begin,
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
    Q_folded = 0.40

    full_T_list, deltaF_values, deltaF_uncertainty = expectations_free_energy(
        Q,
        Q_folded,
        temperature_list,
        frame_begin=frame_begin,
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
    
    # Test entropy/enthalpy of folding calculation:
    S_folding, H_folding = get_entropy_enthalpy(
        deltaF_values, full_T_list,
        )
    
    # Test free energy / entropy / enthalpy bootstrapping calculation:
    # From bootstrapping:
    (full_T_list_boot, deltaF_values_boot, deltaF_uncertainty_boot, \
        deltaS_values_boot, deltaS_uncertainty_boot, \
        deltaH_values_boot, deltaH_uncertainty_boot) = bootstrap_free_energy_folding(
        Q,
        Q_folded,
        frame_begin=frame_begin,
        sample_spacing=2,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=10,
        conf_percent='sigma',
        plotfile_dir=output_directory,
    )

    assert os.path.isfile(f"{output_directory}/free_energy_boot.pdf")
    assert os.path.isfile(f"{output_directory}/entropy_boot.pdf")
    assert os.path.isfile(f"{output_directory}/enthalpy_boot.pdf")
    
    # With specified confidence interval:
    (full_T_list_boot, deltaF_values_boot, deltaF_uncertainty_boot, \
        deltaS_values_boot, deltaS_uncertainty_boot, \
        deltaH_values_boot, deltaH_uncertainty_boot) = bootstrap_free_energy_folding(
        Q,
        Q_folded,
        frame_begin=frame_begin,
        sample_spacing=2,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=10,
        conf_percent=80,
        plotfile_dir=output_directory,
    )

    assert os.path.isfile(f"{output_directory}/free_energy_boot.pdf")
    assert os.path.isfile(f"{output_directory}/entropy_boot.pdf")
    assert os.path.isfile(f"{output_directory}/enthalpy_boot.pdf")
    

def test_expectations_fraction_contacts_dcd(tmpdir):
    """See if we can determine native contacts expectations as a function of T"""

    output_directory = tmpdir.mkdir("output")

    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
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
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5

    # Get native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )

    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=100,
        native_contact_tol=native_contact_tol,
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
    
    
def test_expectations_partial_contact_fractions_1state(tmpdir):
    """
    Test the partial contact fraction expectation code for the trivial case 
    of a 1 state system (should match the original expectation native contact fraction result)
    """

    output_directory = tmpdir.mkdir("output")

    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
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
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5

    # Set production frames:
    frame_begin = 100

    # Get native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )

    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=frame_begin,
        native_contact_tol=native_contact_tol,
    )
    
    # Set all states to be the same
    array_folded_states = np.zeros_like(Q)
    
    output_data = os.path.join(data_path, "output.nc")
    num_intermediate_states=1

    Q_values_partial, T_list_out = expectations_partial_contact_fractions(
        array_folded_states,
        Q,
        frame_begin=100,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
    )
    
    Q_results = expectations_fraction_contacts(
        Q,
        frame_begin=100,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
    )
    
    assert Q_results['Q'].all() == Q_values_partial['0'].all()
    
    
def test_expectations_partial_contact_fractions_2state(tmpdir):
    """
    Test the partial contact fraction expectation code for a 2 state system.
    """

    output_directory = tmpdir.mkdir("output")

    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
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
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5

    # Set production frames:
    frame_begin = 100

    # Get native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )

    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=frame_begin,
        native_contact_tol=native_contact_tol,
    )
    
    # Set 1 as folded, 0 as unfolded:
    Q_folded = 0.3
    array_folded_states = np.multiply((Q>=Q_folded),1)        
    
    output_data = os.path.join(data_path, "output.nc")
    num_intermediate_states=1

    Q_values_partial, T_list_out = expectations_partial_contact_fractions(
        array_folded_states,
        Q,
        frame_begin=100,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
    )
    
    assert len(Q_values_partial) == 2 and type(Q_values_partial) == dict
    
    
def test_expectations_fraction_contacts_dcd_homopolymer_sym_linear(tmpdir):
    """
    See if we can determine native contacts expectations as a function of T,
    with homopolymer end-to-end symmetry checks.
    (linear homopolymer with no sidechains)
    """

    output_directory = tmpdir.mkdir("output")
    
    data_path_linear = os.path.join(data_path, 'linear_24mer')
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path_linear}/stored_cgmodel_24mer_linear.pkl", "rb" ))
    
    # Create list of dcd trajectories to analyze
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path_linear}/replica_{i+1}.dcd")
        
    # Set path to native structure file:    
    native_structure_file=f"{data_path_linear}/native_medoid_min.dcd"
    
    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )

    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=100,
        native_contact_tol=native_contact_tol,
        homopolymer_sym=True,
    )
    
    output_data = os.path.join(data_path_linear, "output.nc")
    num_intermediate_states=1

    results = expectations_fraction_contacts(
        Q,
        frame_begin=100,
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
    )
    

def test_bootstrap_native_contacts_expectation_dcd(tmpdir):
    """Test bootstrapping of native contacts expectation, based on helix contacts"""
    
    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of pdb trajectories to analyze
    # For fraction_native_contacts vs. T, we use state trajectories.
    # However, we can test with the replica pdbs:
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
        cgmodel,
        native_structure_file,
        backbone_type_name='bb',
    )
    
    full_T_list, Q_values, Q_uncertainty, sigmoid_results_boot = bootstrap_native_contacts_expectation(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        output_data=output_data,
        frame_begin=100,
        sample_spacing=20,
        native_contact_tol=1.2,
        num_intermediate_states=1,
        n_trial_boot=10,
        conf_percent='sigma',
        plotfile=f'{output_directory}/native_contacts_boot.pdf',
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_boot.pdf')  
    
    
def test_bootstrap_partial_contacts_expectation_dcd_2state(tmpdir):
    """Test bootstrapping of partial native contacts expectation, based on helix contacts"""
    
    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of pdb trajectories to analyze
    # For fraction_native_contacts vs. T, we use state trajectories.
    # However, we can test with the replica pdbs:
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"
    
    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5    
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )
    
    frame_begin = 100
    
    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=frame_begin,
        native_contact_tol=native_contact_tol,
    )
    
    # Set 1 as folded, 0 as unfolded:
    Q_folded = 0.3
    array_folded_states = np.multiply((Q>=Q_folded),1)    
    
    full_T_list, Q_values, Q_uncertainty, sigmoid_results_boot = bootstrap_partial_contacts_expectation(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        array_folded_states,
        output_data=output_data,
        frame_begin=frame_begin,
        sample_spacing=20,
        native_contact_tol=native_contact_tol,
        num_intermediate_states=1,
        n_trial_boot=10,
        conf_percent='sigma',
        plotfile=f'{output_directory}/partial_contacts_boot.pdf',
        homopolymer_sym=False,
        )
        
    assert os.path.isfile(f'{output_directory}/partial_contacts_boot.pdf')
    

def test_bootstrap_partial_contacts_expectation_dcd_2state_homopolymer_sym(tmpdir):
    """
    Test bootstrapping of partial native contacts expectation, based on helix contacts
    (with homopolymer symmetry check)
    """
    
    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of pdb trajectories to analyze
    # For fraction_native_contacts vs. T, we use state trajectories.
    # However, we can test with the replica pdbs:
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"
    
    # Set cutoff parameters:
    # Cutoff for native structure pairwise distances:
    native_contact_cutoff = 4.0* unit.angstrom

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5    
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, contact_type_dict = get_native_contacts(
        cgmodel,
        native_structure_file,
        native_contact_cutoff,
    )
    
    frame_begin = 100
    
    Q, Q_avg, Q_stderr, decorrelation_spacing = fraction_native_contacts(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        frame_begin=frame_begin,
        native_contact_tol=native_contact_tol,
    )
    
    # Set 1 as folded, 0 as unfolded:
    Q_folded = 0.3
    array_folded_states = np.multiply((Q>=Q_folded),1)    
    
    full_T_list, Q_values, Q_uncertainty, sigmoid_results_boot = bootstrap_partial_contacts_expectation(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        array_folded_states,
        output_data=output_data,
        frame_begin=frame_begin,
        sample_spacing=20,
        native_contact_tol=native_contact_tol,
        num_intermediate_states=1,
        n_trial_boot=10,
        conf_percent='sigma',
        plotfile=f'{output_directory}/partial_contacts_boot.pdf',
        homopolymer_sym=True,
        )
        
    assert os.path.isfile(f'{output_directory}/partial_contacts_boot.pdf')       
    
    
def test_bootstrap_native_contacts_expectation_dcd_homopolymer_sym_linear(tmpdir):
    """
    Test bootstrapping of native contacts expectation, based on helix contacts,
    with homopolymer end-to-end symmetry checks.
    (linear homopolymer with no sidechains)
    """
    
    output_directory = tmpdir.mkdir("output")
    
    data_path_linear = os.path.join(data_path, 'linear_24mer')
    output_data = os.path.join(data_path_linear, "output.nc")
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path_linear}/stored_cgmodel_24mer_linear.pkl", "rb" ))
    
    # Create list of dcd trajectories to analyze
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path_linear}/replica_{i+1}.dcd")
        
    # Set path to native structure file:    
    native_structure_file=f"{data_path_linear}/native_medoid_min.dcd"

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5    
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
        cgmodel,
        native_structure_file,
        backbone_type_name='bb',
    )
    
    full_T_list, Q_values, Q_uncertainty, sigmoid_results_boot = bootstrap_native_contacts_expectation(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        output_data=output_data,
        frame_begin=100,
        sample_spacing=20,
        native_contact_tol=1.2,
        num_intermediate_states=1,
        n_trial_boot=10,
        conf_percent='sigma',
        plotfile=f'{output_directory}/native_contacts_boot_sym.pdf',
        homopolymer_sym=True,
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_boot_sym.pdf')      
    
    
def test_bootstrap_native_contacts_expectation_dcd_homopolymer_sym_sidechain(tmpdir):
    """
    Test bootstrapping of native contacts expectation, based on helix contacts,
    with homopolymer end-to-end symmetry checks.
    (1-1 homopolymer model with sidechains)
    """
    
    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")
    
    # Replica exchange settings
    number_replicas = 12
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of dcd trajectories to analyze
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Set path to native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"

    # Tolerance for current trajectory distances:
    native_contact_tol = 1.5    
    
    # Determine native contacts:
    native_contact_list, native_contact_distances, opt_seq_spacing = get_helix_contacts(
        cgmodel,
        native_structure_file,
        backbone_type_name='bb',
    )
    
    full_T_list, Q_values, Q_uncertainty, sigmoid_results_boot = bootstrap_native_contacts_expectation(
        cgmodel,
        dcd_file_list,
        native_contact_list,
        native_contact_distances,
        output_data=output_data,
        frame_begin=100,
        sample_spacing=20,
        native_contact_tol=1.2,
        num_intermediate_states=1,
        n_trial_boot=10,
        conf_percent='sigma',
        plotfile=f'{output_directory}/native_contacts_boot_sym.pdf',
        homopolymer_sym=True,
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_boot_sym.pdf')          
    
    
def test_optimize_Q_helix_tol_dcd(tmpdir):
    """Test the helix native contact tolerance optimization workflow"""

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")

    # Replica exchange settings
    number_replicas = 12

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"    
    
    (opt_seq_spacing, native_contact_tol, opt_results, Q_expect_results,
    sigmoid_param_opt, sigmoid_param_cov) = optimize_Q_tol_helix(
        cgmodel,
        native_structure_file,
        dcd_file_list,
        num_intermediate_states=0,
        output_data=output_data,
        frame_begin=100,
        frame_stride=100,
        verbose=True,
        plotfile=f'{output_directory}/native_contacts_helix_opt.pdf',
        backbone_type_name='bb',
        brute_step=0.2,
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_helix_opt.pdf')     
    
    
def test_optimize_Q_cut_1d_dcd(tmpdir):
    """
    Test the native contact cutoff 1d optimization workflow (fixed tolerance factor)
    Bounds determined automatically by the backbone sigma parameter.
    """

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")

    # Replica exchange settings
    number_replicas = 12

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"    
    
    (native_contact_cutoff, opt_results, Q_expect_results, \
    sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut_1d(
        cgmodel,
        native_structure_file,
        dcd_file_list,
        num_intermediate_states=0,
        output_data=output_data,
        native_contact_tol=1.3,
        frame_begin=100,
        frame_stride=100,
        verbose=True,
        plotfile=f'{output_directory}/native_contacts_opt_1d.pdf',
        brute_step=0.2,
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_opt_1d.pdf')
    
    
def test_optimize_Q_cut_1d_dcd_bounds_1(tmpdir):
    """
    Test the native contact cutoff 1d optimization workflow (fixed tolerance factor)
    Bounds specified as quantity with tuple value
    """

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")

    # Replica exchange settings
    number_replicas = 12

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"    
    
    (native_contact_cutoff, opt_results, Q_expect_results, \
    sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut_1d(
        cgmodel,
        native_structure_file,
        dcd_file_list,
        num_intermediate_states=0,
        output_data=output_data,
        native_contact_tol=1.3,
        frame_begin=100,
        frame_stride=100,
        verbose=True,
        plotfile=f'{output_directory}/native_contacts_opt_1d.pdf',
        brute_step=0.2,
        bounds=(1.0,4.0)*unit.angstrom,
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_opt_1d.pdf')


def test_optimize_Q_cut_1d_dcd_bounds_2(tmpdir):
    """
    Test the native contact cutoff 1d optimization workflow (fixed tolerance factor)
    Bounds specified as tuple of quantities
    """

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")

    # Replica exchange settings
    number_replicas = 12

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"    
    
    (native_contact_cutoff, opt_results, Q_expect_results, \
    sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut_1d(
        cgmodel,
        native_structure_file,
        dcd_file_list,
        num_intermediate_states=0,
        output_data=output_data,
        native_contact_tol=1.3,
        frame_begin=100,
        frame_stride=100,
        verbose=True,
        plotfile=f'{output_directory}/native_contacts_opt_1d.pdf',
        brute_step=0.2,
        bounds=(1.0*unit.angstrom,4.0*unit.angstrom)
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_opt_1d.pdf')     
    
    
def test_optimize_Q_cut_pdb(tmpdir):
    """Test the native contact cutoff / tolerance 2d optimization workflow"""

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")

    # Replica exchange settings
    number_replicas = 12

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    pdb_file_list = []
    for i in range(number_replicas):
        pdb_file_list.append(f"{data_path}/replica_{i+1}.pdb")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.pdb"    
    
    (native_contact_cutoff, native_contact_tol, opt_results, Q_expect_results, \
    sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut(
        cgmodel,
        native_structure_file,
        pdb_file_list,
        num_intermediate_states=0,
        output_data=output_data,
        frame_begin=100,
        frame_stride=100,
        verbose=True,
        plotfile=f'{output_directory}/native_contacts_opt.pdf',
        minimizer_options={'seed':17, 'maxiter':3, 'atol':0.5},
        )   
        
    assert os.path.isfile(f'{output_directory}/native_contacts_opt.pdf')
    
    
def test_optimize_Q_cut_dcd(tmpdir):
    """Test the native contact cutoff / tolerance 2d optimization workflow"""

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")

    # Replica exchange settings
    number_replicas = 12

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"    
    
    (native_contact_cutoff, native_contact_tol, opt_results, Q_expect_results, \
    sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut(
        cgmodel,
        native_structure_file,
        dcd_file_list,
        num_intermediate_states=0,
        output_data=output_data,
        frame_begin=100,
        frame_stride=100,
        verbose=True,
        plotfile=f'{output_directory}/native_contacts_opt.pdf',
        minimizer_options={'seed':17, 'maxiter':3, 'atol':0.5},
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_opt.pdf')
    

def test_optimize_Q_cut_dcd_bounds_1(tmpdir):
    """
    Test the native contact cutoff / tolerance 2d optimization workflow
    Custom bounds specified for both native contact cutoff (quantity with tuple value)
    and native contact tolerance factor (tuple).
    """

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")

    # Replica exchange settings
    number_replicas = 12

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"    
    
    (native_contact_cutoff, native_contact_tol, opt_results, Q_expect_results, \
    sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut(
        cgmodel,
        native_structure_file,
        dcd_file_list,
        num_intermediate_states=0,
        output_data=output_data,
        frame_begin=100,
        frame_stride=200,
        verbose=True,
        plotfile=f'{output_directory}/native_contacts_opt.pdf',
        minimizer_options={'seed':17, 'maxiter':3, 'atol':0.2},
        bounds_nc_cut=(2.0,4.0)*unit.angstrom,
        bounds_nc_tol=(1.05,1.50),
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_opt.pdf')
    
    
def test_optimize_Q_cut_dcd_bounds_2(tmpdir):
    """
    Test the native contact cutoff / tolerance 2d optimization workflow
    Custom bounds specified for both native contact cutoff (tuple of quantities)
    and native contact tolerance factor (tuple).
    """

    output_directory = tmpdir.mkdir("output")
    output_data = os.path.join(data_path, "output.nc")

    # Replica exchange settings
    number_replicas = 12

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of pdb trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
        
    # Load in native structure file:    
    native_structure_file=f"{structures_path}/medoid_0.dcd"    
    
    (native_contact_cutoff, native_contact_tol, opt_results, Q_expect_results, \
    sigmoid_param_opt, sigmoid_param_cov, contact_type_dict) = optimize_Q_cut(
        cgmodel,
        native_structure_file,
        dcd_file_list,
        num_intermediate_states=0,
        output_data=output_data,
        frame_begin=100,
        frame_stride=200,
        verbose=True,
        plotfile=f'{output_directory}/native_contacts_opt.pdf',
        minimizer_options={'seed':17, 'maxiter':3, 'atol':0.2},
        bounds_nc_cut=(2.0*unit.angstrom,4.0*unit.angstrom),
        bounds_nc_tol=(1.05,1.50),
        )
        
    assert os.path.isfile(f'{output_directory}/native_contacts_opt.pdf')
    
    
def test_Rg_vs_T_sigmoid_fit(tmpdir):
    """
    Test the sigmoid fitting utility on example radius of gyration vs temperature
    data for a coil-to-globule transition.
    """
    
    output_directory = tmpdir.mkdir("output")

    Rg_avg = [6.08747972, 6.16440602, 6.26975632, 6.43288091, 6.75841863, 7.702573, \
        9.65572705, 12.64721094, 16.11618585, 18.66737671, 19.40515363] * unit.angstrom
    T_list = [178.73568219, 205.55173196, 229.38841419, 255.93752622, 283.592048, \
        304.85701798, 319.47800537, 332.8463338, 352.91094873, 413.22849487, 550.] * unit.kelvin

    param_opt, param_cov = fit_sigmoid(
        T_list,
        Rg_avg,
        plotfile=f'{output_directory}/rg_vs_T_sigmoid_fit.pdf',
        xlabel=f'T (K)',
        ylabel=f'$R_g$ (A)',
        )
        
    # param_opt ( 1D numpy array ) - optimized sigmoid parameters (x0, y0, y1, d)    
    # param_cov ( 2D numpy array ) - estimated covariance of param_opt
    
    assert os.path.isfile(f'{output_directory}/rg_vs_T_sigmoid_fit.pdf')
    
    
def test_Rg_vs_T_sigmoid_fit_no_units(tmpdir):
    """
    Test the sigmoid fitting utility on example radius of gyration vs temperature
    data for a coil-to-globule transition.
    (no units on data, such as when using a reduced unit system)
    """
    
    output_directory = tmpdir.mkdir("output")

    Rg_avg = [6.08747972, 6.16440602, 6.26975632, 6.43288091, 6.75841863, 7.702573, \
        9.65572705, 12.64721094, 16.11618585, 18.66737671, 19.40515363]
    T_list = [178.73568219, 205.55173196, 229.38841419, 255.93752622, 283.592048, \
        304.85701798, 319.47800537, 332.8463338, 352.91094873, 413.22849487, 550.]

    param_opt, param_cov = fit_sigmoid(
        T_list,
        Rg_avg,
        plotfile=f'{output_directory}/rg_vs_T_sigmoid_fit.pdf',
        xlabel=f'T*',
        ylabel=f'$R_g$*',
        )
        
    # param_opt ( 1D numpy array ) - optimized sigmoid parameters (x0, y0, y1, d)    
    # param_cov ( 2D numpy array ) - estimated covariance of param_opt
    
    assert os.path.isfile(f'{output_directory}/rg_vs_T_sigmoid_fit.pdf')    
    