"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed  
  
import os
import pickle

from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import (get_opt_temperature_list,
                                           get_temperature_list)
from cg_openmm.parameters.secondary_structure import *
from cg_openmm.simulation.physical_validation import *
from cg_openmm.thermo.calc import *
from numpy.testing import assert_allclose, assert_almost_equal
from openmm import unit

current_path = os.path.dirname(os.path.abspath(__file__))
structures_path = os.path.join(current_path, 'test_structures')
data_path = os.path.join(current_path, 'test_data')
       
    
def test_heat_capacity_calc(tmpdir):  
    """ Test heat capacity calculation"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path, "output.nc")
    
    # number_replicas=12
    # min_temp = 200.0 * unit.kelvin
    # max_temp = 600.0 * unit.kelvin
    
    # 1) With default starting frame (0)
    C_v, dC_v, new_temperature_list, FWHM, Tm, Cv_height, N_eff = get_heat_capacity(
        output_data=output_data,
        num_intermediate_states=2,
        plot_file=f"{plot_directory}/heat_capacity.pdf"
    )
    
    assert os.path.isfile(f"{plot_directory}/heat_capacity.pdf")
    
    # 2) With a non-default starting frame:
    C_v, dC_v, new_temperature_list, FWHM, Tm, Cv_height, N_eff = get_heat_capacity(
        frame_begin=2,
        output_data=output_data,
        num_intermediate_states=2,
        plot_file=f"{plot_directory}/heat_capacity2.pdf"
    )
    
    assert os.path.isfile(f"{plot_directory}/heat_capacity2.pdf")
    
    # Test heat capacity spline fitting / derivative calculation
    derC_v, der2C_v, spline_tck = get_heat_capacity_derivative(
        C_v,
        new_temperature_list,
        plotfile=f"{plot_directory}/dCv_dT.pdf")
        
    assert os.path.isfile(f"{plot_directory}/dCv_dT.pdf")
    
    # Test constant entropy increase temperature spacing optimization
    opt_temperature_list, deltaS_list = get_opt_temperature_list(
        new_temperature_list, C_v,
        number_intermediate_states=2,
        verbose=True)
            
    assert_almost_equal(
        opt_temperature_list[-1].value_in_unit(unit.kelvin),
        new_temperature_list[-1].value_in_unit(unit.kelvin),
        decimal=6
        )
        
    assert_allclose(deltaS_list.value_in_unit(C_v.unit)/np.max(deltaS_list.value_in_unit(C_v.unit)),np.ones(len(deltaS_list)))
        
    
def test_bootstrap_heat_capacity_conf(tmpdir):  
    """Test heat capacity bootstrapping calculation with confidence intervals"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path, "output.nc")
    
    (new_temperature_list, C_v_values, C_v_uncertainty, Tm_value, Tm_uncertainty, 
    Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty, N_eff) = bootstrap_heat_capacity(
        output_data=output_data,
        num_intermediate_states=1,
        frame_begin=2,
        sample_spacing=4,
        n_trial_boot=10,
        plot_file=f"{plot_directory}/heat_capacity_boot.pdf",
        conf_percent=80,
        )
        
    assert FWHM_value.value_in_unit(unit.kelvin) > 0.0
    assert os.path.isfile(f"{plot_directory}/heat_capacity_boot.pdf")
    
    
def test_bootstrap_heat_capacity_sigma(tmpdir):  
    """Test heat capacity bootstrapping calculation with analytical standard deviation"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path, "output.nc")
    
    (new_temperature_list, C_v_values, C_v_uncertainty, Tm_value, Tm_uncertainty, 
    Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty, N_eff) = bootstrap_heat_capacity(
        output_data=output_data,
        num_intermediate_states=1,
        frame_begin=2,
        sample_spacing=4,
        n_trial_boot=10,
        plot_file=f"{plot_directory}/heat_capacity_boot.pdf",
        conf_percent='sigma',
        )
        
    assert FWHM_value.value_in_unit(unit.kelvin) > 0.0
    assert os.path.isfile(f"{plot_directory}/heat_capacity_boot.pdf")
    
    
def test_partial_heat_capacity_2state(tmpdir):
    """Test partial heat capacity by conformational state (unfolded/folded)"""
    
    # First we need to classify each production frame as folded/unfolded
    
    output_directory = tmpdir.mkdir("output")
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path,"output.nc")

    # Replica exchange settings
    number_replicas = 12
    #min_temp = 200.0 * unit.kelvin
    #max_temp = 600.0 * unit.kelvin
    #temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of dcd trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
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
    
    (Cv_partial, dCv_partial, temperature_list, \
    FWHM_partial, Tm_partial, Cv_height_partial, U_expect_confs) = get_partial_heat_capacities(
        array_folded_states=array_folded_states,
        frame_begin=frame_begin,
        sample_spacing=1,
        frame_end=-1,
        output_data=output_data,
        num_intermediate_states=1,
        frac_dT=0.05,
        plot_file=f"{plot_directory}/partial_heat_capacity.pdf",
        )
        
    assert os.path.isfile(f"{plot_directory}/partial_heat_capacity.pdf")    
    
    # Check data types of all output:
    for varname in [
        'Cv_partial','FWHM_partial','Tm_partial','Cv_height_partial','U_expect_confs'
        ]:
        
        assert eval(f'type({varname})') == dict
        assert eval(f'len({varname})') == 2    
    
    
def test_bootstrap_partial_heat_capacity_2state_sigma(tmpdir):
    """
    Test partial heat capacity boostrapping calculation by conformational state 
    (unfolded/folded), with analytical standard deviation.
    """    
    
    # First we need to classify each production frame as folded/unfolded
    
    output_directory = tmpdir.mkdir("output")
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path,"output.nc")

    # Replica exchange settings
    number_replicas = 12
    #min_temp = 200.0 * unit.kelvin
    #max_temp = 600.0 * unit.kelvin
    #temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of dcd trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
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
    
    (T_list, Cv_values, Cv_uncertainty, Tm_value, Tm_uncertainty, \
    Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty, \
    U_expect_values, U_expect_uncertainty) = bootstrap_partial_heat_capacities(
        array_folded_states=array_folded_states,
        frame_begin=frame_begin,
        sample_spacing=1,
        frame_end=-1,
        output_data=output_data,
        num_intermediate_states=1,
        frac_dT=0.05,
        plot_file=f"{plot_directory}/partial_heat_capacity_boot.pdf",
        conf_percent='sigma',
        n_trial_boot=20,
        )
        
    assert os.path.isfile(f"{plot_directory}/partial_heat_capacity_boot.pdf")    
    
    # Check data types of all output:
    for varname in ['Cv_values','Cv_uncertainty',
        'Tm_value','Tm_uncertainty','Cv_height_value','Cv_height_uncertainty',
        'FWHM_value','FWHM_uncertainty','U_expect_values','U_expect_uncertainty'
        ]:
        
        assert eval(f'type({varname})') == dict
        assert eval(f'len({varname})') == 2
        
        
def test_bootstrap_partial_heat_capacity_2state_conf(tmpdir):
    """
    Test partial heat capacity boostrapping calculation by conformational state 
    (unfolded/folded), with confidence intervals.
    """    
    
    # First we need to classify each production frame as folded/unfolded
    
    output_directory = tmpdir.mkdir("output")
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path,"output.nc")

    # Replica exchange settings
    number_replicas = 12
    #min_temp = 200.0 * unit.kelvin
    #max_temp = 600.0 * unit.kelvin
    #temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))

    # Create list of dcd trajectories to analyze
    # For expectation fraction native contacts, we use replica trajectories: 
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
    
    (T_list, Cv_values, Cv_uncertainty, Tm_value, Tm_uncertainty, \
    Cv_height_value, Cv_height_uncertainty, FWHM_value, FWHM_uncertainty, \
    U_expect_values, U_expect_uncertainty) = bootstrap_partial_heat_capacities(
        array_folded_states=array_folded_states,
        frame_begin=frame_begin,
        sample_spacing=2,
        frame_end=-1,
        output_data=output_data,
        num_intermediate_states=1,
        frac_dT=0.05,
        plot_file=f"{plot_directory}/partial_heat_capacity_boot.pdf",
        conf_percent=90,
        n_trial_boot=20,
        )
        
    assert os.path.isfile(f"{plot_directory}/partial_heat_capacity_boot.pdf")    
    
    # Check data types of all output:
    for varname in ['Cv_values','Cv_uncertainty',
        'Tm_value','Tm_uncertainty','Cv_height_value','Cv_height_uncertainty',
        'FWHM_value','FWHM_uncertainty','U_expect_values','U_expect_uncertainty'
        ]:
        
        assert eval(f'type({varname})') == dict
        assert eval(f'len({varname})') == 2        


def test_physical_validation_1(tmpdir):
    """Test physical validation ensemble check"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path,"output.nc")
    
    # 1) Pair option: 'single'
    quantiles = physical_validation_ensemble(
        output_data=output_data,
        output_directory=data_path,
        plotfile=f"{plot_directory}ensemble_check",
        pairs='single',
        ref_state_index=1,
    )

def test_physical_validation_2(tmpdir):
    """Test physical validation ensemble check"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path,"output.nc")    
    
    # 2) Pair option: 'adjacent'
    quantiles = physical_validation_ensemble(
        output_data=output_data,
        output_directory=data_path,
        plotfile=f"{plot_directory}ensemble_check",
        pairs='adjacent',
    )
    
    
def test_physical_validation_3(tmpdir):
    """Test physical validation ensemble check"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path,"output.nc")    
    
    # 3) Pair option: 'all'
    quantiles = physical_validation_ensemble(
        output_data=output_data,
        output_directory=data_path,
        plotfile=f"{plot_directory}ensemble_check",
        pairs='all',
    )
    

def test_physical_validation_4(tmpdir):
    """Test physical validation ensemble check"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path,"output.nc")        
    
    # 4) Pair option: default
    quantiles = physical_validation_ensemble(
        output_data=output_data,
        output_directory=data_path,
        plotfile=f"{plot_directory}ensemble_check",
    )
    
    
def test_physical_validation_5(tmpdir):
    """Test physical validation ensemble check"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path,"output.nc")        
    
    # 5) Pair option: default from bad pair style
    quantiles = physical_validation_ensemble(
        output_data=output_data,
        output_directory=data_path,
        plotfile=f"{plot_directory}ensemble_check",
        pairs='invalid',
    )
    
