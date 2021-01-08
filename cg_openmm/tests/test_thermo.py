"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed  
  
import os
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.thermo.calc import * 
from cg_openmm.parameters.reweight import get_temperature_list, get_opt_temperature_list
from cg_openmm.simulation.physical_validation import *
from numpy.testing import assert_almost_equal    
    
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_data')
       
    
def test_heat_capacity_calc(tmpdir):  
    """ Test heat capacity calculation"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path, "output.nc")
    
    # number_replicas=12
    # min_temp = 200.0 * unit.kelvin
    # max_temp = 300.0 * unit.kelvin
    
    # 1) With default starting frame (0)
    C_v, dC_v, new_temperature_list = get_heat_capacity(
        output_data=output_data,
        num_intermediate_states=2,
        plot_file=f"{plot_directory}/heat_capacity.pdf"
    )
    
    assert os.path.isfile(f"{plot_directory}/heat_capacity.pdf")
    
    # 2) With a non-default starting frame:
    C_v, dC_v, new_temperature_list = get_heat_capacity(
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
    opt_temperature_list = get_opt_temperature_list(
        new_temperature_list, C_v,
        number_intermediate_states=2,
        verbose=True)
            
    assert_almost_equal(
        opt_temperature_list[-1].value_in_unit(unit.kelvin),
        new_temperature_list[-1].value_in_unit(unit.kelvin),
        decimal=6
        )
    

def test_physical_validation(tmpdir):
    """Test physical validation ensemble check"""
    
    plot_directory = tmpdir.mkdir("plot_output")
    output_data = os.path.join(data_path, "output.nc")
    
    # 1) Pair option: 'single'
    quantiles = physical_validation_ensemble(
        output_data=output_data,
        output_directory=data_path,
        plotfile=f"{plot_directory}ensemble_check",
        pairs='single',
        ref_state_index=0
    )
    
    # ***2) and 3) leading to segfault 
    # 2) Pair option: 'adjacent'
    # quantiles = physical_validation_ensemble(
        # output_data=output_data,
        # output_directory=data_path,
        # plotfile=f"{plot_directory}ensemble_check",
        # pairs='adjacent',
    # )
    
    # # 3) Pair option: 'all'
    # quantiles = physical_validation_ensemble(
        # output_data=output_data,
        # output_directory=data_path,
        # plotfile=f"{plot_directory}ensemble_check",
        # pairs='all',
    # )
    
    # 4) Pair option: default
    # quantiles = physical_validation_ensemble(
        # output_data=output_data,
        # output_directory=data_path,
        # plotfile=f"{plot_directory}ensemble_check",
    # )
    
    # 5) Pair option: default from bad pair style
    # quantiles = physical_validation_ensemble(
        # output_data=output_data,
        # output_directory=data_path,
        # plotfile=f"{plot_directory}ensemble_check",
        # pairs='invalid',
    # )
    
    
    