"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed  
  
import copy
import os
import pickle

from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.evaluate_energy import *
from cg_openmm.parameters.reweight import (get_opt_temperature_list,
                                           get_temperature_list)
from cg_openmm.thermo.calc import *
from numpy.testing import assert_allclose, assert_almost_equal
from openmm import unit
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeAnalyzer

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_data')
structures_path = os.path.join(current_path, 'test_structures')

def test_energy_decomposition_dcd(tmpdir):  
    """
    Test the energy decomposition on a cgmodel and medoid, and check that the individual components
    sum to the total potential energy.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Set path to medoid structure file:
    medoid_file = f"{structures_path}/medoid_min.dcd"
    
    # Run the energy decomposition:
    U_decomposition = energy_decomposition(
        cgmodel,
        medoid_file,
    )
    
    # Check the sum of energy components:
    sum_expected = U_decomposition['total']
    sum_actual = 0 * unit.kilojoule_per_mole
    
    print(U_decomposition)
    
    for key,value in U_decomposition.items():
        if key != 'total':
            sum_actual += value
    
    assert sum_expected == sum_actual

    
def test_eval_energy_no_change(tmpdir):  
    """
    Make sure the reevaluated energies are the same as those in the .nc file,
    when no parameters are changed.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_epsilon'] = 1.5 * unit.kilojoule_per_mole # Was 1.5 kJ/mol previously
    
    frame_begin = 100
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=frame_begin,
        frame_end=-1,
        frame_stride=1,
        verbose=True,
    )
    
    # Read in original simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    reporter = MultiStateReporter(output_data, open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)
    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()    
    
    # Rounding error with stored positions and/or energies is on the order of 1E-4
    assert_allclose(U_eval,replica_energies[:,:,frame_begin::],atol=1E-3)
    
    
def test_eval_energy_no_change_parallel(tmpdir):  
    """
    Make sure the reevaluated energies are the same as those in the .nc file,
    when no parameters are changed (running energy evaluations in parallel).
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_epsilon'] = 1.5 * unit.kilojoule_per_mole # Was 1.5 kJ/mol previously
    
    frame_begin = 100
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=frame_begin,
        frame_end=-1,
        frame_stride=1,
        verbose=True,
        n_cpu=2,
    )
    
    # Read in original simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    reporter = MultiStateReporter(output_data, open_mode="r")
    analyzer = ReplicaExchangeAnalyzer(reporter)
    (
        replica_energies,
        unsampled_state_energies,
        neighborhoods,
        replica_state_indices,
    ) = analyzer.read_energies()    
    
    # Rounding error with stored positions and/or energies is on the order of 1E-4
    assert_allclose(U_eval,replica_energies[:,:,frame_begin::],atol=1E-3)
    
    
def test_eval_energy_new_sigma(tmpdir):  
    """
    Test simulation parameter update for varying LJ sigma.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_sigma'] = 2.50 * unit.angstrom
    param_dict['sc_sigma'] = 4.50 * unit.angstrom
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            (q,sigma_bb_updated,eps) = force.getParticleParameters(0)
            (q,sigma_sc_updated,eps) = force.getParticleParameters(1)
        
    assert sigma_bb_updated == param_dict['bb_sigma']
    assert sigma_sc_updated == param_dict['sc_sigma']
    
    
def test_eval_energy_new_epsilon(tmpdir):  
    """
    Test simulation parameter update for varying LJ epsilon.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_epsilon'] = 2.25 * unit.kilojoule_per_mole
    param_dict['sc_epsilon'] = 5.25 * unit.kilojoule_per_mole
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'NonbondedForce':
            (q,sigma,epsilon_bb_updated) = force.getParticleParameters(0)
            (q,sigma,epsilon_sc_updated) = force.getParticleParameters(1)
        
    assert epsilon_bb_updated == param_dict['bb_epsilon']
    assert epsilon_sc_updated == param_dict['sc_epsilon']
    
    
def test_eval_energy_new_bond_length(tmpdir):  
    """
    Test simulation parameter update for varying bond lengths.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_bb_bond_length'] = 2.25 * unit.angstrom
    param_dict['bb_sc_bond_length'] = 2.50 * unit.angstrom
    
    # Bond index 0 is type bb-sc
    # Bond index 1 is type bb-bb
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'HarmonicBondForce':
            (par1, par2, bb_sc_length_updated, k) = force.getBondParameters(0)
            (par1, par2, bb_bb_length_updated, k) = force.getBondParameters(1)
            
    assert bb_bb_length_updated == param_dict['bb_bb_bond_length']
    assert bb_sc_length_updated == param_dict['bb_sc_bond_length']
    
    # Now, try the reverse names:
    # Set up dictionary of parameter change instructions:
    param_dict_rev = {}
    param_dict_rev['sc_bb_bond_length'] = 2.501 * unit.angstrom
    
    # Re-evaluate OpenMM energies:
    # Clear the simulation object to reset the parameters
    del simulation
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict_rev,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'HarmonicBondForce':
            (par1, par2, sc_bb_length_updated, k) = force.getBondParameters(0)
            
    assert sc_bb_length_updated == param_dict_rev['sc_bb_bond_length']
    
    
def test_eval_energy_new_bond_k(tmpdir):  
    """
    Test simulation parameter update for varying bond stiffness.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_bb_bond_force_constant'] = 12000 * unit.kilojoule_per_mole / unit.nanometer**2
    param_dict['bb_sc_bond_force_constant'] = 14000 * unit.kilojoule_per_mole / unit.nanometer**2
    
    # Bond index 0 is type bb-sc
    # Bond index 1 is type bb-bb
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'HarmonicBondForce':
            (par1, par2, length, bb_sc_k_updated) = force.getBondParameters(0)
            (par1, par2, length, bb_bb_k_updated) = force.getBondParameters(1)
            
    assert bb_bb_k_updated == param_dict['bb_bb_bond_force_constant']
    assert bb_sc_k_updated == param_dict['bb_sc_bond_force_constant']
    
    # Now, try the reverse names:
    # Set up dictionary of parameter change instructions:
    param_dict_rev = {}
    param_dict_rev['sc_bb_bond_force_constant'] = 14001 * unit.kilojoule_per_mole / unit.nanometer**2
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict_rev,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'HarmonicBondForce':
            (par1, par2, length, sc_bb_k_updated) = force.getBondParameters(0)
            
    assert sc_bb_k_updated == param_dict_rev['sc_bb_bond_force_constant']
    
    
def test_eval_energy_new_angle_val(tmpdir):  
    """
    Test simulation parameter update for varying equilibrium bond angle.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    # Note: some rounding error possible if we specify angles in degrees, and assert against radians
    param_dict = {}
    param_dict['bb_bb_bb_equil_bond_angle'] = (130 * unit.degrees).in_units_of(unit.radians)
    param_dict['bb_bb_sc_equil_bond_angle'] = ( 95 * unit.degrees).in_units_of(unit.radians)
    
    # Angle index 0 is type sc-bb-bb
    # Angle index 2 is type bb-bb-bb
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'HarmonicAngleForce':
            (par1, par2, par3, bb_bb_sc_angle_updated, k) = force.getAngleParameters(0)
            (par1, par2, par3, bb_bb_bb_angle_updated, k) = force.getAngleParameters(2)
            
    assert bb_bb_bb_angle_updated == param_dict['bb_bb_bb_equil_bond_angle']
    assert bb_bb_sc_angle_updated == param_dict['bb_bb_sc_equil_bond_angle']
    
    # Now, try the reverse names:
    # Set up dictionary of parameter change instructions:
    param_dict_rev = {}
    param_dict_rev['sc_bb_bb_equil_bond_angle'] = (96 * unit.degrees).in_units_of(unit.radians)
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict_rev,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'HarmonicAngleForce':
            (par1, par2, par3, sc_bb_bb_angle_updated, k) = force.getAngleParameters(0)
            
    assert sc_bb_bb_angle_updated == param_dict_rev['sc_bb_bb_equil_bond_angle']
    
    
def test_eval_energy_new_angle_k(tmpdir):  
    """
    Test simulation parameter update for varying bond angle stiffness.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_bb_bb_bond_angle_force_constant'] = 200 * unit.kilojoule_per_mole / unit.radian**2
    param_dict['bb_bb_sc_bond_angle_force_constant'] = 175 * unit.kilojoule_per_mole / unit.radian**2
    
    # Angle index 0 is type sc-bb-bb
    # Angle index 2 is type bb-bb-bb
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'HarmonicAngleForce':
            (par1, par2, par3, angle, bb_bb_sc_k_updated) = force.getAngleParameters(0)
            (par1, par2, par3, angle, bb_bb_bb_k_updated) = force.getAngleParameters(2)
            
    assert bb_bb_bb_k_updated == param_dict['bb_bb_bb_bond_angle_force_constant']
    assert bb_bb_sc_k_updated == param_dict['bb_bb_sc_bond_angle_force_constant']
    
    # Now, try the reverse names:
    # Set up dictionary of parameter change instructions:
    param_dict_rev = {}
    param_dict_rev['sc_bb_bb_bond_angle_force_constant'] = 176 * unit.kilojoule_per_mole / unit.radian**2
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict_rev,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'HarmonicAngleForce':
            (par1, par2, par3, angle, sc_bb_bb_k_updated) = force.getAngleParameters(0)
            
    assert sc_bb_bb_k_updated == param_dict_rev['sc_bb_bb_bond_angle_force_constant']
    
    
def test_eval_energy_new_torsion_val(tmpdir):  
    """
    Test simulation parameter update for varying torsion phase angle.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    # Note: some rounding error possible if we specify angles in degrees, and assert against radians
    param_dict = {}
    param_dict['bb_bb_bb_bb_torsion_phase_angle'] = (-165 * unit.degrees).in_units_of(unit.radians)
    param_dict['bb_bb_bb_sc_torsion_phase_angle'] = (  15 * unit.degrees).in_units_of(unit.radians)
    param_dict['sc_bb_bb_sc_torsion_phase_angle'] = (-160 * unit.degrees).in_units_of(unit.radians)
    
    # OpenMM torsion index 0 is type sc-bb-bb-sc
    # OpenMM torsion index 1 is type sc-bb-bb-bb
    # OpenMM torsion index 4 is type bb-bb-bb-bb
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, per, sc_bb_bb_sc_angle_updated, k) = force.getTorsionParameters(0)
            (par1, par2, par3, par4, per, bb_bb_bb_bb_angle_updated, k) = force.getTorsionParameters(4)
            (par1, par2, par3, par4, per, bb_bb_bb_sc_angle_updated, k) = force.getTorsionParameters(1)
            
    assert sc_bb_bb_sc_angle_updated == param_dict['sc_bb_bb_sc_torsion_phase_angle']
    assert bb_bb_bb_bb_angle_updated == param_dict['bb_bb_bb_bb_torsion_phase_angle']
    assert bb_bb_bb_sc_angle_updated == param_dict['bb_bb_bb_sc_torsion_phase_angle']
    
    # Now, try the reverse names:
    # Set up dictionary of parameter change instructions:
    param_dict_rev = {}
    param_dict_rev['sc_bb_bb_bb_torsion_phase_angle'] = ( 16 * unit.degrees).in_units_of(unit.radians)
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict_rev,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, per, sc_bb_bb_bb_angle_updated, k) = force.getTorsionParameters(1)
            
    assert sc_bb_bb_bb_angle_updated == param_dict_rev['sc_bb_bb_bb_torsion_phase_angle']
    
    
def test_eval_energy_new_torsion_k(tmpdir):  
    """
    Test simulation parameter update for varying torsion stiffness.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_bb_bb_bb_torsion_force_constant'] = 6.0 * unit.kilojoule_per_mole
    param_dict['bb_bb_bb_sc_torsion_force_constant'] = 4.5 * unit.kilojoule_per_mole
    param_dict['sc_bb_bb_sc_torsion_force_constant'] = 3.5 * unit.kilojoule_per_mole
    
    # OpenMM torsion index 0 is type sc-bb-bb-sc
    # OpenMM torsion index 1 is type sc-bb-bb-bb
    # OpenMM torsion index 4 is type bb-bb-bb-bb
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, per, angle, sc_bb_bb_sc_k_updated) = force.getTorsionParameters(0)
            (par1, par2, par3, par4, per, angle, bb_bb_bb_bb_k_updated) = force.getTorsionParameters(4)
            (par1, par2, par3, par4, per, angle, bb_bb_bb_sc_k_updated) = force.getTorsionParameters(1)
            
    assert sc_bb_bb_sc_k_updated == param_dict['sc_bb_bb_sc_torsion_force_constant']
    assert bb_bb_bb_bb_k_updated == param_dict['bb_bb_bb_bb_torsion_force_constant']
    assert bb_bb_bb_sc_k_updated == param_dict['bb_bb_bb_sc_torsion_force_constant']
    
    # Now, try the reverse names:
    # Set up dictionary of parameter change instructions:
    param_dict_rev = {}
    param_dict_rev['sc_bb_bb_bb_torsion_force_constant'] = 4.6 * unit.kilojoule_per_mole
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict_rev,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, per, angle, sc_bb_bb_bb_k_updated) = force.getTorsionParameters(1)
            
    assert sc_bb_bb_bb_k_updated == param_dict_rev['sc_bb_bb_bb_torsion_force_constant']
    
    
def test_eval_energy_new_torsion_periodicity(tmpdir):  
    """
    Test simulation parameter update for varying torsion periodicity.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_bb_bb_bb_torsion_periodicity'] = 2
    param_dict['bb_bb_bb_sc_torsion_periodicity'] = 3
    param_dict['sc_bb_bb_sc_torsion_periodicity'] = 4
    
    # OpenMM torsion index 0 is type sc-bb-bb-sc
    # OpenMM torsion index 1 is type sc-bb-bb-bb
    # OpenMM torsion index 4 is type bb-bb-bb-bb
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, sc_bb_bb_sc_per_updated, angle, k) = force.getTorsionParameters(0)
            (par1, par2, par3, par4, bb_bb_bb_bb_per_updated, angle, k) = force.getTorsionParameters(4)
            (par1, par2, par3, par4, bb_bb_bb_sc_per_updated, angle, k) = force.getTorsionParameters(1)
            
    assert sc_bb_bb_sc_per_updated == param_dict['sc_bb_bb_sc_torsion_periodicity']
    assert bb_bb_bb_bb_per_updated == param_dict['bb_bb_bb_bb_torsion_periodicity']
    assert bb_bb_bb_sc_per_updated == param_dict['bb_bb_bb_sc_torsion_periodicity']
    
    # Now, try the reverse names:
    # Set up dictionary of parameter change instructions:
    param_dict_rev = {}
    param_dict_rev['sc_bb_bb_bb_torsion_periodicity'] = 5
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict_rev,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )
    
    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, sc_bb_bb_bb_per_updated, angle, k) = force.getTorsionParameters(1)
            
    assert sc_bb_bb_bb_per_updated == param_dict_rev['sc_bb_bb_bb_torsion_periodicity']    

    
def test_eval_energy_sums_periodic_torsion_1(tmpdir):  
    """
    Test simulation parameter update for varying multiple torsion periodicity terms.
    Multiple periodicity terms specified as list of quantities.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel with 2 periodic torsion terms
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel_per1_3.pkl", "rb" ))
    
    # Get original torsion list:
    torsion_list = cgmodel.get_torsion_list()
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_bb_bb_bb_torsion_periodicity'] = [1,3]
    param_dict['bb_bb_bb_bb_torsion_force_constant'] = [6.0*unit.kilojoule_per_mole, 2.0*unit.kilojoule_per_mole]
    param_dict['bb_bb_bb_bb_torsion_phase_angle'] = [15*unit.degrees, 5*unit.degrees]
    
    # Also modify some single periodicity sidechain torsions:
    # This cgmodel had one term originally for sidechain types,
    # so can't add more periodicity terms
    param_dict['bb_bb_bb_sc_torsion_periodicity'] = 3
    param_dict['bb_bb_bb_sc_torsion_force_constant'] = 1.5*unit.kilojoule_per_mole
    param_dict['bb_bb_bb_sc_torsion_phase_angle'] = 5*unit.degrees
    
    param_dict['sc_bb_bb_sc_torsion_periodicity'] = 3
    param_dict['sc_bb_bb_sc_torsion_force_constant'] = 2.5*unit.kilojoule_per_mole
    param_dict['sc_bb_bb_sc_torsion_phase_angle'] = 10*unit.degrees    
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )


def test_eval_energy_sums_periodic_torsion_2(tmpdir):  
    """
    Test simulation parameter update for varying multiple torsion periodicity terms.
    Multiple periodicity terms specified as quantities with list values.
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel with 2 periodic torsion terms
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel_per1_3.pkl", "rb" ))
    
    # Get original torsion list:
    torsion_list = cgmodel.get_torsion_list()
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_bb_bb_bb_torsion_periodicity'] = [1,3]
    param_dict['bb_bb_bb_bb_torsion_force_constant'] = [6.0,2.0] * unit.kilojoule_per_mole
    param_dict['bb_bb_bb_bb_torsion_phase_angle'] = [15,5] * unit.degrees
    
    # Also modify some single periodicity sidechain torsions:
    # This cgmodel had one term originally for sidechain types,
    # so can't add more periodicity terms
    param_dict['bb_bb_bb_sc_torsion_periodicity'] = 3
    param_dict['bb_bb_bb_sc_torsion_force_constant'] = 1.5*unit.kilojoule_per_mole
    param_dict['bb_bb_bb_sc_torsion_phase_angle'] = 5*unit.degrees
    
    param_dict['sc_bb_bb_sc_torsion_periodicity'] = 3
    param_dict['sc_bb_bb_sc_torsion_force_constant'] = 2.5*unit.kilojoule_per_mole
    param_dict['sc_bb_bb_sc_torsion_phase_angle'] = 10*unit.degrees    
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )
    
    
def test_eval_energy_sums_periodic_torsion_3(tmpdir):  
    """
    Test simulation parameter update for varying multiple torsion periodicity terms.
    Multiple periodicity terms specified with mixed input types
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel with 2 periodic torsion terms
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel_per1_3.pkl", "rb" ))
    
    # Get original torsion list:
    torsion_list = cgmodel.get_torsion_list()
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}
    param_dict['bb_bb_bb_bb_torsion_periodicity'] = [1,3]
    param_dict['bb_bb_bb_bb_torsion_force_constant'] = [6.0*unit.kilojoule_per_mole, 2.0*unit.kilojoule_per_mole]
    param_dict['bb_bb_bb_bb_torsion_phase_angle'] = [15,5] * unit.degrees
    
    # Also modify some single periodicity sidechain torsions:
    # This cgmodel had one term originally for sidechain types,
    # so can't add more periodicity terms
    param_dict['bb_bb_bb_sc_torsion_periodicity'] = 3
    param_dict['bb_bb_bb_sc_torsion_force_constant'] = 1.5*unit.kilojoule_per_mole
    param_dict['bb_bb_bb_sc_torsion_phase_angle'] = 5*unit.degrees
    
    param_dict['sc_bb_bb_sc_torsion_periodicity'] = 3
    param_dict['sc_bb_bb_sc_torsion_force_constant'] = 2.5*unit.kilojoule_per_mole
    param_dict['sc_bb_bb_sc_torsion_phase_angle'] = 10*unit.degrees    
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )
    
    
def test_eval_energy_all_parameters(tmpdir):  
    """
    Test simulation parameter update (all possible force field parameters at once)
    """
    
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}

    param_dict['bb_sigma'] = 2.50 * unit.angstrom
    param_dict['sc_sigma'] = 4.50 * unit.angstrom    
    
    param_dict['bb_epsilon'] = 2.25 * unit.kilojoule_per_mole
    param_dict['sc_epsilon'] = 5.25 * unit.kilojoule_per_mole   
    
    param_dict['bb_bb_bond_length'] = 2.25 * unit.angstrom
    param_dict['bb_sc_bond_length'] = 2.50 * unit.angstrom    
    
    param_dict['bb_bb_bond_force_constant'] = 12000 * unit.kilojoule_per_mole / unit.nanometer**2
    param_dict['bb_sc_bond_force_constant'] = 14000 * unit.kilojoule_per_mole / unit.nanometer**2    
    
    param_dict['bb_bb_bb_equil_bond_angle'] = (130 * unit.degrees).in_units_of(unit.radians)
    param_dict['bb_bb_sc_equil_bond_angle'] = ( 95 * unit.degrees).in_units_of(unit.radians)    
    
    param_dict['bb_bb_bb_bond_angle_force_constant'] = 200 * unit.kilojoule_per_mole / unit.radian**2
    param_dict['bb_bb_sc_bond_angle_force_constant'] = 175 * unit.kilojoule_per_mole / unit.radian**2    
    
    param_dict['bb_bb_bb_bb_torsion_phase_angle'] = (-165 * unit.degrees).in_units_of(unit.radians)
    param_dict['bb_bb_bb_sc_torsion_phase_angle'] = (  15 * unit.degrees).in_units_of(unit.radians)
    param_dict['sc_bb_bb_sc_torsion_phase_angle'] = (-160 * unit.degrees).in_units_of(unit.radians)    
    
    param_dict['bb_bb_bb_bb_torsion_force_constant'] = 6.0 * unit.kilojoule_per_mole
    param_dict['bb_bb_bb_sc_torsion_force_constant'] = 4.5 * unit.kilojoule_per_mole
    param_dict['sc_bb_bb_sc_torsion_force_constant'] = 3.5 * unit.kilojoule_per_mole    
    
    param_dict['bb_bb_bb_bb_torsion_periodicity'] = 2
    param_dict['bb_bb_bb_sc_torsion_periodicity'] = 3
    param_dict['sc_bb_bb_sc_torsion_periodicity'] = 4
    
    # Bond index 0 is type bb-sc
    # Bond index 1 is type bb-bb    
    
    # Angle index 0 is type sc-bb-bb
    # Angle index 2 is type bb-bb-bb    
    
    # OpenMM torsion index 0 is type sc-bb-bb-sc
    # OpenMM torsion index 1 is type sc-bb-bb-bb
    # OpenMM torsion index 4 is type bb-bb-bb-bb
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=100,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        
        if force_name == 'NonbondedForce':
            (q, sigma_bb_updated, epsilon_bb_updated) = force.getParticleParameters(0)
            (q, sigma_sc_updated, epsilon_sc_updated) = force.getParticleParameters(1)
            
        elif force_name == 'HarmonicBondForce': 
            (par1, par2,
            bb_sc_length_updated,
            bb_sc_k_updated) = force.getBondParameters(0)
            
            (par1, par2,
            bb_bb_length_updated,
            bb_bb_k_updated) = force.getBondParameters(1)     
        
        elif force_name == 'HarmonicAngleForce':
            (par1, par2, par3,
            bb_bb_sc_angle_updated,
            bb_bb_sc_k_updated) = force.getAngleParameters(0)
            
            (par1, par2, par3,
            bb_bb_bb_angle_updated,
            bb_bb_bb_k_updated) = force.getAngleParameters(2)
            
        elif force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4,
            sc_bb_bb_sc_per_updated,
            sc_bb_bb_sc_angle_updated,
            sc_bb_bb_sc_k_updated) = force.getTorsionParameters(0)
            
            (par1, par2, par3, par4,
            bb_bb_bb_bb_per_updated,
            bb_bb_bb_bb_angle_updated,
            bb_bb_bb_bb_k_updated) = force.getTorsionParameters(4)
            
            (par1, par2, par3, par4,
            bb_bb_bb_sc_per_updated,
            bb_bb_bb_sc_angle_updated,
            bb_bb_bb_sc_k_updated) = force.getTorsionParameters(1)
    
    # Check updated nonbonded parameters:
    
    # Check updated bond parameters:
    assert bb_bb_length_updated == param_dict['bb_bb_bond_length']
    assert bb_sc_length_updated == param_dict['bb_sc_bond_length']    
    
    assert bb_bb_k_updated == param_dict['bb_bb_bond_force_constant']
    assert bb_sc_k_updated == param_dict['bb_sc_bond_force_constant']    
    
    # Check updated bond angle parameters:
    assert bb_bb_bb_angle_updated == param_dict['bb_bb_bb_equil_bond_angle']
    assert bb_bb_sc_angle_updated == param_dict['bb_bb_sc_equil_bond_angle']
    
    assert bb_bb_bb_k_updated == param_dict['bb_bb_bb_bond_angle_force_constant']
    assert bb_bb_sc_k_updated == param_dict['bb_bb_sc_bond_angle_force_constant']     
     
    # Check updated torsion parameters: 
    assert sc_bb_bb_sc_angle_updated == param_dict['sc_bb_bb_sc_torsion_phase_angle']
    assert bb_bb_bb_bb_angle_updated == param_dict['bb_bb_bb_bb_torsion_phase_angle']
    assert bb_bb_bb_sc_angle_updated == param_dict['bb_bb_bb_sc_torsion_phase_angle']
     
    assert sc_bb_bb_sc_k_updated == param_dict['sc_bb_bb_sc_torsion_force_constant']
    assert bb_bb_bb_bb_k_updated == param_dict['bb_bb_bb_bb_torsion_force_constant']
    assert bb_bb_bb_sc_k_updated == param_dict['bb_bb_bb_sc_torsion_force_constant']
     
    assert sc_bb_bb_sc_per_updated == param_dict['sc_bb_bb_sc_torsion_periodicity']
    assert bb_bb_bb_bb_per_updated == param_dict['bb_bb_bb_bb_torsion_periodicity']
    assert bb_bb_bb_sc_per_updated == param_dict['bb_bb_bb_sc_torsion_periodicity']
      
 
def test_reeval_heat_capacity(tmpdir):
    """
    Test heat capacity calculation for non-simulated force field parameters (bootstrapping version)
    """
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}

    param_dict['bb_sigma'] = 2.50 * unit.angstrom
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=10,
        frame_end=-1,
        frame_stride=5,
        verbose=False,
    )
    
    # Evaluate heat capacities at simulated and non-simulated force field parameters:
    (Cv_sim, dCv_sim, Cv_reeval, dCv_reeval,
    T_list, FWHM, Tm, Cv_height, N_eff) = get_heat_capacity_reeval(
        U_kln=U_eval,
        output_data=output_data,
        frame_begin=10,
        frame_end=-1,
        sample_spacing=5,
        num_intermediate_states=1,
        frac_dT=0.05,
        plot_file_sim=f"{output_directory}/heat_capacity_sim.pdf",
        plot_file_reeval=f"{output_directory}/heat_capacity_reeval.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/heat_capacity_reeval.pdf")
    assert os.path.isfile(f"{output_directory}/heat_capacity_sim.pdf")
    
    
def test_reeval_heat_capacity_end_frame(tmpdir):
    """
    Test heat capacity calculation for non-simulated force field parameters (bootstrapping version)
    """
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}

    param_dict['bb_sigma'] = 2.50 * unit.angstrom
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=10,
        frame_end=150,
        frame_stride=5,
        verbose=False,
    )
    
    # Evaluate heat capacities at simulated and non-simulated force field parameters:
    (Cv_sim, dCv_sim, Cv_reeval, dCv_reeval,
    T_list, FWHM, Tm, Cv_height, N_eff) = get_heat_capacity_reeval(
        U_kln=U_eval,
        output_data=output_data,
        frame_begin=10,
        frame_end=150,
        sample_spacing=5,
        num_intermediate_states=1,
        frac_dT=0.05,
        plot_file_sim=f"{output_directory}/heat_capacity_sim.pdf",
        plot_file_reeval=f"{output_directory}/heat_capacity_reeval.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/heat_capacity_reeval.pdf")
    assert os.path.isfile(f"{output_directory}/heat_capacity_sim.pdf")
    
    
def test_reeval_heat_capacity_boot(tmpdir):
    """
    Test heat capacity calculation for non-simulated force field parameters (bootstrapping version)
    """
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}

    param_dict['bb_sigma'] = 2.50 * unit.angstrom
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=10,
        frame_end=-1,
        frame_stride=1,
        verbose=False,
    )
    
    # Evaluate heat capacities at simulated and non-simulated force field parameters:
    (new_temperature_list,
    C_v_values, C_v_uncertainty,
    Tm_value, Tm_uncertainty, 
    Cv_height_value, Cv_height_uncertainty,
    FWHM_value, FWHM_uncertainty,
    N_eff_values) = bootstrap_heat_capacity(
        U_kln=U_eval,
        output_data=output_data,
        frame_begin=10,
        frame_end=-1,
        sample_spacing=5,
        num_intermediate_states=1,
        n_trial_boot=10,
        plot_file=f"{output_directory}/heat_capacity_reeval_boot.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/heat_capacity_reeval_boot.pdf")
    
 
def test_reeval_heat_capacity_boot_sparse(tmpdir):
    """
    Test heat capacity calculation for non-simulated force field parameters (bootstrapping version),
    with a sparsifying stride applied to the energy evaluation step.
    """
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}

    param_dict['bb_sigma'] = 2.50 * unit.angstrom
    
    # Skip every other frame to speed up energy eval
    sparsify_stride = 2 
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=10,
        frame_end=-1,
        frame_stride=sparsify_stride,
        verbose=False,
    )
    
    # Evaluate heat capacities at simulated and non-simulated force field parameters:
    (new_temperature_list,
    C_v_values, C_v_uncertainty,
    Tm_value, Tm_uncertainty, 
    Cv_height_value, Cv_height_uncertainty,
    FWHM_value, FWHM_uncertainty,
    N_eff_values) = bootstrap_heat_capacity(
        U_kln=U_eval,
        output_data=output_data,
        frame_begin=10,
        frame_end=-1,
        sparsify_stride=sparsify_stride,
        sample_spacing=5,
        num_intermediate_states=1,
        n_trial_boot=10,
        plot_file=f"{output_directory}/heat_capacity_reeval_boot.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/heat_capacity_reeval_boot.pdf") 
   
   
def test_reeval_heat_capacity_boot_end_frame(tmpdir):
    """
    Test heat capacity calculation for non-simulated force field parameters,
    with an end frame specified (bootstrapping version)
    """
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up dictionary of parameter change instructions:
    param_dict = {}

    param_dict['bb_sigma'] = 2.50 * unit.angstrom
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=0,
        frame_end=150,
        frame_stride=1,
        verbose=False,
    )
    
    # Evaluate heat capacities at simulated and non-simulated force field parameters:
    (new_temperature_list,
    C_v_values, C_v_uncertainty,
    Tm_value, Tm_uncertainty, 
    Cv_height_value, Cv_height_uncertainty,
    FWHM_value, FWHM_uncertainty,
    N_eff_values) = bootstrap_heat_capacity(
        U_kln=U_eval,
        output_data=output_data,
        frame_begin=0,
        frame_end=150,
        sample_spacing=5,
        num_intermediate_states=1,
        n_trial_boot=10,
        plot_file=f"{output_directory}/heat_capacity_reeval_boot.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/heat_capacity_reeval_boot.pdf")
    

def test_eval_FWHM_sequences_no_change_1(tmpdir):
    """
    Test sequence energy/heat capacity evaluation code, with no changes made to the monomer types (A homopolymer),
    checking that heat capacity curve matches the original reference simulation.
    Sequence is single list of monomer dicts.
    Single heat capacity calculation (no bootstrapping)
    """
    output_directory = tmpdir.mkdir("output")    
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up monomer dictionaries:
    A = cgmodel.monomer_types[0]
    # sigma_bb = 2.25 A
    # epsilon_bb = 1.5 kJ/mol
    # sigma_sc = 3.5 A
    # epsilon_sc = 5.0 kJ/mol

    monomer_list = [A]
    
    nmono = len(cgmodel.sequence)
    
    sequence = []
    for i in range(int(nmono)):
        sequence.append(A)

    frame_begin = 100
    frame_end = 150
    sample_spacing = 1
    sparsify_stride = 1
    num_intermediate_states = 1

    # Re-evaluate OpenMM energies:
    (T_list, seq_Cv, seq_Cv_uncertainty,
    seq_Tm, seq_Tm_uncertainty,
    seq_Cv_height, seq_Cv_height_uncertainty,
    seq_FWHM, seq_FWHM_uncertainty,
    seq_N_eff) = eval_energy_sequences(
        cgmodel,
        dcd_file_list,
        temperature_list,
        monomer_list,
        sequence=sequence,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=None,
        plot_dir=output_directory,
        output_data=output_data,
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=sample_spacing,
        sparsify_stride=sparsify_stride,
        verbose=True,
        n_cpu=1,
    )
    for key, value in seq_Cv.items():
        seq_Cv_array = value
    
    # Get heat capacity from original dataset:
    (Cv_ref, dCv_ref, temperature_list_ref,
    FWHM_ref, Tm_ref, Cv_height_ref, N_eff_ref) = get_heat_capacity(
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=int(sample_spacing*sparsify_stride),
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
        plot_file=f"{output_directory}/heat_capacity_ref.pdf"
    )
    
    seq_Cv_array = seq_Cv_array.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
    Cv_ref = Cv_ref.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
    
    assert_allclose(seq_Cv_array, Cv_ref,atol=1E-4)
    
    
def test_eval_FWHM_sequences_no_change_2(tmpdir):
    """
    Test sequence energy/heat capacity evaluation code, with no changes made to the monomer types (AB alt),
    checking that heat capacity curve matches the original reference simulation.
    Sequence is single list of integers corresponding to indices in monomer_list.
    Single heat capacity calculation (no bootstrapping)
    Sparsify stride is applied to evaluate energies of fewer frames.
    """
    output_directory = tmpdir.mkdir("output")    
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up monomer dictionaries:
    A = cgmodel.monomer_types[0]
    # sigma_bb = 2.25 A
    # epsilon_bb = 1.5 kJ/mol
    # sigma_sc = 3.5 A
    # epsilon_sc = 5.0 kJ/mol
    
    B = copy.deepcopy(A)
    # A and B need separate names:
    B["monomer_name"] = "B"

    monomer_list = [A,B]
    
    nmono = len(cgmodel.sequence)
    
    sequence = []
    for i in range(int(nmono/2)):
        sequence.append(0)
        sequence.append(1)

    frame_begin = 100
    frame_end = 150
    sample_spacing = 6
    sparsify_stride = 3
    num_intermediate_states = 1

    # Re-evaluate OpenMM energies:
    (T_list, seq_Cv, seq_Cv_uncertainty,
    seq_Tm, seq_Tm_uncertainty,
    seq_Cv_height, seq_Cv_height_uncertainty,
    seq_FWHM, seq_FWHM_uncertainty,
    seq_N_eff) = eval_energy_sequences(
        cgmodel,
        dcd_file_list,
        temperature_list,
        monomer_list,
        sequence=sequence,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=None,
        plot_dir=output_directory,
        output_data=output_data,
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=sample_spacing,
        sparsify_stride=sparsify_stride,
        verbose=True,
        n_cpu=1,
    )
    for key, value in seq_Cv.items():
        seq_Cv_array = value
    
    # Get heat capacity from original dataset:
    (Cv_ref, dCv_ref, temperature_list_ref,
    FWHM_ref, Tm_ref, Cv_height_ref, N_eff_ref) = get_heat_capacity(
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=int(sample_spacing*sparsify_stride),
        output_data=output_data,
        num_intermediate_states=num_intermediate_states,
        plot_file=f"{output_directory}/heat_capacity_ref.pdf"
    )
    
    seq_Cv_array = seq_Cv_array.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
    Cv_ref = Cv_ref.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
    
    assert_allclose(seq_Cv_array, Cv_ref,atol=1E-4)
        
    
def test_eval_FWHM_boot_sequences_no_change_1(tmpdir):
    """
    Test sequence energy/heat capacity evaluation code, with no changes made to the monomer types (A homopolymer).
    Sequence is single list of monomer dicts.
    Bootstrapping heat capacity calculation.
    """
    output_directory = tmpdir.mkdir("output")    
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up monomer dictionaries:
    A = cgmodel.monomer_types[0]
    # sigma_bb = 2.25 A
    # epsilon_bb = 1.5 kJ/mol
    # sigma_sc = 3.5 A
    # epsilon_sc = 5.0 kJ/mol

    monomer_list = [A]
    
    nmono = len(cgmodel.sequence)
    
    sequence = []
    for i in range(int(nmono)):
        sequence.append(A)

    frame_begin = 100
    frame_end = 150
    sample_spacing = 1
    sparsify_stride = 1
    num_intermediate_states = 1

    # Re-evaluate OpenMM energies:
    (T_list, seq_Cv, seq_Cv_uncertainty,
    seq_Tm, seq_Tm_uncertainty,
    seq_Cv_height, seq_Cv_height_uncertainty,
    seq_FWHM, seq_FWHM_uncertainty,
    seq_N_eff) = eval_energy_sequences(
        cgmodel,
        dcd_file_list,
        temperature_list,
        monomer_list,
        sequence=sequence,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=10,
        plot_dir=output_directory,
        output_data=output_data,
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=sample_spacing,
        sparsify_stride=sparsify_stride,
        verbose=True,
        n_cpu=1,
    )

    
def test_eval_FWHM_boot_sequences_no_change_2(tmpdir):
    """
    Test sequence energy/heat capacity evaluation code, with no changes made to the monomer types (AB alt).
    Sequence is single list of integers corresponding to indices in monomer_list.
    Bootstrapping heat capacity calculation.
    Sparsify stride is applied to evaluate energies of fewer frames.
    """
    output_directory = tmpdir.mkdir("output")    
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up monomer dictionaries:
    A = cgmodel.monomer_types[0]
    # sigma_bb = 2.25 A
    # epsilon_bb = 1.5 kJ/mol
    # sigma_sc = 3.5 A
    # epsilon_sc = 5.0 kJ/mol
    
    B = copy.deepcopy(A)
    # A and B need separate names:
    B["monomer_name"] = "B"

    monomer_list = [A,B]
    
    nmono = len(cgmodel.sequence)
    
    sequence = []
    for i in range(int(nmono/2)):
        sequence.append(0)
        sequence.append(1)

    frame_begin = 100
    frame_end = 150
    sample_spacing = 6
    sparsify_stride = 3
    num_intermediate_states = 1

    # Re-evaluate OpenMM energies:
    (T_list, seq_Cv, seq_Cv_uncertainty,
    seq_Tm, seq_Tm_uncertainty,
    seq_Cv_height, seq_Cv_height_uncertainty,
    seq_FWHM, seq_FWHM_uncertainty,
    seq_N_eff) = eval_energy_sequences(
        cgmodel,
        dcd_file_list,
        temperature_list,
        monomer_list,
        sequence=sequence,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=10,
        plot_dir=output_directory,
        output_data=output_data,
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=sample_spacing,
        sparsify_stride=sparsify_stride,
        verbose=True,
        n_cpu=1,
    )

        
def test_eval_FWHM_sequences_AB(tmpdir):
    """
    Test sequence energy/heat capacity evaluation code with a new monomer type B defined.
    Sequence is single list of monomer dicts.
    Single heat capacity calculation (no bootstrapping)
    """
    output_directory = tmpdir.mkdir("output")    
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up monomer dictionaries:
    A = cgmodel.monomer_types[0]
    # sigma_bb = 2.25 A
    # epsilon_bb = 1.5 kJ/mol
    # sigma_sc = 3.5 A
    # epsilon_sc = 5.0 kJ/mol

    mass = 100 * unit.amu
        
    bb2 = {
        "particle_type_name": "bb2",
        "sigma": 2.20 * unit.angstrom,
        "epsilon": 1.25 * unit.kilojoules_per_mole,
        "mass": mass
    }
    sc2 = {
        "particle_type_name": "sc2",
        "sigma": 3.55 * unit.angstrom,
        "epsilon": 5.25 * unit.kilojoules_per_mole,
        "mass": mass
    }

    B = {
        "monomer_name": "B",
        "particle_sequence": [bb2, sc2],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }

    monomer_list = [A,B]
    
    nmono = len(cgmodel.sequence)
    
    sequence = []
    for i in range(int(nmono/2)):
        sequence.append(A)
        sequence.append(B)
        
    frame_begin = 100
    frame_end = 150
    sample_spacing = 1
    sparsify_stride = 1
    num_intermediate_states = 1

    # Re-evaluate OpenMM energies:
    (T_list, seq_Cv, seq_Cv_uncertainty,
    seq_Tm, seq_Tm_uncertainty,
    seq_Cv_height, seq_Cv_height_uncertainty,
    seq_FWHM, seq_FWHM_uncertainty,
    seq_N_eff) = eval_energy_sequences(
        cgmodel,
        dcd_file_list,
        temperature_list,
        monomer_list,
        sequence=sequence,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=None,
        plot_dir=output_directory,
        output_data=output_data,
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=sample_spacing,
        sparsify_stride=sparsify_stride,
        verbose=True,
        n_cpu=1,
    )

  
def test_eval_FWHM_sequences_ABC(tmpdir):
    """
    Test sequence energy/heat capacity evaluation code with new monomer types B,C defined.
    Sequence is single list of monomer dicts.
    Single heat capacity calculation (no bootstrapping).
    No plotting.
    """
    output_directory = tmpdir.mkdir("output")    
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up monomer dictionaries:
    A = cgmodel.monomer_types[0]
    # sigma_bb = 2.25 A
    # epsilon_bb = 1.5 kJ/mol
    # sigma_sc = 3.5 A
    # epsilon_sc = 5.0 kJ/mol

    mass = 100 * unit.amu
        
    bb2 = {
        "particle_type_name": "bb2",
        "sigma": 2.20 * unit.angstrom,
        "epsilon": 1.25 * unit.kilojoules_per_mole,
        "mass": mass
    }
    sc2 = {
        "particle_type_name": "sc2",
        "sigma": 3.55 * unit.angstrom,
        "epsilon": 5.25 * unit.kilojoules_per_mole,
        "mass": mass
    }
    sc3 = {
        "particle_type_name": "sc2",
        "sigma": 3.55 * unit.angstrom,
        "epsilon": 4.25 * unit.kilojoules_per_mole,
        "mass": mass
    }    

    B = {
        "monomer_name": "B",
        "particle_sequence": [bb2, sc2],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }
    
    C = {
        "monomer_name": "B",
        "particle_sequence": [bb2, sc3],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }    

    monomer_list = [A,B,C]
    
    nmono = len(cgmodel.sequence)
    
    sequence = []
    for i in range(int(nmono/3)):
        sequence.append(A)
        sequence.append(B)
        sequence.append(C)
        
    frame_begin = 100
    frame_end = 150
    sample_spacing = 1
    sparsify_stride = 1
    num_intermediate_states = 1

    # Re-evaluate OpenMM energies:
    (T_list, seq_Cv, seq_Cv_uncertainty,
    seq_Tm, seq_Tm_uncertainty,
    seq_Cv_height, seq_Cv_height_uncertainty,
    seq_FWHM, seq_FWHM_uncertainty,
    seq_N_eff) = eval_energy_sequences(
        cgmodel,
        dcd_file_list,
        temperature_list,
        monomer_list,
        sequence=sequence,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=None,
        plot_dir=None,
        output_data=output_data,
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=sample_spacing,
        sparsify_stride=sparsify_stride,
        verbose=True,
        n_cpu=1,
    )
    

def test_eval_FWHM_sequences_multi_1(tmpdir):
    """
    Test sequence energy/heat capacity evaluation code with a new monomer type B defined,
    checking that heat capacity curve matches the original reference simulation.
    Sequence is list of list of monomer dicts.
    Single heat capacity calculation (no bootstrapping)
    """
    output_directory = tmpdir.mkdir("output")    
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up monomer dictionaries:
    A = cgmodel.monomer_types[0]
    # sigma_bb = 2.25 A
    # epsilon_bb = 1.5 kJ/mol
    # sigma_sc = 3.5 A
    # epsilon_sc = 5.0 kJ/mol

    mass = 100 * unit.amu
        
    bb2 = {
        "particle_type_name": "bb2",
        "sigma": 2.20 * unit.angstrom,
        "epsilon": 1.25 * unit.kilojoules_per_mole,
        "mass": mass
    }
    sc2 = {
        "particle_type_name": "sc2",
        "sigma": 3.55 * unit.angstrom,
        "epsilon": 5.25 * unit.kilojoules_per_mole,
        "mass": mass
    }

    B = {
        "monomer_name": "B",
        "particle_sequence": [bb2, sc2],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }

    monomer_list = [A,B]
    
    nmono = len(cgmodel.sequence)
    
    sequence = []
    
    seq_1 = []
    for i in range(int(nmono/2)):
        seq_1.append(A)
        seq_1.append(B)
        
    seq_2 = []
    for i in range(int(nmono/4)):
        seq_2.append(A)
        seq_2.append(A)
        seq_2.append(B)
        seq_2.append(B)
        
    sequence.append(seq_1)
    sequence.append(seq_2)
        
    frame_begin = 100
    frame_end = 150
    sample_spacing = 1
    sparsify_stride = 1
    num_intermediate_states = 1

    # Re-evaluate OpenMM energies:
    (T_list, seq_Cv, seq_Cv_uncertainty,
    seq_Tm, seq_Tm_uncertainty,
    seq_Cv_height, seq_Cv_height_uncertainty,
    seq_FWHM, seq_FWHM_uncertainty,
    seq_N_eff) = eval_energy_sequences(
        cgmodel,
        dcd_file_list,
        temperature_list,
        monomer_list,
        sequence=sequence,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=None,
        plot_dir=output_directory,
        output_data=output_data,
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=sample_spacing,
        sparsify_stride=sparsify_stride,
        verbose=True,
        n_cpu=1,
    )    
    
 
def test_eval_FWHM_sequences_multi_2(tmpdir):
    """
    Test sequence energy/heat capacity evaluation code with a new monomer type B defined,
    checking that heat capacity curve matches the original reference simulation.
    Sequence is list of list of integers corresponding to indices in monomer_list.
    Single heat capacity calculation (no bootstrapping)
    """
    output_directory = tmpdir.mkdir("output")
    
    # Replica exchange settings
    number_replicas = 12
    min_temp = 200.0 * unit.kelvin
    max_temp = 600.0 * unit.kelvin
    temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
    
    # Load in cgmodel
    cgmodel = pickle.load(open(f"{data_path}/stored_cgmodel.pkl", "rb" ))
    
    # Data file with simulated energies:
    output_data = os.path.join(data_path, "output.nc")
    
    # Create list of replica trajectories to analyze
    dcd_file_list = []
    for i in range(len(temperature_list)):
        dcd_file_list.append(f"{data_path}/replica_{i+1}.dcd")
    
    # Set up monomer dictionaries:
    A = cgmodel.monomer_types[0]
    # sigma_bb = 2.25 A
    # epsilon_bb = 1.5 kJ/mol
    # sigma_sc = 3.5 A
    # epsilon_sc = 5.0 kJ/mol

    mass = 100 * unit.amu
        
    bb2 = {
        "particle_type_name": "bb2",
        "sigma": 2.20 * unit.angstrom,
        "epsilon": 1.25 * unit.kilojoules_per_mole,
        "mass": mass
    }
    sc2 = {
        "particle_type_name": "sc2",
        "sigma": 3.55 * unit.angstrom,
        "epsilon": 5.25 * unit.kilojoules_per_mole,
        "mass": mass
    }

    B = {
        "monomer_name": "B",
        "particle_sequence": [bb2, sc2],
        "bond_list": [[0, 1]],
        "start": 0,
        "end": 0,
    }

    monomer_list = [A,B]
    
    nmono = len(cgmodel.sequence)
    
    sequence = []
    
    seq_1 = []
    for i in range(int(nmono/2)):
        seq_1.append(0)
        seq_1.append(1)
        
    seq_2 = []
    for i in range(int(nmono/4)):
        seq_2.append(0)
        seq_2.append(0)
        seq_2.append(1)
        seq_2.append(1)
        
    sequence.append(seq_1)
    sequence.append(seq_2)
        
    frame_begin = 100
    frame_end = 150
    sample_spacing = 1
    sparsify_stride = 1
    num_intermediate_states = 1

    # Re-evaluate OpenMM energies:
    (T_list, seq_Cv, seq_Cv_uncertainty,
    seq_Tm, seq_Tm_uncertainty,
    seq_Cv_height, seq_Cv_height_uncertainty,
    seq_FWHM, seq_FWHM_uncertainty,
    seq_N_eff) = eval_energy_sequences(
        cgmodel,
        dcd_file_list,
        temperature_list,
        monomer_list,
        sequence=sequence,
        num_intermediate_states=num_intermediate_states,
        n_trial_boot=None,
        plot_dir=output_directory,
        output_data=output_data,
        frame_begin=frame_begin,
        frame_end=frame_end,
        sample_spacing=sample_spacing,
        sparsify_stride=sparsify_stride,
        verbose=True,
        n_cpu=1,
    )     
  