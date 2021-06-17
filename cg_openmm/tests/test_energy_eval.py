"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed  
  
import os
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.evaluate_energy import eval_energy
from cg_openmm.thermo.calc import *
from cg_openmm.parameters.reweight import get_temperature_list, get_opt_temperature_list
from openmmtools.multistate import MultiStateReporter
from openmmtools.multistate import ReplicaExchangeAnalyzer
from numpy.testing import assert_almost_equal, assert_allclose
import pickle
    
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_data')
       
    
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
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=0,
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
    assert_allclose(U_eval,replica_energies,atol=1E-3)
    
    
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
        frame_begin=0,
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
        frame_begin=0,
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
        frame_begin=0,
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
        frame_begin=0,
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
        frame_begin=0,
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
        frame_begin=0,
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
        frame_begin=0,
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
        frame_begin=0,
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
        frame_begin=0,
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
        frame_begin=0,
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
    
    # OpenMM torsion index 0 is type bb-bb-bb-bb
    # OpenMM torsion index 1 is type sc-bb-bb-sc
    # OpenMM torsion index 5 is type bb-bb-bb-sc
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=0,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, per, sc_bb_bb_sc_angle_updated, k) = force.getTorsionParameters(1)
            (par1, par2, par3, par4, per, bb_bb_bb_bb_angle_updated, k) = force.getTorsionParameters(0)
            (par1, par2, par3, par4, per, bb_bb_bb_sc_angle_updated, k) = force.getTorsionParameters(5)
            
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
        frame_begin=0,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, per, sc_bb_bb_bb_angle_updated, k) = force.getTorsionParameters(5)
            
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
    
    # OpenMM torsion index 0 is type bb-bb-bb-bb
    # OpenMM torsion index 1 is type sc-bb-bb-sc
    # OpenMM torsion index 5 is type bb-bb-bb-sc
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=0,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, per, angle, sc_bb_bb_sc_k_updated) = force.getTorsionParameters(1)
            (par1, par2, par3, par4, per, angle, bb_bb_bb_bb_k_updated) = force.getTorsionParameters(0)
            (par1, par2, par3, par4, per, angle, bb_bb_bb_sc_k_updated) = force.getTorsionParameters(5)
            
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
        frame_begin=0,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, per, angle, sc_bb_bb_bb_k_updated) = force.getTorsionParameters(5)
            
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
    
    # OpenMM torsion index 0 is type bb-bb-bb-bb
    # OpenMM torsion index 1 is type sc-bb-bb-sc
    # OpenMM torsion index 5 is type bb-bb-bb-sc
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=0,
        frame_end=-1,
        frame_stride=5,
        verbose=True,
    )

    for force_index, force in enumerate(simulation.system.getForces()):
        force_name = force.__class__.__name__
        if force_name == 'PeriodicTorsionForce':
            (par1, par2, par3, par4, sc_bb_bb_sc_per_updated, angle, k) = force.getTorsionParameters(1)
            (par1, par2, par3, par4, bb_bb_bb_bb_per_updated, angle, k) = force.getTorsionParameters(0)
            (par1, par2, par3, par4, bb_bb_bb_sc_per_updated, angle, k) = force.getTorsionParameters(5)
            
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
        frame_begin=0,
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
    
    # OpenMM torsion index 0 is type bb-bb-bb-bb
    # OpenMM torsion index 1 is type sc-bb-bb-sc
    # OpenMM torsion index 5 is type bb-bb-bb-sc
    
    # Re-evaluate OpenMM energies:
    U_eval, simulation = eval_energy(
        cgmodel,
        dcd_file_list,
        temperature_list,
        param_dict,
        frame_begin=0,
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
            sc_bb_bb_sc_k_updated) = force.getTorsionParameters(1)
            
            (par1, par2, par3, par4,
            bb_bb_bb_bb_per_updated,
            bb_bb_bb_bb_angle_updated,
            bb_bb_bb_bb_k_updated) = force.getTorsionParameters(0)
            
            (par1, par2, par3, par4,
            bb_bb_bb_sc_per_updated,
            bb_bb_bb_sc_angle_updated,
            bb_bb_bb_sc_k_updated) = force.getTorsionParameters(5)
    
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
