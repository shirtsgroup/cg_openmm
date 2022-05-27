"""
Unit and regression test for the cg_openmm package.
"""

import os
import sys

import mdtraj as md
import numpy as np
import pytest
from cg_openmm.utilities.helix_optimize_geometry import *
from cg_openmm.utilities.helix_optimize_nonbonded import *
from numpy.testing import assert_almost_equal
# Import package, test suite, and other packages as needed
from openmm import unit


def test_optimize_helix_simple(tmpdir):
    """
    Test the LJ helix geometry optimization function
    (1-2 and 1-3 nonbonded interactions, no sidechain beads)
    """
    
    # Particle LJ 12-6 parameters:
    sigma = 1.0 * unit.angstrom
    epsilon = 1.0 * unit.kilojoule_per_mole

    # Number of backbone particles:
    n_particle_bb = 12

    # Option to add sidechain particles normal to helix (equal bond length to bb-bb)
    sidechain=False

    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/LJ_helix_test.pdb"
    plotfile = f"{output_directory}/LJ_helix_test.pdf"
    
    opt_solution, geometry = optimize_helix_simple(
        n_particle_bb,sigma,epsilon,sidechain,
        pdbfile=pdbfile, plotfile=plotfile,
        DE_popsize=10)
           
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)

    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 12
        
        
def test_optimize_helix_simple_sidechain(tmpdir):
    """
    Test the LJ helix geometry optimization function
    (1-2 and 1-3 nonbonded interactions, no sidechain beads)
    """
    
    # Particle LJ 12-6 parameters:
    sigma = 2.0 * unit.angstrom
    epsilon = 2.0 * unit.kilojoule_per_mole

    # Number of backbone particles:
    n_particle_bb = 12

    # Option to add sidechain particles normal to helix (equal bond length to bb-bb)
    sidechain=True

    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/LJ_helix_sidechain_test.pdb"
    plotfile = f"{output_directory}/LJ_helix_sidechain_test.pdf"
    
    opt_solution, geometry = optimize_helix_simple(
        n_particle_bb,sigma,epsilon,sidechain,
        pdbfile=pdbfile, plotfile=plotfile,
        DE_popsize=10)

    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24
    

def test_optimize_helix_openmm_unconstrained(tmpdir):
    """
    Test the LJ helix geometry optimization function
    (openmm energy with default [0,0,1] nonbonded exclusions, 1sc, unconstrained bond lengths)
    """
    
    # Particle LJ 12-6 parameters:
    sigma_bb = 1.0 * unit.angstrom
    sigma_sc = 1.0 * unit.angstrom

    epsilon_bb = 1.0 * unit.kilojoule_per_mole
    epsilon_sc = 1.0 * unit.kilojoule_per_mole

    # Number of backbone particles:
    n_particle_bb = 12

    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/LJ_helix_openmm_unconstrained_test.pdb"
    plotfile = f"{output_directory}/LJ_helix_openmm_unconstrained_test.pdf"
    
    opt_solution, geometry = optimize_helix_openmm_energy(
        n_particle_bb, sigma_bb, sigma_sc, epsilon_bb, epsilon_sc,
        pdbfile=pdbfile, plotfile=plotfile, bond_dist_bb=None, bond_dist_sc=None,
        DE_popsize=20)
           
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24
    
    
def test_optimize_helix_openmm_unconstrained_exclusions(tmpdir):
    """
    Test the LJ helix geometry optimization function
    (openmm energy with non-default nonbonded exclusions, 1sc, unconstrained bond lengths)
    """
    
    # Particle LJ 12-6 parameters:
    sigma_bb = 1.0 * unit.angstrom
    sigma_sc = 1.0 * unit.angstrom

    epsilon_bb = 1.0 * unit.kilojoule_per_mole
    epsilon_sc = 1.0 * unit.kilojoule_per_mole

    # Number of backbone particles:
    n_particle_bb = 12

    # Set custom exclusion rules:
    exclusions = {
       "bb_bb_exclusions": [0,0,1],
       "bb_sc_exclusions": [0,1,1],
       "sc_sc_exclusions": [0,1,1],
    }
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/LJ_helix_openmm_unconstrained_test.pdb"
    plotfile = f"{output_directory}/LJ_helix_openmm_unconstrained_test.pdf"
    
    opt_solution, geometry = optimize_helix_openmm_energy(
        n_particle_bb, sigma_bb, sigma_sc, epsilon_bb, epsilon_sc,
        pdbfile=pdbfile, plotfile=plotfile, bond_dist_bb=None, bond_dist_sc=None,
        DE_popsize=20,exclusions=exclusions)
           
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24    
    

def test_optimize_helix_openmm_constrained(tmpdir):
    """
    Test the LJ helix geometry optimization function
    (openmm energy with nonbonded exclusions, 1sc, fixed bond lengths)
    """
    
    # Particle LJ 12-6 parameters:
    sigma_bb = 1.0 * unit.angstrom
    sigma_sc = 1.0 * unit.angstrom

    epsilon_bb = 1.0 * unit.kilojoule_per_mole
    epsilon_sc = 1.0 * unit.kilojoule_per_mole

    # Number of backbone particles:
    n_particle_bb = 12

    # Bond constraints (equilibrium values)
    bond_dist_bb = 1.10 * unit.angstrom
    bond_dist_sc = 1.20 * unit.angstrom    
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/LJ_helix_openmm_constrained_test.pdb"
    plotfile = f"{output_directory}/LJ_helix_openmm_constrained_test.pdf"
    
    opt_solution, geometry = optimize_helix_openmm_energy(
        n_particle_bb, sigma_bb, sigma_sc, epsilon_bb, epsilon_sc,
        pdbfile=pdbfile, plotfile=plotfile,
        bond_dist_bb=bond_dist_bb, bond_dist_sc=bond_dist_sc,
        DE_popsize=20)
               
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that the bond constraints are satisfied:
    bond_dist_bb_out = geometry['bb_bb_distance']
    bond_dist_sc_out = geometry['bb_sc_distance']
    
    assert_almost_equal(
        bond_dist_bb.value_in_unit(unit.angstrom),
        bond_dist_bb_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        bond_dist_sc.value_in_unit(unit.angstrom),
        bond_dist_sc_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24    
    
    
def test_optimize_helix_LJ_parameters_equal_bonds(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (1sc, optimize sigmas with specified radius, pitch, equal bond lengths for bb-bb and bb-sc)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12 
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_equal_bonds_test.pdb"
    plotfile = f"{output_directory}/helix_opt_LJ_openmm_equal_bonds_test.pdf"

    opt_solution, geometry = optimize_helix_LJ_parameters(
        radius, pitch, n_particle_bb,
        bond_dist_bb=None, bond_dist_sc=None,
        equal_bonds=True, DE_popsize=20,
        pdbfile=pdbfile, plotfile=plotfile,
        )
               
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )    
    
    # Check that equal bond length criteria is satisfied:
    r_bb_out = geometry['bb_bb_distance']
    r_bs_out = geometry['bb_sc_distance']
    
    assert_almost_equal(
        r_bb_out.value_in_unit(unit.angstrom),
        r_bs_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24      
    
    
def test_optimize_helix_LJ_parameters_unconstrained_bonds(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (1sc, optimize sigmas with specified radius, pitch, unconstrained independent bond lengths
    for bb-bb and bb-sc)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12 
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_unconstrained_test.pdb"
    plotfile = f"{output_directory}/helix_opt_LJ_openmm_unconstrained_test.pdf"

    opt_solution, geometry = optimize_helix_LJ_parameters(
        radius, pitch, n_particle_bb,
        bond_dist_bb=None, bond_dist_sc=None,
        equal_bonds=False, DE_popsize=20,
        pdbfile=pdbfile, plotfile=plotfile,
        )
               
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24               
    

def test_optimize_helix_LJ_parameters_fixed_bonds(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (1sc, optimize sigmas with specified radius, pitch, fixed bond lengths)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12

    # Bond constraints (equilibrium values)
    bond_dist_bb = 1.10 * unit.angstrom
    bond_dist_sc = 1.20 * unit.angstrom    
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_fixed_bonds_test.pdb"
    plotfile = f"{output_directory}/helix_opt_LJ_openmm_fixed_bonds_test.pdf"

    opt_solution, geometry = optimize_helix_LJ_parameters(
        radius, pitch, n_particle_bb,
        bond_dist_bb=bond_dist_bb, bond_dist_sc=bond_dist_sc,
        pdbfile=pdbfile, plotfile=plotfile,
        equal_bonds=False, DE_popsize=20)

    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that the bond constraints are satisfied:
    bond_dist_bb_out = geometry['bb_bb_distance']
    bond_dist_sc_out = geometry['bb_sc_distance']
    
    assert_almost_equal(
        bond_dist_bb.value_in_unit(unit.angstrom),
        bond_dist_bb_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        bond_dist_sc.value_in_unit(unit.angstrom),
        bond_dist_sc_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24    
    
    
def test_optimize_helix_LJ_parameters_energy_diff_equal_bonds(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (1sc, optimize sigmas with specified radius, pitch, equal bonds,
    energy difference objective function)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12 
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_energy_diff_equal_bonds_test.pdb"
    plotfile = f"{output_directory}/helix_opt_LJ_openmm_energy_diff_equal_bonds_test.pdf"

    geometry, opt_results_outer, opt_results_in = optimize_helix_LJ_parameters_energy_diff(
        radius, pitch, n_particle_bb,
        equal_bonds=True, brute_step_rad=0.2,
        pdbfile=pdbfile, plotfile=plotfile,
        DE_popsize=20)
               
    extended_pdbfile = pdbfile[:-4]+'_ext.pdb'        
               
    assert os.path.isfile(plotfile)
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(extended_pdbfile)
    
    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that equal bond length criteria is satisfied:
    r_bb_out = geometry['bb_bb_distance']
    r_bs_out = geometry['bb_sc_distance']
    
    assert_almost_equal(
        r_bb_out.value_in_unit(unit.angstrom),
        r_bs_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the helix pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24     

    # Check that we can load the extended pdb file
    pdb_loaded = md.load(extended_pdbfile)
    assert pdb_loaded.n_atoms == 24
    
    
def test_optimize_helix_LJ_parameters_energy_diff_unconstrained_bonds(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (1sc, optimize sigmas with specified radius, pitch, independent
    bb-bb and bb-sc bond lengths, energy difference objective function)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_constrained_test.pdb"
    plotfile = f"{output_directory}/helix_opt_LJ_openmm_constrained_test.pdf"

    geometry, opt_results_outer, opt_results_in = optimize_helix_LJ_parameters_energy_diff(
        radius, pitch, n_particle_bb,
        equal_bonds=False, brute_step_rad=0.5,
        brute_step_r_bs=0.5,
        pdbfile=pdbfile, plotfile=plotfile,
        DE_popsize=20)
               
    extended_pdbfile = pdbfile[:-4]+'_ext.pdb'        
               
    assert os.path.isfile(plotfile)
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(extended_pdbfile)
    
    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the helix pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24     

    # Check that we can load the extended pdb file
    pdb_loaded = md.load(extended_pdbfile)
    assert pdb_loaded.n_atoms == 24     
    
    
def test_optimize_helix_LJ_parameters_2sc_equal(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (2sc model, equal sigma_sc: optimize sigma, bb-sc, sc-sc bond lengths
    with specified radius, pitch, fixed bb-bb bond lengths)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12

    # Backbone sigma:
    sigma_bb = 1.2 * unit.angstrom

    # Bond constraints (equilibrium values)
    bond_dist_bb = 1.10 * unit.angstrom  
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_2sc_equal_test.pdb"

    opt_solution, geometry = optimize_helix_LJ_parameters_2sc(
        radius, pitch, n_particle_bb, sigma_bb, bond_dist_bb,
        equal_sc=True, pdbfile=pdbfile, DE_popsize=20)

    assert os.path.isfile(pdbfile)
    
    # Check that the bond constraints are satisfied:
    bond_dist_bb_out = geometry['bb_bb_distance']
    
    assert_almost_equal(
        bond_dist_bb.value_in_unit(unit.angstrom),
        bond_dist_bb_out.value_in_unit(unit.angstrom),
        decimal=6
    )

    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 36  


def test_optimize_helix_LJ_parameters_2sc_equal_exclusions(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (2sc model, equal sigma_sc: optimize sigma, bb-sc, sc-sc bond lengths
    with specified radius, pitch, fixed bb-bb bond lengths and 
    custom nonbonded exclusion rules)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12

    # Backbone sigma:
    sigma_bb = 1.2 * unit.angstrom

    # Bond constraints (equilibrium values)
    bond_dist_bb = 1.10 * unit.angstrom  
    
    # Set custom exclusion rules:
    exclusions = {
       "default_exclusions": [1,1,1],
    }    
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_2sc_equal_test.pdb"

    opt_solution, geometry = optimize_helix_LJ_parameters_2sc(
        radius, pitch, n_particle_bb, sigma_bb, bond_dist_bb,
        equal_sc=True, pdbfile=pdbfile, DE_popsize=20,
        exclusions=exclusions)

    assert os.path.isfile(pdbfile)
    
    # Check that the bond constraints are satisfied:
    bond_dist_bb_out = geometry['bb_bb_distance']
    
    assert_almost_equal(
        bond_dist_bb.value_in_unit(unit.angstrom),
        bond_dist_bb_out.value_in_unit(unit.angstrom),
        decimal=6
    )

    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 36  


def test_optimize_helix_LJ_parameters_2sc_nonequal(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (2sc model, different sigma_sc: optimize sigma, bb-sc, sc-sc bond lengths
    with specified radius, pitch, fixed bb-bb bond lengths)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12

    # Backbone sigma:
    sigma_bb = 1.2 * unit.angstrom

    # Bond constraints (equilibrium values)
    bond_dist_bb = 1.10 * unit.angstrom  
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_2sc_nonequal_test.pdb"

    opt_solution, geometry = optimize_helix_LJ_parameters_2sc(
        radius, pitch, n_particle_bb, sigma_bb, bond_dist_bb,
        equal_sc=False, pdbfile=pdbfile, DE_popsize=20)

    assert os.path.isfile(pdbfile)
    
    # Check that the sigma_sc1 and sigma_sc2 are different:
    sigma_sc1_out = geometry['sigma_sc1']
    sigma_sc2_out = geometry['sigma_sc2']
    
    assert sigma_sc1_out != sigma_sc2_out
    
    # Check that the bond constraints are satisfied:
    bond_dist_bb_out = geometry['bb_bb_distance']
    
    assert_almost_equal(
        bond_dist_bb.value_in_unit(unit.angstrom),
        bond_dist_bb_out.value_in_unit(unit.angstrom),
        decimal=6
    )

    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 36    
    
    
def test_optimize_helix_LJ_parameters_triangle_sc(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (triangle 3sc model: optimize sigma, bb-sc bond length, triangle orientations type 1 and 2,
    with specified radius, pitch, fixed bb-bb bond lengths)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12

    # Backbone sigma:
    sigma_bb = 1.2 * unit.angstrom

    # Bond constraints (equilibrium values)
    bond_dist_bb = 1.10 * unit.angstrom  
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_triangle_sc_test.pdb"

    opt_solution, geometry = optimize_helix_LJ_parameters_triangle_sidechain(
        radius, pitch, n_particle_bb, sigma_bb,
        bond_dist_bb, pdbfile=pdbfile, DE_popsize=20)

    assert os.path.isfile(pdbfile)
    
    # Check that the bond constraints are satisfied:
    bond_dist_bb_out = geometry['bb_bb_distance']
    
    assert_almost_equal(
        bond_dist_bb.value_in_unit(unit.angstrom),
        bond_dist_bb_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    bond_dist_sc_out = geometry['sc_sc_distance']
    sigma_sc_out = geometry['sigma_sc']
    rmin_sc = 2**(1/6)*sigma_sc_out
    
    assert_almost_equal(
        bond_dist_sc_out.value_in_unit(unit.angstrom),
        rmin_sc.value_in_unit(unit.angstrom),
        decimal=6
    )

    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 48     
        
        
def test_optimize_helix_LJ_parameters_triangle_sc_exclusions(tmpdir):
    """
    Test the LJ helix nonbonded parameter optimization function
    (triangle 3sc model: optimize sigma, bb-sc bond length, triangle orientations type 1 and 2,
    with specified radius, pitch, fixed bb-bb bond lengths and custom nonbonded exclusion rules)
    """
    
    # Helical parameters:
    radius = 1.5 * unit.angstrom
    pitch = 1.5 * unit.angstrom

    # Number of backbone particles:
    n_particle_bb = 12

    # Backbone sigma:
    sigma_bb = 1.2 * unit.angstrom

    # Bond constraints (equilibrium values)
    bond_dist_bb = 1.10 * unit.angstrom  
    
    # Set custom exclusion rules:
    exclusions = {
       "default_exclusions": [1,1,1],
    }        
    
    output_directory = tmpdir.mkdir("output")
    pdbfile = f"{output_directory}/helix_opt_LJ_openmm_triangle_sc_test.pdb"

    opt_solution, geometry = optimize_helix_LJ_parameters_triangle_sidechain(
        radius, pitch, n_particle_bb, sigma_bb,
        bond_dist_bb, pdbfile=pdbfile, DE_popsize=20,
        exclusions=exclusions)

    assert os.path.isfile(pdbfile)
    
    # Check that the bond constraints are satisfied:
    bond_dist_bb_out = geometry['bb_bb_distance']
    
    assert_almost_equal(
        bond_dist_bb.value_in_unit(unit.angstrom),
        bond_dist_bb_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    bond_dist_sc_out = geometry['sc_sc_distance']
    sigma_sc_out = geometry['sigma_sc']
    rmin_sc = 2**(1/6)*sigma_sc_out
    
    assert_almost_equal(
        bond_dist_sc_out.value_in_unit(unit.angstrom),
        rmin_sc.value_in_unit(unit.angstrom),
        decimal=6
    )

    # Check that the radius and pitch are satisfied:
    radius_out = geometry['helical_radius']
    pitch_out = geometry['pitch']
    
    assert_almost_equal(
        radius.value_in_unit(unit.angstrom),
        radius_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    assert_almost_equal(
        pitch.value_in_unit(unit.angstrom),
        pitch_out.value_in_unit(unit.angstrom),
        decimal=6
    )
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 48          
        