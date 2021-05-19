"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed
from simtk import unit
import pytest
import mdtraj as md
import sys
import os
from cg_openmm.utilities.helix_modeling import *
import numpy as np
from numpy.testing import assert_almost_equal


def test_optimize_helix(tmpdir):
    """
    Test the LJ helix optimization function
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
    plotfile = f"{output_directory}/LJ_helix.pdf"
    
    opt_solution, geometry = optimize_helix(
        n_particle_bb,sigma,epsilon,sidechain,
        pdbfile=pdbfile, plotfile=plotfile)
           
    assert opt_solution.success == True
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 12
        
        
def test_optimize_helix_sidechain(tmpdir):
    """
    Test the LJ helix optimization function
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
    plotfile = f"{output_directory}/LJ_helix_sidechain.pdf"
    
    opt_solution, geometry = optimize_helix(
        n_particle_bb,sigma,epsilon,sidechain,
        pdbfile=pdbfile, plotfile=plotfile)
           
    assert opt_solution.success == True
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24
    

def test_optimize_helix_openmm_unconstrained(tmpdir):
    """
    Test the LJ helix optimization function
    (openmm energy with nonbonded exclusions, unconstrained bond lengths)
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
    plotfile = f"{output_directory}/LJ_helix_openmm_unconstrained.pdf"
    
    opt_solution, geometry = optimize_helix_openmm_energy(
        n_particle_bb, sigma_bb, sigma_sc, epsilon_bb, epsilon_sc,
        pdbfile=pdbfile, plotfile=plotfile, bond_dist_bb=None, bond_dist_sc=None,
        DE_popsize=50)
           
    assert opt_solution.success == True
    assert os.path.isfile(pdbfile)
    assert os.path.isfile(plotfile)
    
    # Check that we can load the pdb file
    pdb_loaded = md.load(pdbfile)
    assert pdb_loaded.n_atoms == 24
    

def test_optimize_helix_openmm_constrained(tmpdir):
    """
    Test the LJ helix optimization function
    (openmm energy with nonbonded exclusions, fixed bond lengths)
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
    plotfile = f"{output_directory}/LJ_helix_openmm_constrained.pdf"
    
    opt_solution, geometry = optimize_helix_openmm_energy(
        n_particle_bb, sigma_bb, sigma_sc, epsilon_bb, epsilon_sc,
        pdbfile=pdbfile, plotfile=plotfile,
        bond_dist_bb=bond_dist_bb, bond_dist_sc=bond_dist_sc,
        DE_popsize=50)
               
    assert opt_solution.success == True
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
    
    