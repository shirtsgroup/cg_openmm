"""
Unit and regression test for the cg_openmm package.
"""

# Import package, test suite, and other packages as needed
import cg_openmm
import pytest
import sys

def test_cg_openmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cg_openmm" in sys.modules
