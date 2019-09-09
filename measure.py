
import numpy as np
import math, random, statistics
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from cg_openmm.simulation.tools import minimize_structure
from foldamers.cg_model.cgmodel import *
from foldamers.utilities.iotools import *

def theta(cgmodel,bond_angle):
        """
        Returns the bond angle between three backbone atoms, defined by 'bond_angle'.

        Parameters
        ----------

        cgmodel: CGModel() class object.

        bond_angle: List of the indices for three backbone atoms defining the bond angle we want to evaluate.

        Returns
        -------

        theta: The bond angle between three backbone atoms.

        """

        

        return(theta)

