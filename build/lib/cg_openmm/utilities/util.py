import numpy as np
from simtk import openmm as mm
from simtk.openmm import *
from simtk import unit
import simtk.openmm.app.element as elem
from simtk.openmm.app import *


def distance(positions_1,positions_2):
        """
        Calculate the distance between two particles, given their positions.

        :param positions_1: Positions for the first particle
        :type positions_1: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [3] ), simtk.unit )

        :param positions_2: Positions for the first particle
        :type positions_2: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [3] ), simtk.unit )

        :returns:
            - distance ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_) - Distance between two particles

        :Example:

        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> cgmodel = CGModel()
        >>> particle_1_coordinates = cgmodel.positions[0]
        >>> particle_2_coordinates = cgmodel.positions[1]
        >>> particle_distance = distance(particle_1_coordinates,particle_2_coordinates)
  
        """

        direction_comp = np.zeros(3) * positions_1.unit

        for direction in range(len(direction_comp)):
          direction_comp[direction] = positions_1[direction].__sub__(positions_2[direction])

        direction_comb = np.zeros(3) * positions_1.unit.__pow__(2.0)
        for direction in range(3):
          direction_comb[direction] = direction_comp[direction].__pow__(2.0)

        sqrt_arg = direction_comb[0].__add__(direction_comb[1]).__add__(direction_comb[2])

        value = math.sqrt(sqrt_arg._value)
        units = sqrt_arg.unit.sqrt()
        distance = unit.Quantity(value=value,unit=units)

        return(distance)

def get_box_vectors(box_size):
        """
        Given a simulation box length, construct a vector.

        :param box_size: Length of individual sides of a simulation box
        :type box_size: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( float, simtk.unit )

        :returns: 
             - box_vectors ( List( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) - Vectors to use when defining an OpenMM simulation box.

        """

        units = box_size.unit
        a = unit.Quantity(np.zeros([3]), units)
        a[0] = box_size
        b = unit.Quantity(np.zeros([3]), units)
        b[1] = box_size
        c = unit.Quantity(np.zeros([3]), units)
        c[2] = box_size
        box_vectors = [a,b,c]
        return(box_vectors)

def set_box_vectors(system,box_size):
        """
        Impose a set of simulation box vectors on an OpenMM simulation object.

        :param system: OpenMM System()
        :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

        :param box_size: Length of individual sides of a simulation box
        :type box_size: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( float, simtk.unit )

        :returns: 
            - system (`System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_) - OpenMM system object

        """

        a,b,c = get_box_vectors(box_size)
        system.setDefaultPeriodicBoxVectors(a, b, c)
        return(system)

def lj_v(positions_1,positions_2,sigma,epsilon):
        """
        Calculate the Lennard-Jones interaction energy between two particles, given their positions and definitions for their equilbrium interaction distance (sigma) and strength (epsilon).

        :param positions_1: Positions for the first particle
        :type positions_1: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [3] ), simtk.unit )

        :param positions_2: Positions for the first particle
        :type positions_2: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [3] ), simtk.unit )

        :param sigma: Lennard-Jones equilibrium interaction distance for two non-bonded particles
        :type sigma: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :param epsilon: Lennard-Jones equilibrium interaction energy for two non-bonded particles.
        :type epsilon: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

        :returns:
           - v ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - Lennard-Jones interaction energy

        """

        dist = distance(positions_1,positions_2)
        quot = sigma.__div__(dist)
        attr = quot.__pow__(6.0)
        rep = quot.__pow__(12.0)
        v = 4.0*epsilon.__mul__(rep.__sub__(attr))
        return(v)

