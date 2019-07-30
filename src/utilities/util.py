import numpy as np
from simtk import openmm as mm
from simtk.openmm import *
from simtk import unit
import simtk.openmm.app.element as elem
from simtk.openmm.app import *


def distance(positions_1,positions_2):
        """
        Construct a matrix of the distances between all particles.

        Parameters
        ----------

        positions_1: Positions for a particle
        ( np.array( length = 3 ) )

        positions_2: Positions for a particle
        ( np.array( length = 3 ) )

        Returns
        -------

        distance
        ( float * unit )
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

        Assign all side lengths for simulation box.

        Parameters
        ----------

        box_size: Simulation box length ( float * simtk.unit.length )

        """

        units = box_size.unit
        a = unit.Quantity(np.zeros([3]), units)
        a[0] = box_size
        b = unit.Quantity(np.zeros([3]), units)
        b[1] = box_size
        c = unit.Quantity(np.zeros([3]), units)
        c[2] = box_size
        return([a,b,c])

def set_box_vectors(system,box_size):
        """

        Build a simulation box.

        Parameters
        ----------

        system: OpenMM system object

        box_size: Simulation box length ( float * simtk.unit.length )

        """

        a,b,c = get_box_vectors(box_size)
        system.setDefaultPeriodicBoxVectors(a, b, c)
        return(system)

def lj_v(positions_1,positions_2,sigma,epsilon):
        dist = distance(positions_1,positions_2)
        quot = sigma.__div__(dist)
        attr = quot.__pow__(6.0)
        rep = quot.__pow__(12.0)
        v = 4.0*epsilon.__mul__(rep.__sub__(attr))
        return(v)


def get_nonbonded_interaction_list(cgmodel):
        interaction_list = []
        bond_list = [[bond[0]-1,bond[1]-1] for bond in cgmodel.get_bond_list()]
        for particle_1 in range(cgmodel.num_beads):
               for particle_2 in range(cgmodel.num_beads):
                 if particle_1 != particle_2:
                   if [particle_1,particle_2] not in bond_list and [particle_2,particle_1] not in bond_list:
                     if [particle_1,particle_2] not in interaction_list:
                       if [particle_2,particle_1] not in interaction_list:
                         interaction_list.append([particle_1,particle_2])
                     if [particle_2,particle_1] not in interaction_list:
                       if [particle_1,particle_2] not in interaction_list:
                         interaction_list.append([particle_2,particle_1])
        return(interaction_list)

def calculate_nonbonded_energy(cgmodel,particle1=None,particle2=None):
        nonbonded_interaction_list = get_nonbonded_interaction_list(cgmodel)
        positions = cgmodel.positions
        energy = unit.Quantity(0.0,cgmodel.epsilon.unit)
        if particle1 != None:
           dist = distance(positions[particle1],positions[particle2])
           inter_energy = lj_v(positions[particle1],positions[particle2],cgmodel.sigma,cgmodel.epsilon).in_units_of(unit.kilojoules_per_mole)
           energy = energy.__add__(inter_energy)
           return(energy)

        for interaction in nonbonded_interaction_list:
           dist = distance(positions[interaction[0]],positions[interaction[1]])
           inter_energy = lj_v(positions[interaction[0]],positions[interaction[1]],cgmodel.sigma,cgmodel.epsilon).in_units_of(unit.kilojoules_per_mole)
           energy = energy.__add__(inter_energy)
        return(energy)

