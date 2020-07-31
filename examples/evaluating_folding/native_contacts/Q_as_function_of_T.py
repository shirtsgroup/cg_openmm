###
#
# This script can be used to determine suitable
# settings for calculating 'native contacts'.
#
###

import os
import numpy as np
from statistics import mean
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.ensembles.ens_build import *
from foldamers.parameters.secondary_structure import *
from cg_openmm.simulation.rep_exch import *

output_data = "output.nc"
number_replicas = 12
min_temp = 50.0 * unit.kelvin
max_temp = 600.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
list_native_contacts = [[1,7],[3,9],[5,11],[7,13],[9,15],[11,17],[13,19],[15,21],[17,23]] # list of native contacts.  This one is arbitrary, it's every 3rd sidechain pair.

# This distance cutoff determines which nonbonded interactions are considered 'native' contacts
native_structure_contact_distance_cutoff = 0.6 * unit.nanometers 
# this could also be done in terms of a fraction of the original native contact, that would need to be coded
# in a different way.

results = fraction_contacts_expectation(list_native_contacts, native_structure_contact_distance_cutoff,
                                         temperature_list, output_directory="output", output_data="output.nc",
                                         num_intermediate_states=1)

Tunit = temperature_list[0].unit
pyplot.xlabel(f"Temperature ({Tunit})")
pyplot.ylabel("<Q> (Fraction native contacts)")
pyplot.errorbar(results['T'], results['Q'], yerr=results['dQ'])
pyplot.savefig("Q_vs_T.png")
pyplot.show()

