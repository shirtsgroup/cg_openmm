import os
from simtk.openmm.app.forcefield import ForceField
import simtk.openmm.app.element as elem
from simtk.openmm.app.element import Element
from simtk.openmm.app import PDBFile
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.iotools import write_pdbfile_without_topology
from foldamers.src.utilities.util import random_positions
from cg_openmm.src.build.cg_build import write_xml_file, build_system, build_topology
from cg_openmm.src.simulation.tools import run_simulation, build_mm_simulation, minimize_structure
from cg_openmm.src.utilities.iotools import read_mm_energies
from cg_openmm.src.simulation.rep_exch import replica_exchange
temperature = 300.0 * unit.kelvin
print_frequency = 1 # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond

cg_model = CGModel()

simulation = build_mm_simulation(cg_model.topology,cg_model.system,cg_model.positions,simulation_time_step=simulation_time_step,print_frequency=1)

simulation.step(1000)

simulation_energies = read_mm_energies("output.dat")

print(simulation_energies)

replica_energies,reduced_replica_energies,replica_positions,temperature_list = replica_exchange(cg_model.topology,cg_model.system,cg_model.positions,simulation_time_step=5.0*unit.femtosecond)

print(replica_energies)

exit()
