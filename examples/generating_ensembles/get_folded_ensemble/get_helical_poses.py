import os
import csv
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
import random
from statistics import mean
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.parameters.secondary_structure import *
from foldamers.thermo.calc import *
from foldamers.utilities.iotools import *
from foldamers.utilities.util import random_positions
from cg_openmm.build.cg_build import build_topology
from cg_openmm.simulation.rep_exch import *


native_structure_file = str(str(os.getcwd().split("examples")[0]) + "ensembles/12_1_1_0/helix.pdb")
native_structure = md.load(native_structure_file)

# Job settings
top_directory = "output"
if not os.path.exists(top_directory):
    os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20  # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.nanosecond  # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
number_replicas = 10
min_temp = 10.0 * unit.kelvin
max_temp = 50.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
print("Using " + str(len(temperature_list)) + " replicas.")

cgmodel = CGModel()
output_data = str(str(top_directory) + "/output.nc")

if not os.path.exists(output_data):
    replica_energies, replica_positions, replica_states = run_replica_exchange(
        cgmodel.topology,
        cgmodel.system,
        cgmodel.positions,
        temperature_list=temperature_list,
        simulation_time_step=simulation_time_step,
        total_simulation_time=total_simulation_time,
        print_frequency=print_frequency,
        output_data=output_data,
    )
else:
    replica_energies, replica_positions, replica_states = read_replica_exchange_data(
        system=cgmodel.system,
        topology=cgmodel.topology,
        temperature_list=temperature_list,
        output_data=output_data,
        print_frequency=print_frequency,
    )

rmsd_to_helical = np.zeros((len(replica_positions), len(replica_positions[0])))
make_replica_pdb_files(cgmodel.topology, replica_positions)

rmsd_list = []
ensemble_list = []

rmsd_cutoff = 2.0

for replica_index in range(len(replica_positions)):
    trajectory = md.load(str("replica_" + str(replica_index + 1) + ".pdb"))
    rmsds = md.rmsd(trajectory, native_structure)
    rmsd_to_helical[replica_index] = rmsds
    for rmsd_index in range(len(rmsd_to_helical[replica_index])):
        if rmsd_to_helical[replica_index][rmsd_index] < rmsd_cutoff:
            rmsd_list.append(rmsd_to_helical[replica_index][rmsd_index])

            ensemble_list.append(replica_positions[replica_index][rmsd_index])

random.shuffle(ensemble_list)

file_index = 1
for pose in ensemble_list:
    if file_index <= 100:
        file_name = str("pose_" + str(file_index) + ".pdb")
        cgmodel.positions = pose
        write_pdbfile_without_topology(cgmodel, file_name)
        target_traj = md.load(file_name)
        aligned_target_traj = target_traj.superpose(native_structure)
        cgmodel.positions = unit.Quantity(aligned_target_traj.xyz[0], cgmodel.positions.unit)
        write_pdbfile_without_topology(cgmodel, file_name)
        file_index = file_index + 1
    else:
        exit()
