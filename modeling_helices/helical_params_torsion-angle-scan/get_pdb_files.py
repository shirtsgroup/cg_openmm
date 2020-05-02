import os
import numpy as np
import matplotlib.pyplot as pyplot
from statistics import mean
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import (
    get_mbar_expectation,
    get_free_energy_differences,
    get_temperature_list,
)
from foldamers.thermo.calc import calculate_heat_capacity
from foldamers.parameters.secondary_structure import get_helical_parameters
from cg_openmm.build.cg_build import build_topology
from cg_openmm.simulation.rep_exch import *

grid_size = 4

# Job settings
top_directory = "output"
if not os.path.exists(top_directory):
    os.mkdir(top_directory)

polymer_length = 8
backbone_lengths = [1]
sidechain_lengths = [1]
sidechain_positions = [0]
include_bond_forces = False
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = True

# OpenMM simulation settings
print_frequency = 20  # Number of steps to skip when printing output
total_simulation_time = 0.2 * unit.nanosecond  # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data = str(str(top_directory) + "/output.nc")
number_replicas = 20
min_temp = 100.0 * unit.kelvin
max_temp = 250.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
if total_steps > 10000:
    exchange_attempts = round(total_steps / 1000)
else:
    exchange_attempts = 10

###
#
# Coarse grained model settings
#
###
mass = 100.0 * unit.amu
masses = {"backbone_bead_masses": mass, "sidechain_bead_masses": mass}
bond_length = 7.5 * unit.angstrom
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}

sigma = 2.0 * bond_length
sigmas = {"bb_bb_sigma": sigma, "bb_sc_sigma": sigma, "sc_sc_sigma": sigma}
epsilon = 2.0 * unit.kilocalorie_per_mole
epsilons = {"bb_bb_eps": epsilon, "bb_sc_eps": epsilon, "sc_sc_eps": epsilon}
# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {
    "bb_bb_bond_k": bond_force_constant,
    "bb_sc_bond_k": bond_force_constant,
    "sc_sc_bond_k": bond_force_constant,
}

bond_angle_force_constant = 2000 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
    "bb_sc_sc_angle_k": bond_angle_force_constant,
    "sc_sc_sc_angle_k": bond_angle_force_constant,
    "sc_bb_sc_angle_k": bond_angle_force_constant,
    "sc_sc_bb_angle_k": bond_angle_force_constant,
}
equil_bond_angle = 92
equil_bond_angles = {
    "bb_bb_bb_angle_0": equil_bond_angle,
    "bb_bb_sc_angle_0": equil_bond_angle,
    "bb_sc_sc_angle_0": equil_bond_angle,
    "sc_sc_sc_angle_0": equil_bond_angle,
    "sc_bb_sc_angle_0": equil_bond_angle,
    "sc_sc_bb_angle_0": equil_bond_angle,
}

pitch_list = []
radius_list = []
data_file = "helical_data.dat"
if not os.path.exists(data_file):
    data = open(data_file, "w")
    data.write(
        "BB-BB-BB-BB Torsion (Degrees) SC-BB-BB-SC (Degrees) Pitch (Angstroms) Radius (Angstroms) Monomers-per-turn\n"
    )
    data.close()

torsion_force_constant = 2000
torsion_force_constants = {
    "bb_bb_bb_bb_torsion_k": torsion_force_constant,
    "bb_bb_bb_sc_torsion_k": 0,
    "bb_bb_sc_sc_torsion_k": 0,
    "bb_sc_sc_sc_torsion_k": 0,
    "sc_bb_bb_sc_torsion_k": torsion_force_constant,
    "bb_sc_sc_bb_torsion_k": 0,
    "sc_sc_sc_sc_torsion_k": 0,
    "sc_bb_bb_bb_torsion_k": 0,
}

bb_bb_bb_bb_equil_torsion_angle_range = range(47, 58)
bb_bb_bb_bb_equil_torsion_angles = [
    float(equil_torsion_angle * 3.1415 / 180.0)
    for equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angle_range
]
sc_bb_bb_sc_equil_torsion_angle_range = range(-20, 25, 5)
sc_bb_bb_sc_equil_torsion_angles = [
    float(equil_torsion_angle * 3.1415 / 180.0)
    for equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angle_range
]
equil_torsion_angle = 0.0

for bb_bb_bb_bb_equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angles:
    for sc_bb_bb_sc_equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angles:
        print("Performing simulations for a coarse grained model")
        print(
            "with bb_bb_bb_bb torsion angles of "
            + str(round(bb_bb_bb_bb_equil_torsion_angle * 180.0 / 3.1415, 1))
            + " degrees"
        )
        print(
            "and sc_bb_bb_sc torsion angles of "
            + str(round(sc_bb_bb_sc_equil_torsion_angle * 180.0 / 3.1415, 1))
            + " degrees."
        )
        equil_torsion_angles = {
            "bb_bb_bb_bb_torsion_0": bb_bb_bb_bb_equil_torsion_angle,
            "bb_bb_bb_sc_torsion_0": equil_torsion_angle,
            "bb_bb_sc_sc_torsion_0": equil_torsion_angle,
            "bb_sc_sc_sc_torsion_0": equil_torsion_angle,
            "sc_bb_bb_sc_torsion_0": sc_bb_bb_sc_equil_torsion_angle,
            "bb_sc_sc_bb_torsion_0": equil_torsion_angle,
            "sc_sc_sc_sc_torsion_0": equil_torsion_angle,
            "sc_bb_bb_bb_torsion_0": equil_torsion_angle,
        }
        positions = PDBFile("pose_27.pdb").getPositions()
        cgmodel = CGModel(
            polymer_length=polymer_length,
            backbone_lengths=backbone_lengths,
            sidechain_lengths=sidechain_lengths,
            sidechain_positions=sidechain_positions,
            masses=masses,
            sigmas=sigmas,
            epsilons=epsilons,
            bond_lengths=bond_lengths,
            bond_force_constants=bond_force_constants,
            bond_angle_force_constants=bond_angle_force_constants,
            torsion_force_constants=torsion_force_constants,
            equil_bond_angles=equil_bond_angles,
            equil_torsion_angles=equil_torsion_angles,
            include_nonbonded_forces=include_nonbonded_forces,
            include_bond_forces=include_bond_forces,
            include_bond_angle_forces=include_bond_angle_forces,
            include_torsion_forces=include_torsion_forces,
            constrain_bonds=constrain_bonds,
            positions=positions,
        )

        output_data = str(
            str(top_directory)
            + "/torsions_"
            + str(round(bb_bb_bb_bb_equil_torsion_angle, 2))
            + "_"
            + str(round(sc_bb_bb_sc_equil_torsion_angle, 2))
            + ".nc"
        )
        if os.path.exists(output_data):
            replica_energies, replica_positions, replica_states = read_replica_exchange_data(
                system=cgmodel.system,
                topology=cgmodel.topology,
                temperature_list=temperature_list,
                output_data=output_data,
                print_frequency=print_frequency,
            )
            make_replica_pdb_files(cgmodel.topology, replica_positions)
            output_file = str(
                str(top_directory)
                + "/torsions_"
                + str(round(bb_bb_bb_bb_equil_torsion_angle, 2))
                + "_"
                + str(round(sc_bb_bb_sc_equil_torsion_angle, 2))
                + ".pdb"
            )
            minimum_energy_structures = get_minimum_energy_pose(
                cgmodel.topology, replica_energies, replica_positions, file_name=output_file
            )


exit()
