import os
from statistics import mean
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import *
from cg_openmm.ensembles.ens_build import *
from cg_openmm.simulation.rep_exch import *
from cg_openmm.simulation.tools import *

grid_size = 1

# Job settings
output_directory = "output"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Configure Yank (replica exchange) simulation settings
print_frequency = 5  # Number of steps to skip when printing output
total_simulation_time = 500.0 * unit.picosecond
simulation_time_step = 5.0 * unit.femtosecond
output_data = str(str(output_directory) + "/output.nc")
number_replicas = 30
min_temp = 1.0 * unit.kelvin
max_temp = 400.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)

# Model settings
polymer_length = 12
backbone_lengths = [1]
sidechain_lengths = [1]
sidechain_positions = [0]
include_bond_forces = False
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = True

# Bond definitions
bond_length = 7.5 * unit.angstrom
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}
bond_force_constant = 0 * unit.kilocalorie_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {
    "bb_bb_bond_k": bond_force_constant,
    "bb_sc_bond_k": bond_force_constant,
    "sc_sc_bond_k": bond_force_constant,
}

# Particle definitions
mass = 100.0 * unit.amu
masses = {"backbone_bead_masses": mass, "sidechain_bead_masses": mass}
r_min = 3.0 * bond_length  # Lennard-Jones potential r_min
sigma = r_min / (2.0 ** (1 / 6))  # Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
sigmas = {"bb_sigma": sigma, "sc_sigma": sigma}
epsilon = 0.05 * unit.kilocalorie_per_mole
epsilons = {"bb_eps": epsilon, "sc_eps": epsilon}

# Bond angle definitions
bond_angle_force_constant = 0.0001 * unit.kilocalorie_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
}
bb_bb_bb_equil_bond_angle = 120.0 * (
    3.14 / 180.0
)  # OpenMM expects angle definitions in units of radians
bb_bb_sc_equil_bond_angle = 120.0 * (3.14 / 180.0)
equil_bond_angles = {
    "bb_bb_bb_angle_0": bb_bb_bb_equil_bond_angle,
    "bb_bb_sc_angle_0": bb_bb_sc_equil_bond_angle,
}

# Torsion angle definitions (Used to establish a scanning range below)
torsion_force_constant = 0.01 * unit.kilocalorie_per_mole / unit.radian / unit.radian
torsion_force_constants = {
    "bb_bb_bb_bb_torsion_k": torsion_force_constant,
    "sc_bb_bb_sc_torsion_k": torsion_force_constant,
}
bb_bb_bb_bb_equil_torsion_angle = 78.0 * (
    3.14 / 180.0
)  # OpenMM defaults to units of radians for angle definitions
sc_bb_bb_sc_equil_torsion_angle = 120.0 * (3.14 / 180.0)
equil_torsion_angles = {
    "bb_bb_bb_bb_torsion_0": bb_bb_bb_bb_equil_torsion_angle,
    "sc_bb_bb_sc_torsion_0": sc_bb_bb_sc_equil_torsion_angle,
}
torsion_periodicities = {"bb_bb_bb_bb_period": 1, "sc_bb_bb_sc_period": 2}

# Get initial positions from local file
positions = PDBFile("helix.pdb").getPositions()

# Build a coarse grained model
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
    torsion_periodicities=torsion_periodicities,
    include_nonbonded_forces=include_nonbonded_forces,
    include_bond_forces=include_bond_forces,
    include_bond_angle_forces=include_bond_angle_forces,
    include_torsion_forces=include_torsion_forces,
    constrain_bonds=constrain_bonds,
    positions=positions,
)

if os.path.exists(output_data):
    replica_energies, replica_positions, replica_states = read_replica_exchange_data(
        system=cgmodel.system,
        topology=cgmodel.topology,
        temperature_list=temperature_list,
        output_data=output_data,
        print_frequency=print_frequency,
    )
else:
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
    native_structure = get_native_structure(replica_positions, replica_energies, temperature_list)

native_structure_contact_distance_cutoff = 1.15 * cgmodel.get_sigma(
    0
)  # This distance cutoff determines which nonbonded interactions are considered 'native' contacts
native_fraction_cutoff = (
    0.95  # The threshold fraction of native contacts above which a pose is considered 'native'
)
nonnative_fraction_cutoff = (
    0.95  # The threshold fraction of native contacts below which a pose is considered 'nonnative'
)
native_ensemble_size = 10
nonnative_ensemble_size = 100
decorrelate = True
(
    native_ensemble,
    native_ensemble_energies,
    nonnative_ensemble,
    nonnative_ensemble_energies,
) = get_ensembles_from_replica_positions(
    cgmodel,
    replica_positions,
    replica_energies,
    temperature_list,
    decorrelate=decorrelate,
    native_fraction_cutoff=native_fraction_cutoff,
    nonnative_fraction_cutoff=nonnative_fraction_cutoff,
    native_structure_contact_distance_cutoff=native_structure_contact_distance_cutoff,
    native_ensemble_size=native_ensemble_size,
    nonnative_ensemble_size=nonnative_ensemble_size,
)

native_structure = get_native_structure(replica_positions, replica_energies, temperature_list)

nonnative_ensemble_directory = "nonnative_ensemble"
native_ensemble_directory = "native_ensemble"

if os.path.exists(nonnative_ensemble_directory):
    nonnative_ensemble, nonnative_ensemble_energies = get_ensemble_data(
        cgmodel, nonnative_ensemble_directory
    )
    if len(nonnative_ensemble) != nonnative_ensemble_size:
        print(
            "ERROR: "
            + str(len(nonnative_ensemble_energies))
            + " nonnative poses were found in existing output folders, but "
            + str(nonnative_ensemble_size)
            + " poses were requested."
        )
        print(
            "This probably means that the requested ensemble size changed since the script was last run."
        )
        exit()
else:
    os.mkdir(nonnative_ensemble_directory)
    for pose in nonnative_ensemble:
        cgmodel.positions = pose
        write_ensemble_pdb(cgmodel, ensemble_directory=nonnative_ensemble_directory)

if os.path.exists(native_ensemble_directory):
    native_ensemble, native_ensemble_energies = get_ensemble_data(
        cgmodel, native_ensemble_directory
    )
    if len(native_ensemble_energies) != native_ensemble_size:
        print(
            "ERROR: "
            + str(len(native_ensemble_energies))
            + " native poses were found in existing output folders, but "
            + str(native_ensemble_size)
            + " poses were requested."
        )
        print(
            "This probably means that the requested ensemble size changed since the script was last run."
        )
        exit()
else:
    os.mkdir(native_ensemble_directory)
    for pose in native_ensemble:
        cgmodel.positions = pose
        write_ensemble_pdb(cgmodel, ensemble_directory=native_ensemble_directory)


exit()
