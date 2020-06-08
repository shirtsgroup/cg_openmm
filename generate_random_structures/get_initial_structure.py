import os
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology

# Coarse grained model settings
polymer_length = 24
backbone_lengths = [1]
sidechain_lengths = [1]
sidechain_positions = [0]
include_bond_forces = True
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = False  # not sure why, but these need to be off for now.
constrain_bonds = False
random_positions = True

# Bond definitions
bond_length = 1.0 * unit.angstrom
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}
bond_force_constant = 500.0 * unit.kilojoules_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {
    "bb_bb_bond_k": bond_force_constant,
    "bb_sc_bond_k": bond_force_constant,
    "sc_sc_bond_k": bond_force_constant,
}

# Particle definitions
mass = 100.0 * unit.amu
masses = {"backbone_bead_masses": mass, "sidechain_bead_masses": mass}
r_min = 2.0 * bond_length  # Lennard-Jones potential r_min
sigma = r_min / (2.0 ** (1 / 6))  # Factor of /(2.0**(1/6)) is applied to convert r_min to sigma
sigmas = {"bb_sigma": sigma, "sc_sigma": sigma}
epsilon = 0.1 * unit.kilojoule_per_mole
epsilons = {"bb_eps": epsilon, "sc_eps": epsilon}

# Bond angle definitions
bond_angle_force_constant = 15.0 * unit.kilojoules_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
    "bb_sc_sc_angle_k": bond_angle_force_constant,
}
bb_bb_bb_equil_bond_angle = 120.0 * unit.degrees
bb_bb_sc_equil_bond_angle = 120.0 * unit.degrees
equil_bond_angles = {
    "bb_bb_bb_angle_0": bb_bb_bb_equil_bond_angle,
    "bb_bb_sc_angle_0": bb_bb_sc_equil_bond_angle,
    "bb_sc_sc_angle_0": bb_bb_sc_equil_bond_angle,
}

# Torsion angle definitions                                                                                                     
torsion_force_constant = 0 * unit.kilojoule_per_mole
torsion_force_constants = {
    "bb_bb_bb_bb_torsion_k": torsion_force_constant,
    "bb_bb_bb_sc_torsion_k": torsion_force_constant,
    "bb_bb_sc_sc_torsion_k": torsion_force_constant
}

equil_torsion_angle = 0.0 * unit.degrees

equil_torsion_angles = {
    "bb_bb_bb_bb_torsion_0":equil_torsion_angle,
    "bb_bb_bb_sc_torsion_0":equil_torsion_angle,
    "bb_bb_sc_sc_torsion_0":equil_torsion_angle
}
torsion_periodicities = {
    "bb_bb_bb_bb_period": 3,
    "bb_bb_bb_sc_period": 1,
    "bb_bb_bb_sc_period": 1
}

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
    random_positions=random_positions,
)
import pdb
pdb.set_trace()
#file_name = "12mer_3b1s_initial_structure.pdb"
#file_name = "12mer_1b1s_initial_structure.pdb"
file_name = "24mer_1b1s_initial_structure.pdb"
#file_name = "12mer_1b2s_initial_structure.pdb"  # this one looks a bit off, and takes a long time
#file_name = "12mer_2b1s_initial_structure.pdb"
#file_name = "12mer_2b2s_initial_structure.pdb"


write_pdbfile_without_topology(cgmodel, file_name)

