import os
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology

# Coarse grained model settings
include_bond_forces = True
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = False  # not sure why, but these need to be 'False' for now.
constrain_bonds = False
random_positions = True
heteropolymer = True
# Bond definitions
bond_length = 1.0 * unit.angstrom

a_bond_lengths = {
    "bb_bb_bond_length": 2.0*bond_length,
    "bb_sc_bond_length": 2.0*bond_length,
    "sc_sc_bond_length": 2.0*bond_length,
}

b_bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}

c_bond_lengths = {
    "bb_bb_bond_length": 1.5*bond_length,
    "bb_sc_bond_length": 1.5*bond_length,
    "sc_sc_bond_length": 1.5*bond_length,
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
a_sigmas = {"bb_sigma": sigma, "sc_sigma": sigma}
b_sigmas = {"bb_sigma": 0.8*sigma, "sc_sigma": 0.8*sigma}
c_sigmas = {"bb_sigma": 1.2*sigma, "sc_sigma": 1.2*sigma}

epsilon = 0.1 * unit.kilojoule_per_mole
a_epsilons = {"bb_eps": epsilon, "sc_eps": epsilon}
b_epsilons = {"bb_eps": 2*epsilon, "sc_eps": 2*epsilon}
c_epsilons = {"bb_eps": 3*epsilon, "sc_eps": 3*epsilon}

# Bond angle definitions
bond_angle_force_constant = 15.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
    "bb_sc_sc_angle_k": bond_angle_force_constant,
}

bb_bb_bb_equil_bond_angle = 120.0 * unit.degrees
bb_bb_sc_equil_bond_angle = 120.0 * unit.degrees
bb_sc_sc_equil_bond_angle = 120.0 * unit.degrees
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
    "bb_bb_sc_sc_torsion_k": torsion_force_constant,
}


equil_torsion_angle = 0.0 * unit.degrees
equil_torsion_angles = {
    "bb_bb_bb_bb_torsion_0": equil_torsion_angle,
    "bb_bb_bb_sc_torsion_0": equil_torsion_angle,
    "bb_bb_sc_sc_torsion_0": equil_torsion_angle,
}
torsion_periodicities = {
    "bb_bb_bb_bb_period": 3,
    "bb_bb_bb_sc_period": 1,
    "bb_bb_sc_sc_period": 1}

A = {'monomer_name': "A", 'backbone_length': 1, 'sidechain_length': 1, 'sidechain_positions': [0], 'bond_lengths': a_bond_lengths, 'epsilons': a_epsilons, 'sigmas': a_sigmas}
B = {'monomer_name': "B", 'backbone_length': 2, 'sidechain_length': 1, 'sidechain_positions': [0], 'bond_lengths': b_bond_lengths, 'epsilons': b_epsilons, 'sigmas': b_sigmas}  
C = {'monomer_name': "C", 'backbone_length': 3, 'sidechain_length': 1, 'sidechain_positions': 1, 'bond_lengths': c_bond_lengths, 'epsilons': c_epsilons, 'sigmas': c_sigmas}

sequence = 4*[A,B,C]

# Build a coarse grained model
cgmodel = CGModel(
    masses=masses,
    bond_lengths=a_bond_lengths,
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
    sequence=sequence,
    monomer_types=[A,B,C]
)
file_name = "12mer_ABC4s_initial_structure.pdb"
write_pdbfile_without_topology(cgmodel, file_name)
