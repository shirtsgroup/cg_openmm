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

mass = 100.0 * unit.amu
# Particle definitions

a_sigma = 1.8 * unit.angstroms
b_sigma = 1.4 * unit.angstroms
c_sigma = 2.2 * unit.angstroms

a_epsilon = 0.1 * unit.kilojoule_per_mole
b_epsilon = 0.2 * unit.kilojoule_per_mole
c_epsilon = 0.3 * unit.kilojoule_per_mole

abb = {"particle_type_name": "abb", "mass" : mass, "sigma" : a_sigma, "epsilon" : a_epsilon}
asc = {"particle_type_name": "asc", "mass" : mass, "sigma" : a_sigma, "epsilon" : a_epsilon}
bbb = {"particle_type_name": "bbb", "mass" : mass, "sigma" : b_sigma, "epsilon" : b_epsilon}
bsc = {"particle_type_name": "bsc", "mass" : mass, "sigma" : b_sigma, "epsilon" : b_epsilon}
cbb = {"particle_type_name": "cbb", "mass" : mass, "sigma" : c_sigma, "epsilon" : c_epsilon}
csc = {"particle_type_name": "csc", "mass" : mass, "sigma" : c_sigma, "epsilon" : c_epsilon}
particle_type_list = [abb,asc,bbb,bsc,cbb,csc]

bond_length = 2.0 * unit.angstroms
bond_lengths = {
    "default_bond_length" : bond_length,
    "bbb_bbb_bond_length" : 0.5 * bond_length,
    "bbb_bsc_bond_length" : 0.5 * bond_length,
    "bsc_bsc_bond_length" : 0.5 * bond_length,
    "cbb_cbb_bond_length" : 1.5 * bond_length,
    "cbb_csc_bond_length" : 1.5 * bond_length,
    "csc_csc_bond_length" : 1.5 * bond_length,
}

bond_force_constants = {
    "default_bond_force_constant" : 500.0 * unit.kilojoules_per_mole / unit.nanometer / unit.nanometer
}

bond_angle_force_constants = {
    "default_bond_angle_force_constant" : 15.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
}

equil_bond_angles = {
    "default_equil_bond_angle" : 120.0 * unit.degrees
}

A = {
    "monomer_name": "A",
    "particle_sequence": [abb,asc],
    "bond_list" : [[0,1]], # sidechain is on first bead
    "start": 0,
    "end": 0
}

B = {
    "monomer_name": "B",
    "particle_sequence": [bbb,bbb,bsc],
    "bond_list" : [[0,1],[0,2]],  # sidechain is on first bead
    "start": 0,
    "end": 1
}

C = {
    "monomer_name": "C",
    "particle_sequence": [cbb,cbb,cbb,csc],
    "bond_list" : [[0,1],[1,2],[1,3]], # sidechain is on 2nd bead
    "start": 0,
    "end": 2
}

sequence = 4 * [A, B, C]

# Build a coarse grained model
cgmodel = CGModel(
    particle_type_list = particle_type_list,
    bond_lengths=bond_lengths,
    bond_force_constants=bond_force_constants,
    bond_angle_force_constants=bond_angle_force_constants,
    equil_bond_angles=equil_bond_angles,
    include_nonbonded_forces=include_nonbonded_forces,
    include_bond_forces=include_bond_forces,
    include_bond_angle_forces=include_bond_angle_forces,
    include_torsion_forces=include_torsion_forces,
    constrain_bonds=constrain_bonds,
    random_positions=random_positions,
    sequence=sequence,
    monomer_types=[A, B, C],
)
file_name = "12mer_ABC4s_initial_structure.pdb"
write_pdbfile_without_topology(cgmodel, file_name)
