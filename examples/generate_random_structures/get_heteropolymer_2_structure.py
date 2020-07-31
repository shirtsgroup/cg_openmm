import os
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from simtk.unit import kilojoules_per_mole as kj
from simtk.unit import nanometer as nm

# Coarse grained model settings
include_bond_forces = True
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = False  # not sure why, but these need to be 'False' for now.
constrain_bonds = False
random_positions = True
heteropolymer = True
# Bond definitions

# mass and charge are defaults.
a = {"particle_type_name": "a", "sigma": 0.3 * nm, "epsilon":0.1 * kj }
b = {"particle_type_name": "b", "sigma": 0.4 * nm, "epsilon":0.2 * kj }
c = {"particle_type_name": "c", "sigma": 0.2 * nm, "epsilon":0.3 * kj }
particle_type_list = [a,b,c]

epsilons = {"aa_eps": 0.1 * kj, "ab_eps": 0.2 * kj,"bc_eps": 0.5 * kj}
sigmas = {"aa_sigma": 0.1 * nm, "bc_sigma": 0.2 * nm,"ac_sigma": 0.3 * nm}

bond_lengths = {
    "default_bond_length": 0.2 * nm,
    "a_a_bond_length": 0.3 * nm,
    "a_b_bond_length": 0.3 * nm,
    "b_b_bond_length": 0.3 * nm,
    "b_c_bond_length": 0.4 * nm,
    "a_c_bond_length": 0.4 * nm,
}

bond_force_constants = {
    "default_bond_force_constant": 500 * kj/(nm*nm),
}

bond_angle_force_constants = {
    "default_bond_angle_force_constant": 50 * kj/(unit.radian * unit.radian),
}

equil_bond_angles = {
    "default_equil_bond_angle": 120.0 * unit.degrees,
    "a_a_a_equil_bond_angle" : 60.0 * unit.degrees
}

torsion_force_constants = {
    "a_a_a_b_torsion_force_constant": 20.0 * kj,
}


equil_torsion_angles = {
    "a_a_a_b_equil_torsion_angle": 0.0 * unit.radians,
}

A = {  # a weird triangular backbone, with 2-bead side chain
    "monomer_name": "A",
    "particle_sequence": [a,a,a,b,c],  
    "bond_list": [[0,1], [1,2],[0,2],[1,3],[3,4]],
    "start": 0,
    "end": 2,
}

B = {   # a 3-1 bead
    "monomer_name": "B",
    "particle_sequence": [b,b,b,c],  
    "bond_list": [[0,1],[1,2],[2,3]],
    "start": 0,
    "end": 2,
}

C = {   # a 1-2 bead
    "monomer_name": "C",
    "particle_sequence": [b,c,c],  
    "bond_list": [[0,1], [1,2]],
    "start": 0,
    "end": 0,
}

sequence = 5 * [A, B, C]

# Build a coarse grained model
cgmodel = CGModel(
    particle_type_list=particle_type_list,
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
    random_positions=random_positions,
    sequence=sequence,
    monomer_types=[A, B, C],
)
file_name = "20mer_ABC5_initial_structure.pdb"
write_pdbfile_without_topology(cgmodel, file_name)
