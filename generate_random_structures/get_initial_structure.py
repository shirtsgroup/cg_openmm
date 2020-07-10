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

# Bond definitions
bond_lengths = {
    "default_bond_length": 1.0 * unit.angstrom,
}

bond_force_constant = 500.0 * unit.kilojoules_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {
    "default_bond_force_constant" : bond_force_constant
}

# Particle definitions
# mass and charge are defaults.
bb = {
    "particle_type_name": "bb",
    "sigma": 0.2 * unit.nanometers,
    "epsilon":0.1 * unit.kilojoules_per_mole,
    "mass":100.0 * unit.amu
}
sc = {
    "particle_type_name": "sc",
    "sigma": 0.2 * unit.nanometers,
    "epsilon":0.1 * unit.kilojoules_per_mole,
    "mass":100.0 * unit.amu }

# Bond angle definitions
bond_angle_force_constants = {
    "default_bond_angle_force_constant" : 15.0 * unit.kilojoule_per_mole / unit.radian / unit.radian
}

equil_bond_angles = {
    "default_equil_bond_angle" : 120.0 * unit.degrees
}

# Torsion angle definitions
torsion_force_constants = {
    "default_torsion_force_constant" : 10 * unit.kilojoule_per_mole
}

equil_torsion_angles = {
    "default_equil_torsion_angle" : 0.0 * unit.degrees
}

torsion_periodicities = {
    "default_torsion_periodicity": 1,
    "bb_bb_bb_bb_period": 3,
}

# define the monomer
A = {
    "monomer_name": "A",
    "particle_sequence":[bb,bb,sc],
    "bond_list":[[0,1],[1,2]],
    "start": 0,
    "end": 1
}

sequence = 12 * [A]

# Build a coarse grained model
cgmodel = CGModel(
    particle_type_list=[bb,sc],
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
    sequence=sequence,
    constrain_bonds=constrain_bonds,
    random_positions=random_positions,
    monomer_types=[A],
)
file_name = "12mer_2b1s_initial_structure.pdb"
# file_name = "12mer_1b1s_initial_structure.pdb"
# file_name = "24mer_1b1s_initial_structure.pdb"
# file_name = "12mer_1b2s_initial_structure.pdb"  # this one looks a bit off, and takes a long time
# file_name = "12mer_2b1s_initial_structure.pdb"
# file_name = "12mer_2b2s_initial_structure.pdb"


write_pdbfile_without_topology(cgmodel, file_name)
