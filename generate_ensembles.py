#!/usr/bin/python

import os, timeit
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from cg_openmm.src.simulation.tools import build_mm_simulation, run_simulation
from cg_openmm.src.utilities.iotools import read_mm_energies

# OpenMM simulation settings
temperature = 300.0 * unit.kelvin
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.picosecond
simulation_time_step = 5.0 * unit.femtosecond

# Polymer properties
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]

# Particle properties
mass = 10.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
sigmas = {'bb_bb_sigma': 2.5 * unit.angstrom,'bb_sc_sigma': 2.5 * unit.angstrom,'sc_sc_sigma': 2.5 * unit.angstrom}
epsilons = {'bb_bb_eps': 0.5 * unit.kilocalorie_per_mole,'bb_sc_eps': 0.5 * unit.kilocalorie_per_mole,'sc_sc_eps': 0.5 * unit.kilocalorie_per_mole}

# Bonded interaction properties
bond_length = 1.0 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
constrain_bonds = False
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

# Bond angle properties
bond_angle_force_constant = 200 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': bond_angle_force_constant,'bb_sc_sc_angle_k': bond_angle_force_constant,'sc_sc_sc_angle_k': bond_angle_force_constant, 'sc_bb_sc_angle_k': bond_angle_force_constant, 'sc_sc_bb_angle_k': bond_angle_force_constant}
equil_bond_angle = 120
equil_bond_angles = {'bb_bb_bb_angle_0': equil_bond_angle,'bb_bb_sc_angle_0': equil_bond_angle,'bb_sc_sc_angle_0': equil_bond_angle,'sc_sc_sc_angle_0': equil_bond_angle, 'sc_bb_sc_angle_0': equil_bond_angle,'sc_sc_bb_angle_0': equil_bond_angle}

# Torsion properties
torsion_force_constant = 200
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': torsion_force_constant,'bb_bb_sc_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_sc_torsion_k': torsion_force_constant, 'sc_bb_bb_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_bb_torsion_k': torsion_force_constant, 'sc_sc_sc_sc_torsion_k': torsion_force_constant,  'sc_bb_bb_bb_torsion_k': torsion_force_constant}
equil_torsion_angle = 0
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': equil_torsion_angle,'bb_bb_bb_sc_torsion_0': equil_torsion_angle,'bb_bb_sc_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_bb_torsion_0': equil_torsion_angle, 'sc_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_bb_torsion_0': equil_torsion_angle}

top_directory = 'generate_ensembles'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

file = open(str(top_directory)+"/time.dat","w")
file.write("Polymer_length (# monomer units)  Simulation Time (picoseconds)  Wall Clock Time (seconds)")
file.close()

for polymer_length in [8,10,20,30]:
 print("Running simulations with 'polymer_length' ="+str(polymer_length))
 cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=True,include_bond_angle_forces=True,include_torsion_forces=True,constrain_bonds=constrain_bonds)
 number_steps = round(total_simulation_time.__div__(simulation_time_step))
  # Build an OpenMM simulation object
 output_pdb = str(str(top_directory)+'/'+str(polymer_length)+'_'+str(total_simulation_time._value)+'.pdb')
 output_data = str(str(top_directory)+'/'+str(polymer_length)+'_'+str(total_simulation_time._value)+'.dat')
 simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=temperature,simulation_time_step=simulation_time_step,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)
 run_simulation(cgmodel,top_directory,total_simulation_time,simulation_time_step,temperature,print_frequency)
 simulation_data = read_mm_energies(str(str(top_directory)+'/simulation.dat'))

 # Run simulation using CPU
 cpu_start_time = timeit.default_timer()
 simulation.step(number_steps)
 cpu_end_time = timeit.default_timer()
 cpu_time = cpu_end_time - cpu_start_time
 file = open(str(top_directory)+"/time.dat","a")
 file.write(str(polymer_length)+" "+str(total_simulation_time)+" "+str(cpu_time))
 file.close()
exit()
