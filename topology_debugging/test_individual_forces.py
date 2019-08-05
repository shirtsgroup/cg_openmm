#!/usr/bin/python

import os
from simtk import unit
from simtk.openmm import LangevinIntegrator
import matplotlib.pyplot as pyplot
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.utilities.util import random_positions
from cg_openmm.src.simulation.tools import build_mm_simulation, run_simulation
from cg_openmm.src.build.cg_build import *
from cg_openmm.src.utilities.iotools import *

###
#
# OpenMM simulation settings
#
###

temperature = 300.0 * unit.kelvin
print_frequency = 1 # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = total_simulation_time.__div__(simulation_time_step)

###
#
# Coarse grained model settings
#
###

# Global definitions
polymer_length=10
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]

# Particle properties
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
sigmas = {'bb_bb_sigma': 18.5 * unit.angstrom,'bb_sc_sigma': 18.5 * unit.angstrom,'sc_sc_sigma': 18.5 * unit.angstrom}
epsilons = {'bb_bb_eps': 0.5 * unit.kilocalorie_per_mole,'bb_sc_eps': 0.5 * unit.kilocalorie_per_mole,'sc_sc_eps': 0.5 * unit.kilocalorie_per_mole}

# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
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

top_directory = 'test_individual_forces'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

dir = str(str(top_directory)+'/no_forces')
if not os.path.exists(dir): os.mkdir(dir)
cg_model = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=False,include_bond_forces=False,include_bond_angle_forces = False,include_torsion_forces=False,constrain_bonds=constrain_bonds)
cg_model.positions = random_positions(cg_model,use_library=False)
positions = cg_model.positions
integrator = LangevinIntegrator(temperature._value,0.1,simulation_time_step.in_units_of(unit.picosecond)._value)
simulation = build_mm_simulation(cg_model.topology, cg_model.system,positions,temperature=temperature,simulation_time_step=simulation_time_step,print_frequency=1)
simulation.context.setPositions(cg_model.positions)
forces = simulation.context.getState(getForces=True).getForces()
print(forces)
run_simulation(cg_model,dir,total_simulation_time,simulation_time_step,temperature,print_frequency)
#energies_no_forces = read_mm_energies(str(str(top_directory)+'/no_forces/simulation.dat'))

dir = str(str(top_directory)+'/nonbonded_forces')
if not os.path.exists(dir): os.mkdir(dir)
cg_model = CGModel(positions=positions,polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=False,include_bond_angle_forces = False,include_torsion_forces=False,constrain_bonds=constrain_bonds)
integrator = LangevinIntegrator(temperature._value, total_simulation_time.in_units_of(unit.picosecond)._value, simulation_time_step.in_units_of(unit.picosecond)._value)
simulation = Simulation(cg_model.topology, cg_model.system, integrator)
simulation.context.setPositions(cg_model.positions)
forces = simulation.context.getState(getForces=True).getForces()
print(forces)

#run_simulation(cgmodel,dir,total_simulation_time,simulation_time_step,temperature,print_frequency)
#energies_nonbonded_forces = read_mm_energies(str(str(top_directory)+'/nonbonded_forces/simulation.dat'))

dir = str(str(top_directory)+'/bond_forces')
if not os.path.exists(dir): os.mkdir(dir)
cg_model = CGModel(positions=positions,polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=True,include_bond_angle_forces = False,include_torsion_forces=False,constrain_bonds=constrain_bonds)
integrator = LangevinIntegrator(temperature._value, total_simulation_time.in_units_of(unit.picosecond)._value, simulation_time_step.in_units_of(unit.picosecond)._value)
simulation = Simulation(cg_model.topology, cg_model.system, integrator)
simulation.context.setPositions(cg_model.positions)
forces = simulation.context.getState(getForces=True).getForces()
print(forces)

#run_simulation(cgmodel,dir,total_simulation_time,simulation_time_step,temperature,print_frequency)
#energies_bond_forces = read_mm_energies(str(str(top_directory)+'/bond_forces/simulation.dat'))

dir = str(str(top_directory)+'/bond_angle_forces')
if not os.path.exists(dir): os.mkdir(dir)
cg_model = CGModel(positions=positions,polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=True,include_bond_angle_forces = True,include_torsion_forces=False,constrain_bonds=constrain_bonds)

integrator = LangevinIntegrator(temperature._value, total_simulation_time.in_units_of(unit.picosecond)._value, simulation_time_step.in_units_of(unit.picosecond)._value)
simulation = Simulation(cg_model.topology, cg_model.system, integrator)
simulation.context.setPositions(cg_model.positions)
forces = simulation.context.getState(getForces=True).getForces()
print(forces)

#run_simulation(cgmodel,dir,total_simulation_time,simulation_time_step,temperature,print_frequency)
#energies_bond_angle_forces = read_mm_energies(str(str(top_directory)+'/bond_angle_forces/simulation.dat'))

dir = str(str(top_directory)+'/torsion_forces')
if not os.path.exists(dir): os.mkdir(dir)
cg_model = CGModel(positions=positions,polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=True,include_bond_angle_forces = True,include_torsion_forces=True,constrain_bonds=constrain_bonds)

integrator = LangevinIntegrator(temperature._value, total_simulation_time.in_units_of(unit.picosecond)._value, simulation_time_step.in_units_of(unit.picosecond)._value)
simulation = Simulation(cg_model.topology, cg_model.system, integrator)
simulation.context.setPositions(cg_model.positions)
forces = simulation.context.getState(getForces=True).getForces()
print(forces)

#run_simulation(cgmodel,dir,total_simulation_time,simulation_time_step,temperature,print_frequency)
#energies_torsion_forces = read_mm_energies(str(str(top_directory)+'/torsion_forces/simulation.dat'))

dir = str(str(top_directory)+'/all_forces')
if not os.path.exists(dir): os.mkdir(dir)
cg_model = CGModel(positions=positions,polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=True,include_bond_forces=True,include_bond_angle_forces =True,include_torsion_forces=True,constrain_bonds=constrain_bonds)
integrator = LangevinIntegrator(temperature._value, total_simulation_time.in_units_of(unit.picosecond)._value, simulation_time_step.in_units_of(unit.picosecond)._value)
simulation = Simulation(cg_model.topology, cg_model.system, integrator)
simulation.context.setPositions(cg_model.positions)
forces = simulation.context.getState(getForces=True).getForces()
print(forces)

#run_simulation(cgmodel,dir,total_simulation_time,simulation_time_step,temperature,print_frequency)
#energies_all_forces = read_mm_energies(str(str(top_directory)+'/all_forces/simulation.dat'))


# Plot results

figure = pyplot.figure(0)
y_max = 0.0

if len(energies_no_forces['step']) > 0:
 x_no_forces = np.array([float(int(step)*simulation_time_step._value) for step in energies_no_forces['step']])
 y_no_forces = np.array([float(energy) for energy in energies_no_forces['potential_energy']])
 y_max = y_no_forces.max()
 pyplot.plot(x_no_forces,y_no_forces,figure=figure,color='black',label='No forces')
if len(energies_nonbonded_forces['step']) > 0:
 x_nonbonded_forces = np.array([float(int(step)*simulation_time_step._value) for step in energies_nonbonded_forces['step']])
 y_nonbonded_forces = np.array([float(energy) for energy in energies_nonbonded_forces['potential_energy']])
# if y_nonbonded_forces.max() > y_max: ymax = y_nonbonded_forces.max()
 pyplot.plot(x_nonbonded_forces,y_nonbonded_forces,figure=figure,color='purple',label='Nonbonded only')
if len(energies_no_forces['step']) > 0:
 x_bond_forces = np.array([float(int(step)*simulation_time_step._value) for step in energies_bond_forces['step']])
 y_bond_forces = np.array([float(energy) for energy in energies_bond_forces['potential_energy']])
# if y_bond_forces.max() > y_max: ymax = y_bond_forces.max()
 pyplot.plot(x_bond_forces,y_bond_forces,figure=figure,color='green',label='Bonds only')
if len(energies_no_forces['step']) > 0:
 x_bond_angle_forces = np.array([float(int(step)*simulation_time_step._value) for step in energies_bond_angle_forces['step']])
 y_bond_angle_forces = np.array([float(energy) for energy in energies_bond_angle_forces['potential_energy']])
# if y_bond_angle_forces.max() > y_max: ymax = y_bond_angle_forces.max()
 pyplot.plot(x_bond_angle_forces,y_bond_angle_forces,figure=figure,color='orange',label='Bond angles only')
if len(energies_no_forces['step']) > 0:
 x_torsion_forces = np.array([float(int(step)*simulation_time_step._value) for step in energies_torsion_forces['step']])
 y_torsion_forces = np.array([float(energy) for energy in energies_torsion_forces['potential_energy']])
# if y_torsion_forces.max() > y_max: ymax = y_torsion_forces.max()
 pyplot.plot(x_torsion_forces,y_torsion_forces,figure=figure,color='blue',label='Torsions only')
if len(energies_all_forces['step']) > 0:
 x_all_forces = np.array([float(int(step)*simulation_time_step._value) for step in energies_all_forces['step']])
 y_all_forces = np.array([float(energy) for energy in energies_all_forces['potential_energy']])
# if y_all_forces.max() > y_max: ymax = y_all_forces.max()
 pyplot.plot(x_all_forces,y_all_forces,figure=figure,color='red',label='All Forces')

pyplot.xlabel("Simulation Time ( Femtoseconds )")
pyplot.ylabel("Potential Energy ( kJ / mol )")
pyplot.title("Potential Energy vs. Simulation Time")
pyplot.savefig(str(str(top_directory)+"/potential_energies_for_individual_forces.png"))
pyplot.legend()
pyplot.show()
pyplot.close()


exit()
