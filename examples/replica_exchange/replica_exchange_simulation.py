#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.src.cg_model.cgmodel import CGModel
from cg_openmm.src.simulation.rep_exch import replica_exchange
from yank.multistate import MultiStateReporter
from yank.multistate import ReplicaExchangeAnalyzer

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)
run_simulations = False

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas =10
temperature_increment = 5 # unit.kelvin
temperature_list = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(0,number_replicas*temperature_increment,temperature_increment)]
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

###
#
# Coarse grained model settings
#
###

# Global definitions
polymer_length=8
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]
include_bond_forces=False
include_bond_angle_forces=True
include_nonbonded_forces=True
include_torsion_forces=True
constrain_bonds = True

# Particle properties
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
sigma = 18.0 * unit.angstrom
sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma}
epsilon = 0.5 * unit.kilocalorie_per_mole
epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': 0.5 * epsilon}

# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
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

cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds)

# Run a replica exchange simulation with this cgmodel
if run_simulations:
  replica_energies,replica_positions,replica_states,temperature_list = replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
else:
  reporter = MultiStateReporter(output_data, open_mode='r', checkpoint_interval=print_frequency)
  analyzer = ReplicaExchangeAnalyzer(reporter)
  replica_energies,unsampled_state_energies,neighborhoods,replica_states = analyzer.read_energies()
  total_steps = len(replica_energies[0][0])
  replica_positions = unit.Quantity(np.zeros([len(temperature_list),len(temperature_list),total_steps,cgmodel.system.getNumParticles(),3]),unit.nanometer)

  for step in range(total_steps):
    sampler_states = reporter.read_sampler_states(iteration=step)
    for replica_index in range(len(temperature_list)):
      for thermodynamic_state_index in range(len(temperature_list)):
        for particle in range(cgmodel.system.getNumParticles()):
          for cart in range(3):
            replica_positions[replica_index][thermodynamic_state_index][step][particle][cart] = sampler_states[replica_index].positions[particle][cart]

steps_per_stage = round(total_steps/exchange_attempts)

# Get the minimum energy structure sampled during the simulation
minimum_energy = 0.0
for replica in range(len(replica_energies)):
  energies = np.array([energy for energy in replica_energies[replica][replica]])
  for energy in range(len(energies)):
    if energies[energy] < minimum_energy:
      minimum_energy = energies[energy]
      minimum_energy_structure = replica_positions[replica][replica][energy]
file = open(str("re_min.pdb"),"w")
PDBFile.writeFile(cgmodel.topology,minimum_energy_structure,file=file)

figure = pyplot.figure(0)
for replica in range(len(replica_energies)):
  simulation_times = np.array([float(int(step*steps_per_stage)*simulation_time_step.in_units_of(unit.picosecond)._value) for step in range(len(replica_energies[replica][replica]))])
  energies = np.array([float(energy) for energy in replica_energies[replica][replica]])
  pyplot.plot(simulation_times,energies,figure=figure)
pyplot.xlabel("Simulation Time ( Picoseconds )")
pyplot.ylabel("Potential Energy ( kJ / mol )")
pyplot.title("Replica Exchange Simulation")
pyplot.legend([temperature._value for temperature in temperature_list])
pyplot.savefig(str(str(top_directory)+"/replica_exchange_energies.png"))
pyplot.show()
pyplot.close()

figure = pyplot.figure(1)
for replica in range(len(replica_states)):
  simulation_times = np.array([float(int(step*steps_per_stage)*simulation_time_step.in_units_of(unit.picosecond)._value) for step in range(len(replica_states[replica]))])
  state_indices = np.array([int(round(state)) for state in replica_states[replica]])
  pyplot.plot(simulation_times,state_indices,figure=figure)
pyplot.xlabel("Simulation Time ( Picoseconds )")
pyplot.ylabel("Thermodynamic State Index")
pyplot.title("State Exchange Summary")
pyplot.legend([i for i in range(len(replica_states))])
pyplot.savefig(str(str(top_directory)+"/replica_exchange_state_transitions.png"))
pyplot.show()
pyplot.close()

exit()
