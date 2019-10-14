import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.thermo.calc import calculate_heat_capacity
from foldamers.utilities.util import random_positions
from cg_openmm.build.cg_build import build_topology
from cg_openmm.simulation.rep_exch import *
from foldamers.ensembles.ens_build import *
from foldamers.thermo.calc import *

# Job settings
output_directory = 'output'
if not os.path.exists(output_directory):
  os.mkdir(output_directory)

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time_list = [5.0 * unit.nanosecond,10.0 * unit.nanosecond, 15.0 * unit.nanosecond]
simulation_time_step = 5.0 * unit.femtosecond

# Yank (replica exchange) simulation settings
output_data=str(str(output_directory)+"/output.nc")
number_replicas = 30
min_temp = 5.0 * unit.kelvin
max_temp = 100.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
print("Using "+str(len(temperature_list))+" replicas.")

###
#
# Global coarse grained model definitions
#
###
backbone_length=1
sidechain_length=1
sidechain_positions=0
include_bond_forces=False
include_bond_angle_forces=True
include_nonbonded_forces=True
include_torsion_forces=True
constrain_bonds = True

# Particle properties
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}

# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

# Bond angle properties
bond_angle_force_constant = 0.0002 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': bond_angle_force_constant,'bb_sc_sc_angle_k': bond_angle_force_constant,'sc_sc_sc_angle_k': bond_angle_force_constant, 'sc_bb_sc_angle_k': bond_angle_force_constant, 'sc_sc_bb_angle_k': bond_angle_force_constant}
equil_bond_angle = 92
equil_bond_angles = {'bb_bb_bb_angle_0': equil_bond_angle,'bb_bb_sc_angle_0': equil_bond_angle,'bb_sc_sc_angle_0': equil_bond_angle,'sc_sc_sc_angle_0': equil_bond_angle, 'sc_bb_sc_angle_0': equil_bond_angle,'sc_sc_bb_angle_0': equil_bond_angle}

# Torsion properties
torsion_force_constant = 0.0002
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': torsion_force_constant,'bb_bb_sc_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_sc_torsion_k': torsion_force_constant, 'sc_bb_bb_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_bb_torsion_k': torsion_force_constant, 'sc_sc_sc_sc_torsion_k': torsion_force_constant,  'sc_bb_bb_bb_torsion_k': torsion_force_constant}
equil_torsion_angle = 52
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': equil_torsion_angle,'bb_bb_bb_sc_torsion_0': equil_torsion_angle,'bb_bb_sc_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_bb_torsion_0': equil_torsion_angle, 'sc_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_bb_torsion_0': equil_torsion_angle}

sigma = 2.0 * bond_length
sigmas = {'bb_bb_sigma': sigma,'sc_sc_sigma': sigma}
epsilon = 0.2 * unit.kilocalorie_per_mole
epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon}

# Heteropolymer definitions
polymer_length=12

cgmodel = CGModel(polymer_length=polymer_length,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds)

C_v_list = []
dC_v_list = []

for total_simulation_time in total_simulation_time_list:
   total_steps = round(total_simulation_time.__div__(simulation_time_step))
   output_data=str(str(output_directory)+"/output"+str(total_simulation_time._value)+".nc")
   if total_steps > 10000:
     exchange_attempts = round(total_steps/1000)
   else:
     exchange_attempts = 10

   if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
     make_replica_pdb_files(cgmodel.topology,replica_positions)
   else:
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

     C_v,dC_v,new_temperature_list = get_heat_capacity(replica_energies,temperature_list,num_intermediate_states=1)
     C_v_list.append(C_v)
     dC_v_list.append(dC_v)
     

file_name = "heat_capacity.png"
figure = pyplot.figure(1)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
temperatures = np.array([temperature for temperature in new_temp_list])
legend_labels = [str(total_simulation_time._value) for total_simulation_time in total_simulation_time_list]

for C_v,dC_v in zip(C_v_list,dC_v_list):
 C_v = np.array([C_v[i] for i in range(len(C_v))])
 dC_v = np.array([dC_v[i] for i in range(len(dC_v))])
 pyplot.errorbar(temperatures,C_v,yerr=dC_v,figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
pyplot.title("Heat capacity for variable simulation time (ns)")
pyplot.legend(legend_labels)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

exit()
