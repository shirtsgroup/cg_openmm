import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.parameters.reweight import get_mbar_expectation, get_free_energy_differences
from foldamers.src.thermo.calc import calculate_heat_capacity
from cg_openmm.src.build.cg_build import build_topology
from cg_openmm.src.simulation.rep_exch import *

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)
run_simulations = True

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
temperature_list = [30.70, 32.65, 34.61, 36.59, 38.58, 40.58, 42.60, 44.63, 46.68, 48.77, 50.86, 52.95, 55.06, 57.19, 59.33, 61.49, 63.67, 65.87, 68.08, 70.32, 72.44, 74.71, 77.00, 79.30, 81.63]#, 83.97, 86.34, 88.72, 91.14, 93.57, 96.02, 98.50, 100.99, 103.51, 106.05, 108.58, 111.16, 113.77, 116.40, 119.06, 121.74, 124.45, 127.18, 129.94, 132.73, 135.51, 138.35, 141.21, 144.10, 147.03, 149.99, 152.97, 155.97, 159.02, 162.08, 165.18, 168.32, 171.48, 174.67, 177.90, 181.16, 184.45, 187.78, 191.14, 194.54, 197.97, 201.43, 205.18, 208.72, 212.30, 215.91, 219.56, 223.25, 226.98, 230.74, 234.55, 238.40, 242.29, 246.21, 250.20, 254.21, 258.26, 262.35, 266.50, 270.69, 274.92, 279.19, 283.52, 287.89, 292.30, 296.77, 301.28, 305.84, 310.44, 315.10, 319.81, 324.57, 329.38, 334.24, 339.16, 344.14, 349.16, 354.23, 359.37, 364.56, 369.80, 375.10, 380.46, 385.88, 391.36, 396.91, 400.00]
#temperature_list = [5.00, 10.76, 16.61, 22.51, 28.55, 34.71, 41.00, 47.44, 54.04, 60.81, 67.74, 74.86, 82.16, 89.41, 97.10, 105.02, 113.17, 121.57, 130.21, 139.13, 148.32, 157.79, 167.56, 177.64, 188.05, 198.92, 210.04, 221.52, 233.38, 245.64, 258.32, 271.42, 284.98, 299.00, 313.50, 328.51, 344.02, 360.09, 376.72, 393.93, 400.00]
temperature_list = [temp * unit.kelvin for temp in temperature_list]
number_replicas = len(temperature_list)
print("Using "+str(len(temperature_list))+" replicas.")
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

# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

epsilon = 0.5 * unit.kilocalorie_per_mole
epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': 0.5 * epsilon}

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

C_v_list = []
dC_v_list = []
folding_T_list = []

sigma_list = [ sigma * bond_length.unit for sigma in range(round(bond_length._value*1.5),round(bond_length._value*2.5))]
epsilon_list = [ 0.5 * epsilon * unit.kilocalorie_per_mole for epsilon in range(10)]
for sigma in sigma_list: 
 for epsilon in epsilon_list:
  print("Performing simulations and heat capacity analysis for a coarse grained model")
  print("with sigma values of "+str(sigma)+" and epsilon values of "+str(epsilon))
  sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma}
  epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': 0.5 * epsilon}
  cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds)

  # Run a replica exchange simulation with this cgmodel
  output_data = str(str(top_directory)+"/sig_"+str(sigma._value)+".nc")
  if run_simulations:
    print("Running simulations.")
    replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  else:
    replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  steps_per_stage = round(total_steps/exchange_attempts)
  plot_replica_exchange_energies(replica_energies,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)
  plot_replica_exchange_summary(replica_states,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)

  num_intermediate_states = 1
  print("Calling pymbar.")
  mbar,E_kn,E_expect,dE_expect,new_temp_list = get_mbar_expectation(replica_energies,temperature_list,num_intermediate_states)

  #print(new_temp_list)
  #print(E_kn)

  mbar,E_kn,DeltaE_expect,dDeltaE_expect,new_temp_list = get_mbar_expectation(E_kn,temperature_list,num_intermediate_states,mbar=mbar,output='differences')

  mbar,E_kn,E2_expect,dE2_expect,new_temp_list = get_mbar_expectation(E_kn**2,temperature_list,num_intermediate_states,mbar=mbar)

  df_ij,ddf_ij = get_free_energy_differences(mbar)

  print("Calculating the heat capacity.")
  C_v,dC_v = calculate_heat_capacity(E_expect,E2_expect,dE_expect,DeltaE_expect,dDeltaE_expect,df_ij,ddf_ij,new_temp_list,len(temperature_list),num_intermediate_states)
  C_v_list.append(C_v)
  dC_v_list.append(dC_v)
  folding_T_list.append(new_temp_list[np.argmax(C_v)])
  print(folding_T_list)

file_name = "heat_capacity.png"
figure = pyplot.figure(1)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
temperatures = np.array([temperature._value for temperature in new_temp_list])
legend_labels = [ str("$\sigma / r_{bond}$= "+str(round(i/bond_length._value,2)))  for i in sigma_range]

for C_v,dC_v in zip(C_v_list,dC_v_list):
 C_v = np.array([C_v[i][0] for i in range(len(C_v))])
 dC_v = np.array([dC_v[i][0] for i in range(len(dC_v))])
 pyplot.errorbar(temperatures,C_v,yerr=dC_v,figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
pyplot.title("Heat capacity for variable $\sigma / r_{bond}$")
pyplot.legend(legend_labels)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()


exit()
