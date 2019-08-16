import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.parameters.reweight import *
from foldamers.src.thermo.calc import calculate_heat_capacity
from foldamers.src.utilities.util import random_positions
from cg_openmm.src.build.cg_build import build_topology
from cg_openmm.src.simulation.rep_exch import *

grid_size = 11

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)
run_simulations = True

# OpenMM simulation settings
print_frequency = 5 # Number of steps to skip when printing output
total_simulation_time = 0.5 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas = 50
min_temp = 10.0 * unit.kelvin
max_temp = 20.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
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
polymer_length=20
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]
include_bond_forces=False
include_bond_angle_forces=False
include_nonbonded_forces=True
include_torsion_forces=False
constrain_bonds = True

# Particle properties
mass = 100.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}

# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

file = open(str("Cv_data_new.dat"),"w")
file.write("Epsilon(kCal/mol) T(Kelvin) Cv(kCal/mol*Kelvin)\n")
file.close()

file = open(str("dCv_data_new.dat"),"w")
file.write("Epsilon(kCal/mol) T(Kelvin) dCv(kCal/mol*Kelvin)\n")
file.close()

C_v_list = []
dC_v_list = []
folding_T_list = []

sigma_list = [2.65 * bond_length]
#sigma_list = [ (1.5 + i*0.1) * bond_length for i in range(grid_size)]
epsilon_list = [ unit.Quantity((2.75 + i*0.02),unit.kilocalorie_per_mole) for i in range(grid_size)]
for sigma in sigma_list: 
 for epsilon in epsilon_list:
  print("Performing simulations and heat capacity analysis for a coarse grained model")
  print("with sigma values of "+str(sigma)+" and epsilon values of "+str(epsilon))
  sigmas = {'bb_bb_sigma': sigma,'bb_sc_sigma': sigma,'sc_sc_sigma': sigma}
  epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': 0.5 * epsilon}
  cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds)

  # Run a replica exchange simulation with this cgmodel
  output_data = str(str(top_directory)+"/sig_"+str(sigma._value)+"_eps_"+str(epsilon._value)+".nc")
#  if not os.path.exists:
#    print("Running simulations.")
  skip = False
  if not skip:
   success = False
   while not success:
    try:
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
     success = True
    except:
     cgmodel.positions = random_positions(cgmodel)
     print("Simulation attempt failed, retrying.")
#  else:
#    replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

   steps_per_stage = round(total_steps/exchange_attempts)
   plot_replica_exchange_energies(replica_energies,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)
   plot_replica_exchange_summary(replica_states,temperature_list,simulation_time_step,steps_per_stage=steps_per_stage)

  if skip:
   replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)
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

  file = open(str("Cv_data_new.dat"),"a")
  for temp in range(len(new_temp_list)):
   file.write(str(round(epsilon._value,2))+" "+str(round(new_temp_list[temp],2))+" "+str(round(C_v[temp],4))+"\n")
  file.close
  file = open(str("dCv_data_new.dat"),"a")
  for temp in range(len(new_temp_list)):
   file.write(str(round(epsilon._value,2))+" "+str(round(new_temp_list[temp],2))+" "+str(round(dC_v[temp],4))+"\n")
  file.close

  C_v_list.append(C_v)
  dC_v_list.append(dC_v)
  folding_T_list.append(new_temp_list[np.argmax(C_v)])

file_name = "heat_capacity.png"
figure = pyplot.figure(1)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
temperatures = np.array([temperature for temperature in new_temp_list])
legend_labels = [ str("$\epsilon$= "+str(round(epsilon._value,2))) for epsilon in epsilon_list]

for C_v,dC_v in zip(C_v_list,dC_v_list):
 C_v = np.array([C_v[i] for i in range(len(C_v))])
 dC_v = np.array([dC_v[i] for i in range(len(dC_v))])
 pyplot.errorbar(temperatures,C_v,yerr=dC_v,figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
pyplot.title("Heat capacity for variable $\epsilon$")
pyplot.legend(legend_labels)
pyplot.xlim(min_temp._value,max_temp._value)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()


exit()
