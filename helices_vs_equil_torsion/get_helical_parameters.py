import os
import numpy as np
import matplotlib.pyplot as pyplot
from statistics import mean
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.parameters.reweight import get_mbar_expectation, get_free_energy_differences, get_temperature_list
from foldamers.src.thermo.calc import calculate_heat_capacity
from foldamers.src.parameters.secondary_structure import get_helical_parameters
from cg_openmm.src.build.cg_build import build_topology
from cg_openmm.src.simulation.rep_exch import *

grid_size = 4

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 0.2 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas = 10
min_temp = 100.0 * unit.kelvin
max_temp = 150.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

###
#
# Coarse grained model settings
#
###


pitch_list = []
radius_list = []
data_file = "helical_data.dat"
if not os.path.exists(data_file):
  data = open(data_file,"w")
  data.write("BB-BB-BB-BB Torsion (Degrees) SC-BB-BB-SC (Degrees) Pitch (Angstroms) Radius (Angstroms) Monomers-per-turn\n")
  data.close()

torsion_force_constant = 200
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': 0,'bb_bb_sc_sc_torsion_k': 0, 'bb_sc_sc_sc_torsion_k': 0, 'sc_bb_bb_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_bb_torsion_k': 0, 'sc_sc_sc_sc_torsion_k': 0,  'sc_bb_bb_bb_torsion_k': 0
}

bb_bb_bb_bb_equil_torsion_angle_range = range(-25,30,5)
bb_bb_bb_bb_equil_torsion_angles = [float(equil_torsion_angle*3.1415/180.0) for equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angle_range]
sc_bb_bb_sc_equil_torsion_angle_range = range(-25,30,5)
sc_bb_bb_sc_equil_torsion_angles = [float(equil_torsion_angle*3.1415/180.0) for equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angle_range]
equil_torsion_angle = 0.0

for bb_bb_bb_bb_equil_torsion_angle in bb_bb_bb_bb_equil_torsion_angles:
 for sc_bb_bb_sc_equil_torsion_angle in sc_bb_bb_sc_equil_torsion_angles:
  print("Performing simulations for a coarse grained model")
  print("with bb_bb_bb_bb torsion angles of "+str(round(bb_bb_bb_bb_equil_torsion_angle*180.0/3.1415,1))+" degrees")
  print("and sc_bb_bb_sc torsion angles of "+str(round(sc_bb_bb_sc_equil_torsion_angle*180.0/3.1415,1))+" degrees.")
  equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': bb_bb_bb_bb_equil_torsion_angle,'bb_bb_bb_sc_torsion_0': equil_torsion_angle,'bb_bb_sc_sc_torsion_0': equil_torsion_angle,'bb_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_sc_torsion_0': sc_bb_bb_sc_equil_torsion_angle, 'bb_sc_sc_bb_torsion_0': equil_torsion_angle, 'sc_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_bb_torsion_0': equil_torsion_angle}
  cgmodel = CGModel(equil_torsion_angles=equil_torsion_angles,torsion_force_constants=torsion_force_constants,include_bond_angle_forces=False)
  output_data = str(str(top_directory)+"/torsions_"+str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+".nc")
  if not os.path.exists(output_data):
   success = False
   while not success:
    try:
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
     success = True
    except:
     os.remove(output_data)
  else:
    replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)

  steps_per_stage = round(total_steps/exchange_attempts)

  #if not os.path.exists(output_data):

  minimum_energy_structures = get_minimum_energy_pose(cgmodel.topology,replica_energies,replica_positions)

  p_list = []
  r_list = []
  mpt_list = []
  for structure in minimum_energy_structures:
    cgmodel.positions = structure
    pitch,radius,monomers_per_turn = get_helical_parameters(cgmodel)
    p_list.append(pitch)
    r_list.append(radius)
    mpt_list.append(monomers_per_turn)
  pitch = mean(np.array([float(p) for p in p_list]))
  radius = mean(np.array([float(r) for r in r_list]))
  monomers_per_turn = mean(np.array([float(mpt) for mpt in mpt_list]))

  data = open("helical_data.dat","a")
  data.write(str(round(bb_bb_bb_bb_equil_torsion_angle,2))+"_"+str(round(sc_bb_bb_sc_equil_torsion_angle,2))+" "+str(round(float(pitch),3))+" "+str(round(float(radius),3))+" "+str(round(float(monomers_per_turn),3))+"\n")
  data.close()

file_name = str(str(top_directory)+"/heat_capacity.png")
figure = pyplot.figure(1)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
try:
  temperatures = np.array([temperature._value for temperature in new_temp_list])
except:
  temperatures = np.array([temperature for temperature in new_temp_list])

folding_temperature = []
for C_v in C_v_list:
 C_v = np.array([C_v[i][0] for i in range(len(C_v))])
 folding_temperature.append(max(C_v))

folding_temperature = np.array([temp for temp in folding_temperature])
x = np.unique([sigma._value for sigma in sigma_list])
y = np.unique([epsilon._value for epsilon in epsilon_list])
X,Y = np.meshgrid(x,y)
Z = folding_temperature.reshape(len(y),len(x))

pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Folding Temperature ( Kelvin )")
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

exit()
