import os
import numpy as np
import matplotlib.pyplot as pyplot
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from foldamers.src.thermo.calc import *
from foldamers.src.parameters.reweight import *
from cg_openmm.src.simulation.tools import *
from cg_openmm.src.simulation.rep_exch import *

# Job settings
top_directory = 'replica_exchange_output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 100 # Number of steps to skip when printing output
total_simulation_time = 0.5 * unit.nanosecond # Units = picoseconds
simulation_time_step = 0.5 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data=str(str(top_directory)+"/output.nc")
number_replicas = 30
min_temp = 5.0 * unit.kelvin
max_temp = 100.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
temperature = temperature_list[0]
print("Using "+str(len(temperature_list))+" replicas.")
if total_steps > 10000:
   exchange_attempts = round(total_steps/1000)
else:
   exchange_attempts = 10

polymer_length = 15
backbone_lengths=[1]
sidechain_lengths=[1]
sidechain_positions=[0]
include_bond_forces=False
include_bond_angle_forces=False
include_nonbonded_forces=True
include_torsion_forces=False
constrain_bonds = True

mass = 10.0 * unit.amu
masses = {'backbone_bead_masses': mass, 'sidechain_bead_masses': mass}
bond_length = 1.0 * unit.angstrom
bond_lengths = {'bb_bb_bond_length': bond_length,'bb_sc_bond_length': bond_length,'sc_sc_bond_length': bond_length}

epsilon = 0.2 * unit.kilocalorie_per_mole
epsilons = {'bb_bb_eps': epsilon,'bb_sc_eps': epsilon,'sc_sc_eps': epsilon}
sigmas = {'bb_bb_sigma': 1.0 * bond_length,'sc_sc_sigma': 2.0*bond_length}

bond_force_constant = 0 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {'bb_bb_bond_k': bond_force_constant, 'bb_sc_bond_k': bond_force_constant, 'sc_sc_bond_k': bond_force_constant}

bond_angle_force_constant = 0 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {'bb_bb_bb_angle_k': bond_angle_force_constant,'bb_bb_sc_angle_k': bond_angle_force_constant,'bb_sc_sc_angle_k': bond_angle_force_constant,'sc_sc_sc_angle_k': bond_angle_force_constant, 'sc_bb_sc_angle_k': bond_angle_force_constant, 'sc_sc_bb_angle_k': bond_angle_force_constant}
equil_bond_angle = 0
equil_bond_angles = {'bb_bb_bb_angle_0': equil_bond_angle,'bb_bb_sc_angle_0': equil_bond_angle,'bb_sc_sc_angle_0': equil_bond_angle,'sc_sc_sc_angle_0': equil_bond_angle, 'sc_bb_sc_angle_0': equil_bond_angle,'sc_sc_bb_angle_0': equil_bond_angle}

torsion_force_constant = 0
torsion_force_constants = {'bb_bb_bb_bb_torsion_k': torsion_force_constant,'bb_bb_bb_sc_torsion_k': torsion_force_constant,'bb_bb_sc_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_sc_torsion_k': torsion_force_constant, 'sc_bb_bb_sc_torsion_k': torsion_force_constant, 'bb_sc_sc_bb_torsion_k': torsion_force_constant, 'sc_sc_sc_sc_torsion_k': torsion_force_constant,  'sc_bb_bb_bb_torsion_k': torsion_force_constant}
equil_torsion_angle = 0
equil_torsion_angles = {'bb_bb_bb_bb_torsion_0': equil_torsion_angle,'bb_bb_bb_sc_torsion_0': equil_torsion_angle,'bb_bb_sc_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_sc_torsion_0': equil_torsion_angle, 'bb_sc_sc_bb_torsion_0': equil_torsion_angle, 'sc_sc_sc_sc_torsion_0': equil_torsion_angle, 'sc_bb_bb_bb_torsion_0': equil_torsion_angle}

positions = PDBFile("Fobe_helix.pdb").getPositions()

cgmodel = CGModel(polymer_length=polymer_length,backbone_lengths=backbone_lengths,sidechain_lengths=sidechain_lengths,sidechain_positions=sidechain_positions,masses=masses,sigmas=sigmas,epsilons=epsilons,bond_lengths=bond_lengths,bond_force_constants=bond_force_constants,bond_angle_force_constants=bond_angle_force_constants,torsion_force_constants=torsion_force_constants,equil_bond_angles=equil_bond_angles,equil_torsion_angles=equil_torsion_angles,include_nonbonded_forces=include_nonbonded_forces,include_bond_forces=include_bond_forces,include_bond_angle_forces=include_bond_angle_forces,include_torsion_forces=include_torsion_forces,constrain_bonds=constrain_bonds,positions=positions)

if not os.path.exists(output_data):
  replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
  make_replica_pdb_files(cgmodel.topology,replica_positions)
else:
  replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)
  make_replica_pdb_files(cgmodel.topology,replica_positions)

C_v,dC_v,new_temperature_list = get_heat_capacity(replica_energies,temperature_list,num_intermediate_states=1)

temperature_list = []
C_v_list = []
dC_v_list = []
for T_index in range(len(new_temperature_list)):
  temperature = new_temperature_list[T_index]
  if temperature < 30.0:
    temperature_list.append(temperature)
    C_v_list.append(C_v[T_index])
    dC_v_list.append(dC_v[T_index])

plot_heat_capacity(C_v_list,dC_v_list,temperature_list,file_name="Cv_low_T.png")
get_minimum_energy_pose(cgmodel.topology,replica_energies,replica_positions)

exit()
