import os
import csv
import numpy as np
import matplotlib.pyplot as pyplot
from statistics import mean
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from foldamers.cg_model.cgmodel import CGModel
from foldamers.parameters.reweight import *
from foldamers.parameters.secondary_structure import *
from foldamers.thermo.calc import *
from cg_openmm.simulation.rep_exch import *

native_structure_file = str(str(os.getcwd().split('examples/native_contacts')[0])+"ensembles/12_1_1_0/helix.pdb")

native_structure = PDBFile(native_structure_file).getPositions()

# Job settings
top_directory = 'output'
if not os.path.exists(top_directory):
  os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.nanosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data = str(str(top_directory)+"/test_output.nc")
number_replicas = 10
min_temp = 300.0 * unit.kelvin
max_temp = 350.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp,max_temp,number_replicas)
print("Using "+str(len(temperature_list))+" replicas.")

cgmodel = CGModel()

if not os.path.exists(output_data):
     replica_energies,replica_positions,replica_states = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,print_frequency=print_frequency,output_data=output_data)
else:
     print("Reading simulation data.")
     replica_energies,replica_positions,replica_states = read_replica_exchange_data(system=cgmodel.system,topology=cgmodel.topology,temperature_list=temperature_list,output_data=output_data,print_frequency=print_frequency)


print("Building an array of the fraction of native contacts for each pose.")
Q_kn = np.zeros((len(replica_positions),len(replica_positions[0])))
for replica_index in range(len(replica_positions)):
     trajectory = replica_positions[replica_index]
     for pose_index in range(len(trajectory)):
       pose = trajectory[pose_index]
       Q_kn[replica_index][pose_index] = fraction_native_contacts(cgmodel,pose,native_structure)

print("Reweighting results with MBAR.")
mbar,E_kn,result,dresult,new_temp_list = get_mbar_expectation(replica_energies,temperature_list,0)

print("Calculating free energy differences with MBAR.")
F_expect,dF_expect = get_free_energy_differences(mbar)
print("Calculating the expecation value for Q with MBAR.") 
Q_expect,dQ_expect = mbar.computeExpectations(Q_kn)

x = np.array([temperature._value for temperature in temperature_list])
y = np.array([float(Q) for Q in Q_expect])
X,Y = np.meshgrid(x,y)
Z = F_expect

figure = pyplot.figure(1)
pyplot.pcolormesh(X,Y,Z)
pyplot.colorbar()
pyplot.xlabel("T ( Kelvin )")
pyplot.ylabel("<Q> (Fraction native contacts)")
pyplot.title("Free energies for a 1-1 homo-12-mer")
pyplot.savefig("F_Q_T_test.png")
pyplot.show()
pyplot.close()
exit()
