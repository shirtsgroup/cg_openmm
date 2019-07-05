#!/usr/bin/python

import os, timeit
from simtk import unit
from foldamers.src.cg_model.cgmodel import basic_cgmodel
from cg_openmm.src.build.cg_build import build_mm_simulation

# OpenMM simulation settings
temperature = 300.0 * unit.kelvin
print_frequency = 20 # Number of steps to skip when printing output
total_simulation_time = 0.1 * unit.picosecond
simulation_time_step = 0.01 * unit.femtosecond

# Coarse grained model settings
backbone_length=1
sidechain_length=1
sidechain_positions=[0]
mass = unit.Quantity(10.0,unit.amu)
sigma = unit.Quantity(7.5,unit.angstrom)
bond_length = unit.Quantity(3.0,unit.angstrom)
epsilon = unit.Quantity(0.5,unit.kilocalorie_per_mole)
if not os.path.exists('bridges_test'):
  os.mkdir('bridges_test')

file = open("bridges_test/cpu_time.dat","w")
file.write("Polymer_length (# monomer units)  Simulation Time (picoseconds)  Wall Clock Time (seconds)")
file.close()

for polymer_length in [5,6,7,8]:
  cgmodel = basic_cgmodel(polymer_length=polymer_length,backbone_length=backbone_length,sidechain_length=sidechain_length,sidechain_positions=sidechain_positions,mass=mass,bond_length=bond_length,sigma=sigma,epsilon=epsilon)

  for total_simulation_time in [i * total_simulation_time for i in [1.0,2.0,5.0]]:
    number_steps = total_simulation_time.__div__(simulation_time_step)
    # Build an OpenMM simulation object
    output_pdb = str('bridges_test/cpu_'+str(polymer_length)+'_'+str(total_simulation_time._value)+'.pdb')
    output_data = str('bridges_test/cpu_'+str(polymer_length)+'_'+str(total_simulation_time._value)+'.dat')
    simulation = build_mm_simulation(cgmodel.topology,cgmodel.system,cgmodel.positions,temperature=temperature,simulation_time_step=simulation_time_step,output_pdb=output_pdb,output_data=output_data,print_frequency=print_frequency)

    # Run simulation using CPU
    cpu_start_time = timeit.default_timer()
#    simulation.step(number_steps)
    cpu_end_time = timeit.default_timer()
    cpu_time = cpu_end_time - cpu_start_time
    file = open("bridges_test/cpu_time.dat","a")
    file.write(str(polymer_length)+" "+str(total_simulation_time)+" "+str(cpu_time))
    file.close()
exit()
